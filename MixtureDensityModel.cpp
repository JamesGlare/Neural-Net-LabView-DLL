#include "stdafx.h"
#include "MixtureDensityModel.h"

/*	L - total NOUT
*	K - Number of mixture densities PER block
*	Blocks - Number of blocks.
*	LBlock - Length of block in x/y direction.
*/

/* Example
*	L = 16
*	K = 3
*	Blocks = 4
*	LBlock = 16/4 = 4
*	NIN must be = Blocks* K * (LBlock+2) = 4*3*(8+2) = 12 * 10 = 120
*/

// Need to use Macros in this case to make the code more readable.
// Can't return blocks as lvalues from function

#define __MU(b,k)		block(k,b*LBlock,1, LBlock) // Blocks are zero-based as everything else! b in (0, ..., Blocks-1)
#define __MUS(b)		block(0, b*LBlock, K, LBlock)
#define __SIGMA(b,k)	operator()(k,Blocks*LBlock+b)
#define __SIGMACOL(b)	block(0, Blocks*LBlock+b,K,1)
#define __SIGMAS		block(0,Blocks*LBlock,K,Blocks)
#define __PI(b,k)		operator()(k,Blocks*(LBlock+1)+b)
#define __PICOL(b)		block(0,Blocks*(LBlock+1)+b,K,1)
#define __PIS			block(0, Blocks*(LBlock + 1), K, Blocks)


MixtureDensityModel::MixtureDensityModel(size_t _K, size_t _L, size_t _Blocks, size_t NIN) : K(_K), L(_L), Blocks(_Blocks), LBlock(_L/ _Blocks  ) {
	init();
	assert(NIN == Blocks*K*(LBlock + 2));
}
// Initialization routine.
void MixtureDensityModel::init() {

	param = MAT(K, (LBlock + 2)*Blocks);
	// Initialize Mixture coefficients to something.
	param.setZero();
	// they will be overwritten by the network outputs immediately...
}
MixtureDensityModel::~MixtureDensityModel()
{
}

/*	Forward function 
*	We use this to compute the conditional average of the network prediction.
*	E[ t | x]
*/
void MixtureDensityModel::conditionalMean( MAT& networkOut) {
	networkOut = MAT(L, 1); // L == Blocks*LBlock

	//Compute the sum of the weighted means
	// [(K,1)^T x (K,LBLock)]^T = (LBlock,1)
	for (size_t b = 0; b < Blocks; ++b) {
		networkOut.block(b*LBlock, 0, LBlock, 1) = (param.__PICOL(b).transpose() * param.__MUS(b)).transpose();
	}
}

void MixtureDensityModel::maxMixtureCoefficient(MAT& networkOut) {

	networkOut = MAT(L, 1); // L == Blocks*LBlock

	for (size_t b = 0; b < Blocks; ++b) {
		// (1) find max mixture coefficient in O(n) time
		fREAL max = 0;
		size_t maxK = -1; // start with valid index

		fREAL temp = 0;
		for (size_t k = 0; k < K; ++k) {
			temp = param.__PI(b, k); //  / param.__SIGMA(b, k)
			if (temp > max) {
				max = temp;
				maxK = k;
			}
		}

		networkOut.block(b*LBlock, 0, LBlock, 1) = param.__MU(b,maxK).transpose();
	}

}

void MixtureDensityModel::getParameters(MAT& toCopyTo) {
	toCopyTo.resize(K, (LBlock + 2)*Blocks);
	toCopyTo = param;
	toCopyTo.resize(K* (LBlock + 2)*Blocks, 1);
}
void MixtureDensityModel::updateParameters(MAT& networkOut) {
	assert(networkOut.size() == K * (LBlock + 2)*Blocks);
	networkOut.resize(K, (LBlock + 2)*Blocks);
	static MAT ones = MAT::Constant(K, 1, 1.0f);
	
	
	fREAL normCoeff = 0;
	int32_t b = 0;
	//#pragma omp parallel for private(b)
	for (size_t b = 0; b < Blocks; ++b) {
	
		// (1) UPDATE MEAN VALUEs

		param.__MUS(b) = networkOut.__MUS(b);
	
		// (2) UPDATE THE SIGMAS

		param.__SIGMACOL(b) = networkOut.__SIGMACOL(b).unaryExpr(&exp_fREAL);

		// (3) UPDATE THE PIS

		normCoeff = 0;
		param.__PICOL(b) = networkOut.__PICOL(b); //  - networkOut.__PICOL(b).maxCoeff()*ones
		param.__PICOL(b) = param.__PICOL(b).unaryExpr(&exp_fREAL);
		normCoeff = param.__PICOL(b).sum();
		param.__PICOL(b) /= normCoeff;
	}
	networkOut.resize(K*Blocks*(LBlock + 2), 1);
	/* OLD - SINGLE BLOCK CODE
	// (1.1) subtract the maximum output, which prevents the exponentials from easily blowing up
	// found this online on @hardmarus blog.
	param.rightCols(1) = networkOut.rightCols(1); //  -networkOut.rightCols(1).maxCoeff()*ones; 
	// (1.2) Now, take the exponential.
	param.rightCols(1) = param.rightCols(1).unaryExpr(&exp_fREAL); // had to explicitly define an exponential due to typing problems ;/
	fREAL norm = param.rightCols(1).sum();
	param.rightCols(1) /= norm; // normalize
	// (2) Update the mean values.
	param.leftCols(L) = networkOut.leftCols(L);
	// (3) Update the stds
	param.block(0, L, K, 1) = networkOut.block(0, L, K, 1).unaryExpr(&exp_fREAL);
	// (4) leave the matrix as it was
	networkOut.resize(K*(L + 2),1); */
}

/*	Compute derivative of error function
	wRt cross-entropy error.
*/
MAT MixtureDensityModel::computeErrorGradient(const MAT& t) {

	static const MAT epsilons = MAT::Constant(K, 1, 1E-8);
	static const fREAL epsilon = 1E-14;
	static const MAT Ls = MAT::Constant(K, 1, LBlock);
	static MAT gamma(K, Blocks);
	MAT errorGrad(K, Blocks*(LBlock + 2));

	// The computation of the gradient can be split up 
	// and parallelized over all blocks, since they should be
	// completely independent.

	//#pragma omp parallel for shared( gamma) private(b)
	for (size_t b = 0; b < Blocks; ++b) {
		// split up the incoming target vector in blocks 
		// note the column-major format.
		// A 2x 2 matrix would split up in the following way
		//  0	2
		//	1	3

		const MAT& tBlock = t.block(b*LBlock, 0, LBlock, 1);
		fREAL normCoeff = 0;

		// (1) COMPUTE GAMMA MATRIX

		for (size_t k = 0; k < K; ++k) {
			gamma(k, b) = param.__PI(b, k) * normalDistribution(tBlock, param.__MU(b, k).transpose(), param.__SIGMA(b, k) * param.__SIGMA(b, k));
			normCoeff += gamma(k, b);
		}
		gamma.block(0, b, K, 1) /= normCoeff;

		// (2) COMPUTE PI GRADIENT

		errorGrad.__PICOL(b) = param.__PICOL(b) - gamma.block(0, b, K, 1);

		// (3) COMPUTE MU GRADIENT
		MAT t_mu_diff = param.__MUS(b) - tBlock.transpose().replicate(K, 1); // (K, LBlock) Matrix
		MAT sigmaColSquare = (param.__SIGMACOL(b)).unaryExpr(&norm);

		errorGrad.__MUS(b) = gamma.block(0, b, K, 1).cwiseQuotient(sigmaColSquare).replicate(1, LBlock);//(gamma.block(0, b, K, 1).cwiseQuotient(param.__SIGMACOL(b).unaryExpr(&norm))).replicate(1, LBlock); // make it a row vector
		errorGrad.__MUS(b) = errorGrad.__MUS(b).cwiseProduct(t_mu_diff);

		// (4) COMPUTE SIGMA GRADIENT

		t_mu_diff = (t_mu_diff.unaryExpr(&norm)).rowwise().sum(); //|| mu_k - t|| ^2 -> (K,1) again

		errorGrad.__SIGMACOL(b) = gamma.block(0, b, K, 1).cwiseProduct(Ls - t_mu_diff.cwiseQuotient(sigmaColSquare));
	}


	errorGrad.resize(Blocks*K*(LBlock + 2), 1);
	return errorGrad; 

	/*

	// (1) Compute Gamma matrix as posterior 
	// gamma_k ( t | x ) = pi_k * gauss(t | mu_k, var_k) * NORM
	static MAT gamma(K,1); // bayesian prior - we define it here  as static variable so that we can multithread over it

	fREAL normCoeff = 0;
	int32_t i = 0;
	//#pragma omp parallel for private(i) shared(gamma, normCoeff)
	for (i = 0; i < K; ++i) {
		gamma(i,0) = param(i,L+1)*normalDistribution(t, param.block(i, 0, 1, L).transpose(),  param(i,L)*param(i, L));
		//#pragma omp critical
		normCoeff += gamma(i,0);
	}
	gamma /= normCoeff;  // normalize

	//(2) Go through the different parts of the parameter matrix 
	// and compute the associated error.
	
	// (2.1) Mixture Density Coefficient error
	errorGrad.rightCols(1) = param.rightCols(1) - gamma;
	
	// (2.2) Mu Error gamma_k * (mu_k - t)
	// Note, that gamma is (K,1)
	// t is (L,1)
	// and mu is (K,L)
	MAT temp = param.leftCols(L) - t.transpose().replicate(K, 1);


	// temp is used later again and is a (K,L) matrix
	
	errorGrad.leftCols(L) = (gamma.cwiseQuotient( param.block(0,L,K,1).unaryExpr(&norm)  )).replicate(1,L); // make it a row vector
	errorGrad.leftCols(L) =  errorGrad.leftCols(L).cwiseProduct(temp);
	
	// (2.3) Variance Error
	// The variance is a single column in the param & error-gradient matrix
	// of length K 
	// collapse temp rowwise and take the squared norm
	temp = (temp.unaryExpr(&norm)).rowwise().sum(); //|| mu_k - t|| ^2 -> (K,1) again
	// the gradient wRt to the variance reads gamma_k * (L - || t - mu_k || ^2 / var_k^2 ) 
	//errorGrad.block(0, 1, K, 1) = -gamma.cwiseProduct(temp.cwiseQuotient(param.block(0, 1, K, 1).unaryExpr(&cube) + epsilons));
	//errorGrad.block(0, 1, K, 1) += gamma.cwiseQuotient(param.block(0, 1, K, 1) + epsilons);
	errorGrad.block(0, L, K, 1) = gamma.cwiseProduct(Ls - temp.cwiseQuotient(param.block(0, L, K, 1).unaryExpr(&norm) ));
	assert(errorGrad.allFinite());

	errorGrad.resize(K*(L + 2), 1);
	return errorGrad; */
}

fREAL MixtureDensityModel::negativeLogLikelihood(const MAT& t) const
{
	const fREAL epsilon = 1E-14;
	fREAL result = 0;
	fREAL blockResult = 0;
	int32_t i = 0;
	fREAL temp = 0;
	for (size_t b = 0; b < Blocks; ++b) {
		blockResult = 0;
		MAT tBlock = t.block(b*LBlock, 0, LBlock, 1);
		//#pragma omp parallel for private(i, temp) shared(result)
		for (size_t k = 0; k < K; ++k) {
			temp = param.__PI(b,k) *  normalDistribution(tBlock, param.__MU(b,k).transpose(), param.__SIGMA(b, k)*param.__SIGMA(b, k));
			//#pragma omp critical
			blockResult += temp;
		}
		result -= 1.0f/Blocks*log(blockResult);
	}
	return result;
}
