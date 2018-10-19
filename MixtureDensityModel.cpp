#include "stdafx.h"
#include "MixtureDensityModel.h"

/*
*/

MixtureDensityModel::MixtureDensityModel(size_t _K, size_t _L) : K(_K), L(_L) {
	init();
}

MixtureDensityModel::~MixtureDensityModel()
{
}

/*	Forward function 
*	We use this to compute the conditional average of the network prediction.
*	E[ t | x]
*/
void MixtureDensityModel::conditionalMean( MAT& networkOut) {

	//Compute the sum of the weighted means
	// [(K,1)^T x (K,L)]^T = (L,1)

	//networkOut =  (getMixtureCoefficients().transpose()*getMus()).transpose(); // needs to be (L,1)
	networkOut = (param.leftCols(1).transpose()*param.rightCols(L)).transpose();
}

void MixtureDensityModel::maxMixtureCoefficient(MAT& networkOut) {

	// (1) find max mixture coefficient in O(n) time
	fREAL max = 0;
	size_t maxIndex = 0; // start with valid index
	for (size_t i = 0; i < K; ++i) {
		if (param(i, 0) > max) {
			max = param(i, 0);
			maxIndex = i;
		}
	}

	networkOut = param.block(maxIndex, 2, 1, L);
}

void MixtureDensityModel::getParameters(MAT& toCopyTo) {
	toCopyTo.resize(K, L + 2);
	toCopyTo = param;
	toCopyTo.resize(K*(L + 2), 1);
}
void MixtureDensityModel::updateParameters(MAT& networkOut) {
	assert(networkOut.size() == K*(L+2));
	networkOut.resize(K, L + 2);
	static MAT ones = MAT::Constant(K, 1, 1.0f);
	// (1) Update Mixture coefficients
	
	// (1.1) subtract the maximum output, which prevents the exponentials from easily blowing up
	// found this online on @hardmarus blog.
	param.leftCols(1) = networkOut.leftCols(1) - networkOut.leftCols(1).maxCoeff()*ones; 
	// (1.2) Now, take the exponential.
	param.leftCols(1) = param.leftCols(1).unaryExpr(&exp_fREAL); // had to explicitly define an exponential due to typing problems ;/
	fREAL norm = param.leftCols(1).sum();
	param.leftCols(1) /= norm; // normalize
	// (2) Update the mean values.
	param.rightCols(L) = networkOut.rightCols(L);
	// (3) Update the stds
	param.block(0, 1, K, 1) = networkOut.block(0, 1, K, 1).unaryExpr(&exp_fREAL);
	// (4) leave the matrix as it was
	networkOut.resize(K*(L + 2),1);
}

// Initialization routine.
void MixtureDensityModel::init() {

	param = MAT(K, L + 2);
	// Initialize Mixture coefficients to something.
	param.setZero();
	// they will be overwritten by the network outputs immediately...
}

MATBLOCK MixtureDensityModel::getMixtureCoefficients()  const {
	return  param.block(0, 0, K, 1); // implicit cast, but fine
}

MATBLOCK MixtureDensityModel::getMus() const {
	return param.block(0, 2, K, L);
}

MATBLOCK MixtureDensityModel::getSpecificMu(size_t i) const {
	assert(i < K);
	return param.block(i, 2, 1, L);
}

MATBLOCK MixtureDensityModel::getVariances() const {
	// TODO: insert return statement here
	return param.block(0, 1, K, 1); // ensure it's not zero
}

fREAL MixtureDensityModel::getSpecificVariance(size_t k) const
{
	return param(k, 1);
}


/*	Compute derivative of error function
	wRt cross-entropy error.
*/
MAT MixtureDensityModel::computeErrorGradient(const MAT& t) {
	static const MAT epsilons = MAT::Constant(K, 1, 1E-8);
	static const fREAL epsilon = 1E-8;
	static const MAT Ls = MAT::Constant(K, 1, L);

	MAT errorGrad(K, L + 2);

	// (1) Compute Gamma matrix as posterior 
	// gamma_k ( t | x ) = pi_k * gauss(t | mu_k, var_k) * NORM
	static MAT gamma(K,1); // bayesian prior - we define it here  as static variable so that we can multithread over it

	fREAL normCoeff = 0;
	int32_t i = 0;
	#pragma omp parallel for private(i) shared(gamma, normCoeff)
	for (i = 0; i < K; ++i) {
		gamma(i,0) = param(i,0)*normalDistribution(t, param.block(i, 2, 1, L).transpose(), epsilon+ param(i,1)*param(i, 1));
		#pragma omp critical
		normCoeff += gamma(i,0);
	}
	gamma /= normCoeff;  // normalize

	//(2) Go through the different parts of the parameter matrix 
	// and compute the associated error.
	
	// (2.1) Mixture Density Coefficient error
	errorGrad.leftCols(1) = param.leftCols(1) - gamma;
	
	// (2.2) Mu Error gamma_k * (mu_k - t)
	// Note, that gamma is (K,1)
	// t is (L,1)
	// and mu is (K,L)
	MAT temp = param.rightCols(L) - t.transpose().replicate(K, 1);


	// temp is used later again and is a (K,L) matrix
	
	errorGrad.rightCols(L) = (gamma.cwiseQuotient( param.block(0,1,K,1).unaryExpr(&norm) + epsilons)).replicate(1,L); // make it a row vector
	errorGrad.rightCols(L) =  errorGrad.rightCols(L).cwiseProduct(temp);
	
	// (2.3) Variance Error
	// The variance is a single column in the param & error-gradient matrix
	// of length K 
	// collapse temp rowwise and take the squared norm
	temp = (temp.unaryExpr(&norm)).rowwise().sum(); //|| mu_k - t|| ^2 -> (K,1) again
	// the gradient wRt to the variance reads gamma_k * (L - || t - mu_k || ^2 / var_k^2 ) 
	//errorGrad.block(0, 1, K, 1) = -gamma.cwiseProduct(temp.cwiseQuotient(param.block(0, 1, K, 1).unaryExpr(&cube) + epsilons));
	//errorGrad.block(0, 1, K, 1) += gamma.cwiseQuotient(param.block(0, 1, K, 1) + epsilons);
	errorGrad.block(0, 1, K, 1) = gamma.cwiseProduct(Ls - temp.cwiseQuotient(param.block(0, 1, K, 1).unaryExpr(&norm)+epsilons ));
	assert(errorGrad.allFinite());

	errorGrad.resize(K*(L + 2), 1);
	return errorGrad; 
}

fREAL MixtureDensityModel::negativeLogLikelihood(const MAT& t) const
{
	const fREAL epsilon = 1E-14;
	fREAL result = 0;
	int32_t i = 0;
	double temp = 0;
	#pragma omp parallel for private(i, temp) shared(result)
	for (i = 0; i < K; ++i) {
		temp = ((double) param(i, 0)) * (double) normalDistribution(t, param.block(i, 2, 1, L).transpose(), param(i, 1)*param(i, 1)+epsilon);
		#pragma omp critical
		result += temp;
	}
	
	return -log(result);

}
