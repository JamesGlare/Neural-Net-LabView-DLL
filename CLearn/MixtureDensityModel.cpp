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
// after resizing the mu-part to (NOUTY, NOUTX*K)
// Enumerate blocks along column (y) direction
// B1 B3 
// B2 B4

#define __MUS2D(b,k)	block(_BLOCKY(b)*BlockY, _BLOCKX(b)*BlockX+k*NOUTX, BlockY, BlockX)
#define __MU2D(k)		block(0, k*NOUTX, NOUTY, NOUTX)
#define __ALLMUS		topRows(L*Blocks*K)
#define __SIGMA(b,k)	operator()(k,Blocks*LBlock+b)
#define __SIGMACOL(b)	block(0, Blocks*LBlock+b,K,1)
#define __SIGMAS		block(0,Blocks*LBlock,K,Blocks)
#define __PI(b,k)		operator()(k,Blocks*(LBlock+1)+b)
#define __PICOL(b)		block(0,Blocks*(LBlock+1)+b,K,1)
#define __PIS			block(0, Blocks*(LBlock + 1), K, Blocks)


MixtureDensityModel::MixtureDensityModel(size_t _NOUTY, size_t _NOUTX, size_t _features, size_t _BlockY, size_t _BlockX, CNetLayer& lower) :
	K(_features), L(_NOUTX*_NOUTY), NOUTX(_NOUTX), NOUTY(_NOUTY), Blocks((_NOUTX/_BlockX)*(_NOUTY/_BlockY)), LBlock(_BlockX*_BlockY), BlockX(_BlockX), BlockY(_BlockY), 
	DiscarnateLayer(_NOUTX*_NOUTY, actfunc_t::NONE, lower ) {
	init();
	// HACK - in C++ access control works on a per-class basis rather than on per-object basis.
	changeActFunc(lower, actfunc_t::NONE);
	assert(getNIN() == Blocks*K*(LBlock + 2)); // OR K*(L+2*Blocks)
	assert(LBlock*Blocks == L);
	assert(BlockX*BLocks == NOUTX);
	assert(BlockY*Blocks == NOUTY);
	// Blocks*K*(LBlock+2) == K*Blocks* NOUTX*NOUTY/Blocks + 2 * Blocks*K == K*NOUTX*NOUTY+ 2*Blocks*K
}
MixtureDensityModel::MixtureDensityModel(size_t _NOUTX, size_t _NOUTY, size_t _features, size_t _BlockX, size_t _BlockY) :
	K(_features), L(_NOUTX*_NOUTY), NOUTX(_NOUTX), NOUTY(_NOUTY), Blocks((_NOUTX / _BlockX)*(_NOUTY / _BlockY)), LBlock(_BlockX*_BlockY), BlockX(_BlockX), BlockY(_BlockY), 
	DiscarnateLayer(_NOUTX*_NOUTY, Blocks*K*(LBlock + 2), actfunc_t::NONE) {
	init();
}

// Initialization routine.
void MixtureDensityModel::init() {


	MU = MAT(L*K,1); // need to resize of course when we operate in 2D space
	PI = MAT(Blocks*K,1);
	SIG = MAT(Blocks*K,1 );

	// Initialize Mixture coefficients to something.
	MU.setZero();
	PI.setOnes();
	SIG.setOnes();
	// they will be overwritten by the network outputs immediately...

	// L -> NOUT = NOUTX*NOUTY
	// if several features are incoming, you have (NOUTY, features*NOUTX)
	// block - should correspond to the number of separate incoming blocks
	// e.g. deconvolution outputs.
	// We need to reorder the mu-part of the input accordingly.
	// K == Features

}
void MixtureDensityModel::saveToFile(ostream & os) const
{
}
void MixtureDensityModel::loadFromFile(ifstream & in)
{
}
MixtureDensityModel::~MixtureDensityModel()
{
}

layer_t MixtureDensityModel::whoAmI() const {
	return layer_t::mixtureDensity;
}
void MixtureDensityModel::forProp(MAT & in, bool saveActivation, bool recursive)
{
	updateParameters(in);
	// Choose what to ouput - 2 choices: Conditional mean of all modes
	// or the mode with the highest likelihood.
	maxMixtureCoefficient(in); // Size: (L,1)

	// Save the output - we need it if we backprop later...
	if (saveActivation)
		actSave = in;

	// Propagate the max mixture coefficient OR conditional mean to the next layer
	if (getHierachy() != hierarchy_t::output && recursive) {
		above->forProp(in, saveActivation, true);
	}
}

void MixtureDensityModel::backPropDelta(MAT & delta, bool recursive) {
	deltaSave = delta;
	if (getHierachy() != hierarchy_t::input) { // ... should be true
		MAT t = reconstructTarget(delta);
		delta = computeErrorGradient(t);
		if (recursive)
			below->backPropDelta(delta, true);
	} 
}

/*	Forward function 
*	We use this to compute the conditional average of the network prediction.
*	E[ t | x]
*/
void MixtureDensityModel::conditionalMean( MAT& networkOut) {
	networkOut.resize(NOUTY, NOUTX); // L == Blocks*LBlock
	MU.resize(NOUTY, K*NOUTX);
	PI.resize(Blocks, K);
	static MAT block(BlockY, BlockX);

	//Compute the sum of the weighted means
	// [(K,1)^T x (K,LBLock)]^T = (LBlock,1)
	for (size_t b = 0; b < Blocks; ++b) {
		block = PI(b, 0)*MU.__MUS2D(b, 0);
		for (size_t k = 1; k < K; ++k) {
			block += PI(b, k) * MU.__MUS2D(b, k);
		}
		networkOut.__MUS2D(b,0) = block;
	}
	// Resize everything back to how it was
	MU.resize(L*K, 1);
	PI.resize(Blocks*K, 1);
	networkOut.resize(L, 1);
}

void MixtureDensityModel::maxMixtureCoefficient(MAT& networkOut) {

	networkOut.resize(NOUTY, NOUTX); 
	MU.resize(NOUTY, K*NOUTX);
	PI.resize(Blocks, K);

	for (size_t b = 0; b < Blocks; ++b) {
		// (1) find max mixture coefficient in O(n) time
		fREAL max = 0.0f;
		size_t maxK = -1; // start with valid index

		fREAL temp = 0.0f;
		for (size_t k = 0; k < K; ++k) {
			temp = PI(b, k); //  /SIG(b, k)
			if (temp > max) {
				max = temp;
				maxK = k;
			}
		}

		networkOut.__MUS2D(b,0) = MU.__MUS2D(b,maxK);
	}
	MU.resize(L*K, 1);
	PI.resize(Blocks*K, 1);
	networkOut.resize(L, 1);
}

void MixtureDensityModel::getParameters(MAT& toCopyTo) {
	
	// (1) resize & copy MU
	//MU.resize(NOUTY*NOUTX*K, 1);
	toCopyTo.topRows(K*L) = MU;
	//MU.resize(NOUTY, NOUTX*K);
	
	// (2) Resize & copy sigmaS
	//SIG.resize(Blocks*K, 1);
	toCopyTo.block(L*K, 0, K*Blocks, 1) = SIG;
	//SIG.resize(Blocks, K);

	// (3) Resize & copy pis
	//PI.resize(Blocks*K, 1);
	toCopyTo.bottomRows(K*Blocks) = PI;
	//PI.resize(Blocks, K);
}
void MixtureDensityModel::updateParameters(MAT& networkOut) {
	
	// (1) Resize & copy MU
	MU = networkOut.topRows(L*K);
	
	// (2) Sigmas need to be expoenential of network output
	SIG = networkOut.block(L*K, 0, K*Blocks, 1).unaryExpr(&exp_fREAL);

	// (3) Slightly more complicated: Mixture cofficients
	PI = networkOut.bottomRows(K*Blocks).unaryExpr(&exp_fREAL);
	PI.resize(Blocks, K);
	// (3.1) Normalize each block over all feature-mixture-coefficients
	fREAL normCoeff = 0;
	for (size_t b = 0; b < Blocks; ++b) {

		normCoeff = PI.row(b).sum();
		PI.row(b) /= normCoeff;
	}
	// (3.2) resize back 
	PI.resize(Blocks*K, 1);
}

/*	Compute derivative of error function
	wRt cross-entropy error.
*/
MAT MixtureDensityModel::computeErrorGradient(MAT& t) {

	static const fREAL epsilon = 1E-14;
	static MAT GAMMA(Blocks, K);
	MAT errorGrad(K*Blocks*(LBlock + 2),1);
	
	static MAT errorGrad_MU(NOUTY, NOUTX*K);
	static MAT errorGrad_SIG(Blocks, K);
	static MAT errorGrad_PI(Blocks, K);

	// Resize matrices to make submatrix selection easier (or even possible)
	t.resize(NOUTY, NOUTX); //  target 
	MU.resize(NOUTY, NOUTX*K); // MUS
	SIG.resize(Blocks, K); // Standard deviations
	PI.resize(Blocks, K); // Mixture Coefficients
	int32_t b = 0;
	// FOR EACH BLOCK... 
	//#pragma omp parallel for private(b) shared(t,MU,SIG,PI, errorGrad_MU, errorGrad_PI, errorGrad_SIG)
	for (b = 0; b < Blocks; ++b) {

		MAT tBlock = t.__MUS2D(b, 0); // output has only single feature
		
		// (1) Compute GAMMA Matrix
		fREAL normCoeff = 0.0f;
		for (size_t k = 0; k < K; ++k) {
			GAMMA(b, k) = PI(b, k)*normalDistribution( tBlock, MU.__MUS2D(b, k), SIG(b,k)*SIG(b, k));
			normCoeff += GAMMA(b, k);
		}
		// Normalize GAMMA
		GAMMA.row(b) /= normCoeff;

		// (2) Compute PI Gradient
		errorGrad_PI.row(b) = PI.row(b) - GAMMA.row(b);

		fREAL var = 1.0f;
		for (size_t k = 0; k < K; ++k) {
			var = SIG(b, k)*SIG(b, k);
			// (3) Compute MU Gradient
			errorGrad_MU.__MUS2D(b, k) = GAMMA(b, k) /var *(MU.__MUS2D(b,k) - tBlock);
			// (4) Compute SIG Gradient
			errorGrad_SIG(b, k) = -GAMMA(b, k) * ((MU.__MUS2D(b, k) - tBlock).squaredNorm() / var - L);
		}
	}

	// (5) Combine into errorGrad Matrix 
	// (5.1) PIS
	errorGrad_PI.resize(Blocks*K, 1);
	errorGrad.bottomRows(Blocks*K) = errorGrad_PI;
	errorGrad_PI.resize(Blocks,K);

	// (5.2) MUS
	errorGrad_MU.resize(L*K, 1);
	errorGrad.topRows(L*K) = errorGrad_MU;
	errorGrad_MU.resize(NOUTY, NOUTX*K);
	
	// (5.3) Sigma
	errorGrad_SIG.resize(Blocks*K, 1);
	errorGrad.block(L*K, 0, Blocks*K, 1) = errorGrad_SIG;
	errorGrad_SIG.resize(Blocks,K);

	// (6) Resize t, MU, SIG & PI
	MU.resize(L*K, 1);
	SIG.resize(Blocks*K, 1);
	PI.resize(Blocks*K, 1);
	t.resize(L, 1);

	return errorGrad; 
}

MAT MixtureDensityModel::reconstructTarget(const MAT & diffMatrix)
{
	assert(diffMatrix.size() == getNOUT());
	return actSave - diffMatrix; // target = estimate - delta
}

fREAL MixtureDensityModel::negativeLogLikelihood(MAT& t)
{
	const fREAL epsilon = 1E-14;
	fREAL result = 0;
	fREAL blockResult = 0;
	int32_t i = 0;
	fREAL temp = 0;

	t.resize(NOUTY, NOUTX);
	PI.resize(Blocks, K);
	MU.resize(NOUTY, NOUTX*K);
	SIG.resize(Blocks, K);

	for (size_t b = 0; b < Blocks; ++b) {
		blockResult = 0;
		MAT tBlock = t.__MUS2D(b, 0);
		//#pragma omp parallel for private(i, temp) shared(result)
		for (size_t k = 0; k < K; ++k) {
			temp = PI(b,k) *  normalDistribution(tBlock, MU.__MUS2D(b,k), SIG(b,k)*SIG(b,k));
			//#pragma omp critical
			blockResult += temp;
		}
		result -= 1.0f/Blocks*log(blockResult);
	}
	// Resize everything back
	t.resize(L, 1);
	PI.resize(Blocks*K, 1);
	SIG.resize(Blocks, K);
	MU.resize(L*K, 1);

	return result;
}
