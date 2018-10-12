#include "stdafx.h"
#include "MixtureDensityModel.h"

/*
*/
typedef Eigen::Block<MAT, -1 - 1, false> MATBLOCK;
typedef Eigen::DenseBase<fREAL> DENSEMAT;

MixtureDensityModel::MixtureDensityModel(size_t _K, size_t _L) : K(_K), L(_L) {
	init();
}

/*	Forward function 
*	We use this to compute the conditional average of the network prediction.
*	E[ t | x]
*/
MAT MixtureDensityModel::conditionalMean(const MAT& networkOut)
{
	updateParameters(networkOut);
	//Compute the sum of the weighted means
	// [(K,1) x (K,L)]^T = (L,1)
	return (getMixtureCoefficients()*getMus()).transpose(); // needs to be (L,1)
}

void MixtureDensityModel::updateParameters(const MAT& networkOut) {
	// (1) Update Mixture coefficients
	param.block(0, 0, K, 1) = networkOut.block(0, 0, K, 1).unaryExpr(&exp_fREAL); // had to explicitly define an exponential due to typing problems ;/
	param.block(0, 0, K, 1) /= param.block(0, 0, K, 1).sum(); // normalize
	// (2) Update the mean values.
	param.block(0, 2, K, L) = networkOut.block(0, 2, K, L);
	// (3) Update the variances
	param.block(0, 1, K, 1) = param.block(0, 1, K, 1).unaryExpr(&exp_fREAL);
}

// Initialization routine.
void MixtureDensityModel::init() {

	param = MAT(K, L + 2);
	param.setZero();
}

const MAT& MixtureDensityModel::getMixtureCoefficients() const {
	return  param.block(0, 0, K, 1); // implicit cast, but fine
}

const MAT& MixtureDensityModel::getMus() const {
	return param.block(0, 2, K, L);
}

const MAT& MixtureDensityModel::getSpecificMu(size_t i) const {
	assert(i < K);
	return param.block(i, 2, 1, L);
}

const MAT & MixtureDensityModel::getVariances() const
{
	// TODO: insert return statement here
	return param.block(0, 1, K, 1);
}

fREAL MixtureDensityModel::getSpecificVariance(size_t k) const
{
	return param(k, 1);
}


/*	Compute derivative of error function
	wRt cross-entropy error.
*/
MAT MixtureDensityModel::computeErrorGradient(const MAT& t) const {

	MAT errorGrad(K, L + 2);
	// (1) Compute Gamma matrix as posterior 
	// gamma_k ( t | x ) = pi_k * gauss(t | mu_k, var_k) * NORM
	
	MAT gamma(K,1);

	fREAL normCoeff = 0;
	int32_t i = 0;
	#pragma omp parallel for private(i) shared(gamma)
	for (i = 0; i < K; ++i) {
		gamma(i,0) = normalDistribution(t, getSpecificMu(i).transpose(), getSpecificVariance(i));
		normCoeff += gamma(i,0);
	}
	gamma /= normCoeff;  // normalize

	//(2) Go through the different parts of the parameter matrix 
	// and compute the associated error.
	
	// (2.1) Mixture Density Coefficient error
	//errorGrad.block(0, 0, K, 1) = getMixtureCoefficients() - gamma;
	errorGrad.block(0,0,K,1) = getMixtureCoefficients() - gamma;

	// (2.2) Mu Error gamma_k * (mu_k - t)
	// Note, that gamma is (K,1)
	// t is (L,1)
	// and mu is (K,L)
	MAT temp = getMus() - t.transpose().replicate(K, 1);
	// temp is used later again and is a (K,L) matrix
	
	errorGrad.block(0, 2, K, L) = (gamma.cwiseQuotient(getVariances())).transpose().replicate(1,L); // make it a row vector
	errorGrad.block(0, 2, K, L) = errorGrad.block(0, 2, K, L).cwiseProduct(temp);
	
	// (2.3) Variance Error
	// The variance is a single column in the param & error-gradient matrix
	// of length K 
	static const MAT Ls = MAT::Constant(K, 1, L);

	// collapse temp rowwise and take the squared norm
	temp = temp.rowwise().sum();
	// the gradient wRt to the variance reads gamma_k * (L - || t - mu_k || ^2 / var_k^2 ) 
	errorGrad.block(0, 1, K, 1) = gamma.cwiseProduct(Ls - temp.unaryExpr(&norm).cwiseQuotient(getVariances()));
	
	return errorGrad; 
}

fREAL MixtureDensityModel::negativeLogLikelihood(const MAT& t) const
{
	fREAL result = 0;
	int32_t i = 0;
	#pragma omp parallel for private(i)
	for (i = 0; i < K; ++i) {
		result += param(i, 0)*normalDistribution(t, getSpecificMu(i).transpose(), getSpecificVariance(i));
	}
	return -log(result);
}
