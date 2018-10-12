#pragma once

#include "defininitions.h"

#ifndef CNET_MIXTUREDENSITY
#define CNET_MIXTUREDENSITY
/**
	MixtureDensityModel (Bishop 1994)
	The network is used to predict means and variances for a series of normal distributions
	which predict the probability of a target variable having a certain value.

	Like in other parts of this library, we want to transport these parameters
	in a single matrix. Therefore, we have to agree where in the matrix we store what.
	There are
		(1) K mixture coefficients
		(2) L x K components of all the mean-vectors (mus)
		(3) K components of the variance matrix (assume spherical Gaussian -> single variance)

	All these components are transmitted in a single matrix of size 
	K x (L+2 ) according to the following format:
	Columns 
	0			1				2			...		L + 1
	Pi_0		var_0			Mu_{0,0}			Mu_{0,L-1}
	Pi_1		var_1			Mu_{1,0}			Mu_{1,L-1}
	.			.				.					.
	.			.				.					.
	.			.				.					.
	Pi_(K-1)	var_(K-1)		Mu_{K-1,0} ...		Mu_{K-1, L-1}

	The parameters are stored in the param matrix.
**/

class MixtureDensityModel {
public:
	MixtureDensityModel(size_t _K, size_t L);
	MAT conditionalMean(const MAT& networkOut);
	MAT computeErrorGradient(const MAT& t) const;
	fREAL negativeLogLikelihood(const MAT& t) const;
	inline size_t getNOUT() const { return K*(L + 2); };

private:

	size_t K; // number of mixture coefficients
	size_t L; // just for convenience - dimension of network output
	
	
	MAT param; // Matrix(K,L) of kernel centres mu_j. For each mixture kernel, a scalar variance. We assume the gaussian to be spherical.
				// Mixture coefficients - sum to one.
	void init();
	void updateParameters(const MAT& networkOut);
	// Private getter functions for insertion into other functions
	// return rvalue references
	const MAT& getMixtureCoefficients() const ;
	const MAT& getMus() const;
	const MAT& getSpecificMu( size_t k) const;
	const MAT& getVariances() const;
	fREAL getSpecificVariance( size_t k) const;

};

#endif
