#pragma once

#include "defininitions.h"
#include "DiscarnateLayer.h"
#ifndef CNET_MIXTUREDENSITY
#define CNET_MIXTUREDENSITY
/**
	MixtureDensityModel (Bishop 1994)
	---------------------------------
	The network is used to predict means and variances for a series of normal distributions
	which predict the probability of a target variable having a certain value.

	We change the model slightly and perform the mixture density operation only over
	a square block of the output. This way, we have completely independent predictions
	for all the different blocks. 

	Like in other parts of this library, we want to transport these parameters
	in a single matrix. Therefore, we have to agree where in the matrix we store what.
	There are
		(1) K mixture coefficients - there are K* (L/LBlock)-many
		(2) LBlock x K components of all the mean-vectors (mus)
		(3) K components of the variance matrix (actually we store the standard deviations) 
			(assume spherical Gaussian -> single variance)

	
	For each block, we have the following structure. 

	All these components are transmitted in a single matrix of size 
	K x (LBlock +2 ) according to the following format:																	SIGMA											PI
	Block 0								Block 1							... Block Blocks-1		...						Block 0		...		Block Blocks-1 				Block 0				...			Block-1
	0			...		LBlock-1		LBlock				2*LBlock-1		(Blocks-1)*LBLock	...	BLocks*LBlock-1		Blocks*LBLock		Blocks*LBlock+Blocks-1		Blocks*LBlock+Blocks			BLocks*(LBlock+2)-1
	Mu_{0,0}	...		Mu_{0,LB-1}		Mu_{0,0}	...		Mu_{0,LB-1}		Mu_{0,0}				Mu_{0,LB-1}			Sigma_0		...		Sigma_0						Pi_0							Pi_0
	Mu_{1,0}	...		Mu_{1,LB-1}		Mu_{1,0}				.			.											Sigma_1		...		Sigma_1						Pi_1							Pi_1
	.					.				.						.			.											.					.							.								.
	.					.				.						.			.											.					.							.								.
	.					.				.						.														.					.							.								.
	Mu_{K-1,0}	...		Mu_{K-1, LB-1}	Mu_{K-1,0}	...		Mu_{K-1, LB-1}	Mu_{K-1,0}			...	Mu_{K-1, LB-1}		Sigma_(K-1) ...		Sigma_(K-1)					Pi_(K-1)						Pi_(K-1)

	The parameters are stored in the param matrix in the x-direction.

	Each block has K*(LBlock+2) elements. The tot
**/


class MixtureDensityModel : public DiscarnateLayer{
public:
	MixtureDensityModel(size_t _K, size_t _L, size_t _BLock, CNetLayer& lower);
	MixtureDensityModel(size_t _K, size_t _L, size_t _BLock, size_t NIN);
	MixtureDensityModel(const MixtureDensityModel* other) = delete;
	~MixtureDensityModel();

	// Overwrite virtual functions from Discarnatelayer
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive);

	fREAL negativeLogLikelihood(const MAT& t) const;
	inline size_t getNOUT() const { return L; };

private:
	void conditionalMean(MAT& networkOut); // output function
	void maxMixtureCoefficient(MAT& networkOut); // output function
	void updateParameters(MAT& networkOut);
	void getParameters(MAT& toCopyTo); // output function
	MAT computeErrorGradient(const MAT& t);
	MAT reconstructTarget(const MAT& diffMatrix);

	size_t K; // number of mixture coefficients
	size_t L; // just for convenience - dimension of network output
	size_t LBlock;
	size_t Blocks; // 
	
	MAT param; // Matrix(K,L) of kernel centres mu_j. For each mixture kernel, a scalar variance. We assume the gaussian to be spherical.
				// Mixture coefficients - sum to one.
	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};

#endif
