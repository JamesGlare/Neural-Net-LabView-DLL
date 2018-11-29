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
	K x (LBlock +2 ) according to the following format:																	
	MU = (NOUTY, NOUTX* K), SIGPI= K*Blocks*2 
	L = NOUTY * NOUTX
	
	The parameters are stored in the param matrix in the x-direction.

	Each block has K*(LBlock+2) elements. The tot
**/


class MixtureDensityModel : public DiscarnateLayer{
public:
	MixtureDensityModel(size_t _NOUTX, size_t _NOUTY, size_t _features, size_t _BlockX, size_t _BlockY, CNetLayer& lower);
	MixtureDensityModel(size_t _NOUTX, size_t _NOUTY, size_t _features, size_t _BlockX, size_t _BlockY);
	~MixtureDensityModel();
	layer_t whoAmI() const;

	// Overwrite virtual functions from Discarnatelayer
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive);

	fREAL negativeLogLikelihood(MAT& t);
	inline size_t getNOUT() const { return L; };

private:
	void conditionalMean(MAT& networkOut); // output function
	void maxMixtureCoefficient(MAT& networkOut); // output function
	void updateParameters(MAT& networkOut);
	void getParameters(MAT& toCopyTo); // output function
	MAT computeErrorGradient(MAT& t);
	MAT reconstructTarget(const MAT& diffMatrix);

	// 2D index function
	inline size_t _BLOCKY(size_t b) const { return b % (uint32_t)sqrt(Blocks); };
	inline size_t _BLOCKX(size_t b) const { return b / (uint32_t)sqrt(Blocks); }

	size_t K; // number of mixture coefficients
	size_t L; // just for convenience - dimension of network output
	size_t LBlock;
	size_t Blocks; // 
	size_t BlockX;
	size_t BlockY;
	size_t NOUTX;
	size_t NOUTY;
	
	MAT SIG; // sigmas, each (block, feature) has a single variance (std)
	MAT PI;
	MAT MU;

	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};

#endif
