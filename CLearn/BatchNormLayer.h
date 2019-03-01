#pragma once
#include "defininitions.h"
#include "CNetLayer.h"
#include "BatchBuffer.h"
#include "Stepper.h"


#ifndef CNET_BATCHNORMLAYER
#define CNET_BATCHNORMLAYER
/* Batch normalization layer
*  This type of layer sits akwardly with the class structure
*  and therefore inherits directly from CNetLayer.
*  It's neither physical nor discarnate, since it has a few
*  parameters, but does not follow things like weight-norm.
*/
class BatchNormLayer : public CNetLayer {
public:
	BatchNormLayer(size_t NIN);
	BatchNormLayer(CNetLayer& lower);
	~BatchNormLayer();

	// type
	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool training, bool recursive); // recursive
	// backprop
	void backPropDelta(MAT& delta, bool recursive); // recursive
	void applyUpdate(const learnPars& pars, MAT& input, bool recursive); // recursive

private:
	void normalize(MAT& in);

	void init();
	MAT currMean; // mu_B
	MAT currSigma; // sqrt(var+epsilon)
	MAT gamma;
	MAT beta;

	BatchBuffer buffer;

	Stepper gamma_stepper;
	Stepper beta_stepper;
};

#endif // !CNET_BATCHNORMLAYER
