#pragma once
#include "defininitions.h"

#ifndef CNET_STEPPER
#define CNET_STEPPER
/*	Implements the stochastic gradient descent.
*	The Stepper class currently supports
*		(1) Momentum-based gradient descent
*		(2) ADAM
*		(3) Conjugate gradient.
*/
class Stepper {
public:
	Stepper(MATIND _layerIndex); // give all matrices, so that stepper can see dimensions
	void stepLayer(MAT& W, const MAT& grad, const learnPars&  pars);
	void giveRMSgrad(const MAT& rmsGrad);
	//void stepLayer_weightNormalized(MAT& layer, MAT& const V_gradient, MAT& const G_gradient,  learnPars& const pars);
	void reset();

private:
	
	/* Nesterov accelerated momentum 
	*/
	MAT velocity;
	void doMomentumStep(MAT& x, const MAT& grad, const learnPars& pars);
	
	/* Adam optimization
	*/
	void doAdamStep(MAT& x, const MAT& grad, const learnPars& pars);
	void resetAdam();
	void resetRMSProp();

	bool mode_adamStep;
	MAT mt;
	MAT vt;
	fREAL alphat = 0;
	fREAL beta1 = 0.5;
	fREAL beta1t = 0.9;
	fREAL beta2 = 0.9;
	fREAL beta2t = 0.999;
	MAT epsilon;

	/* RMSProp optimization
	*/
	bool mode_RMSProp;
	MAT prev_avgGrad;
	MAT w_RMS;
	void doRMSPropStep(MAT& x, const MAT& grad, const learnPars& pars);

	/* Other
 	*/
	void clipWeights(MAT& x, fREAL clip);
	
};
#endif