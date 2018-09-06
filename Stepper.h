#pragma once
#include "defininitions.h"

#ifndef CNET_STEPPER
#define CNET_STEPPER

class Stepper {
public:
	Stepper(MATIND _layerIndex); // give all matrices, so that stepper can see dimensions
	void stepLayer(MAT& layer, MAT& gradient, const learnPars&  pars);

	void notifyFormChange(MATIND _newIndex);
	//void stepLayer_weightNormalized(MAT& layer, MAT& const V_gradient, MAT& const G_gradient,  learnPars& const pars);

private:
	
	/* Nesterov accelerated momentum 
	*/
	MAT velocity;
	void doMomentumStep(MAT& layer, const MAT& gradient, const learnPars& pars);

	/* Conjugate gradient 
	*/
	void doConjugateStep(MAT& layer, const MAT& gradient, const learnPars& pars);
	bool mode_conjugateGradient;
	void resetConjugate(const MAT& gradient);

	MAT hi;
	MAT gi_prev;
	fREAL gamma;
	
	/* Adam optimization
	*/
	void doAdamStep(MAT& layer, const MAT& gradient, const learnPars& pars);
	void resetAdam();
	bool mode_adamStep;
	MAT mt;
	MAT vt;
	fREAL beta1 = 0.9;
	fREAL beta1t = 0.9;
	fREAL beta2 = 0.999;
	fREAL beta2t = 0.999;
	MAT epsilon;
};
#endif