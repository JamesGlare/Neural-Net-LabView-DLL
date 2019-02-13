#include "stdafx.h"
#include "Stepper.h"

Stepper::Stepper(MATIND _WIndex) {
	/* Initialized momentum stuff
	*/
	velocity = MAT(_WIndex.rows, _WIndex.cols);
	velocity.setZero();
	/* RMSProp
	*/
	
	/* Adam step
	*/
	vt = MAT(_WIndex.rows, _WIndex.cols);
	mt = MAT(_WIndex.rows, _WIndex.cols);
	vt.setZero();
	mt.setZero();
	epsilon = MAT(_WIndex.rows, _WIndex.cols);
	epsilon.setConstant(1E-8);
	beta1t = beta1;
	beta2t = beta2;
	alphat = 0;
	/* RMSprop step
	*/
	MAT prev_avgGrad = MAT(_WIndex.rows, _WIndex.cols); 
	MAT w_RMS = MAT(_WIndex.rows, _WIndex.cols);
	mode_RMSProp = false;
	mode_adamStep = false;
}

void Stepper::reset() {
	resetAdam();
//	resetConjugate();
	velocity.setZero();
	resetRMSProp();
	mode_RMSProp = false;
	mode_adamStep = false;
}

void Stepper::doMomentumStep(MAT& W, const MAT& grad,  const learnPars& pars) {
	//layer += pars.gamma*velocity; // NESTEROV 1
	// (2) apply the velocity step
	velocity = pars.gamma*velocity - pars.eta*grad;
	W = (1.0f - pars.lambda)*W + velocity;
	
	//layer -= pars.gamma*velocity; // NESTEROV 2 - now we can calculate the next gradient
}
void Stepper::resetAdam() {
	beta1t = beta1;
	beta2t = beta2;
	mt.setZero();
	vt.setZero();
}

void Stepper::resetRMSProp(){
	prev_avgGrad.setZero();
	w_RMS.setZero();
}

void Stepper::clipWeights(MAT& W, fREAL clip) {
	clipParameters(W, clip);
}
void Stepper::doAdamStep(MAT& W, const MAT& grad, const learnPars& pars) {

		beta1t *= beta1;
		beta2t *= beta2;
		alphat = pars.eta *  sqrt(abs(1.0f - beta2t)) / (1.0f - beta1t);
		mt = beta1*mt + (1.0f - beta1)*grad;
		vt = beta2*vt + (1.0f - beta2)*grad.unaryExpr(&norm);
		
		W = (1.0f - pars.lambda)*W - alphat*mt.cwiseQuotient(vt.unaryExpr(&sqroot) + epsilon);
		
}
void Stepper::doRMSPropStep(MAT & W, const MAT & grad, const learnPars & pars) {
	prev_avgGrad *= 0.9;
	prev_avgGrad += 0.1*grad.unaryExpr(&norm);

	W = (1.0f - pars.lambda)*W - pars.eta*grad.cwiseQuotient( (prev_avgGrad + epsilon).unaryExpr(&sqroot));
}
/* Nesterov accelerated Momentum AND Conjugate Gradient in one
*/
void Stepper::stepLayer(MAT& W, const MAT& grad, const learnPars& pars) {
	if (pars.rmsprop) {
		if (!mode_RMSProp) {
			velocity.setZero();
			resetRMSProp();
			prev_avgGrad = grad.unaryExpr(&norm);
			mode_adamStep = false;
			mode_RMSProp = true;
		} else {
			doRMSPropStep(W, grad, pars);
		}
		
	} else if (pars.adam) {
		/*	Adam optimization
		*	see D Kingma, J Lei Ba, 2015
		*   m_t = beta1 * m_(t-1) + ( 1-beta1) * grad 
		*   v_t = beta2*v_(t-1)+(1-beta2)*grad cw* grad
		*	v^_t = v_t/ (1-beta2^t)
		*	m^_t = m_t/(1-beta1^t) 
		*	theta_t = theta_t - eta*m^_t/ (sqrt( v^_t) +eps) (element wise)
		*/
		if (!mode_adamStep) {
			resetAdam();
			velocity.setZero(); //set the momentum velocity to zero, in case we switch back to momentum-descent.

			mode_adamStep = true;
			mode_RMSProp = false;
		} else {
			doAdamStep(W, grad, pars);
		}
	} else {
		mode_RMSProp = false;
		mode_adamStep = false;
		/* Nesterov Accelerated Momentum
		* v_t = gamma*v_(t-1) +eta*grad[ theta - gamma*v_(t-1) ]
		* theta = theta - v_t 
		*/
		// (1) The gradient was computed at  theta - gamma*v_t-1 so reset that
		doMomentumStep(W, grad, pars);
	}
	// Wasserstein GAN clipping
	/*if (abs(pars.GAN_c) > 0.0f) {
		clipWeights(W, pars.GAN_c);
	}*/
}

void Stepper::giveRMSgrad(const MAT & rmsGrad){
	w_RMS = rmsGrad;
}
