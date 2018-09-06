#include "stdafx.h"
#include "Stepper.h"

Stepper::Stepper(MATIND _layerIndex) {
	/* Initialized momentum stuff
	*/
	velocity = MAT(_layerIndex.rows, _layerIndex.cols);
	velocity.setZero();
	/* Conjugate gradient
	*/
	hi = MAT(_layerIndex.rows, _layerIndex.cols); // same size as gradient = layer.size
	hi.setZero();
	gi_prev = MAT(_layerIndex.rows, _layerIndex.cols);  // gradient
	gi_prev.setZero();
	gamma = 0.0f;

	/* Adam step
	*/
	vt = MAT(_layerIndex.rows, _layerIndex.cols);
	mt = MAT(_layerIndex.rows, _layerIndex.cols);
	epsilon = MAT(_layerIndex.rows, _layerIndex.cols);
	epsilon.setConstant(1E-8);

	vt.setZero();
	mt.setZero();
	mode_adamStep = false;
	mode_conjugateGradient = false;
}
void Stepper::notifyFormChange(MATIND _newForm) {
	hi.resize(_newForm.rows, _newForm.cols);
	gi_prev.resize(_newForm.rows, _newForm.cols);
	velocity.resize(_newForm.rows, _newForm.cols);
}

void Stepper::resetConjugate(const MAT& gradient) {
	gi_prev = gradient;
	hi= gradient;
}

void Stepper::doConjugateStep(MAT& layer, const MAT& gradient, const learnPars& pars) {
	// Actually do the conjugate gradient step.
	gamma = (gradient.cwiseProduct(gradient - gi_prev)).sum() / (gi_prev.cwiseProduct(gi_prev)).sum();
	if (!isnan(gamma) && !isinf(gamma)) {
		hi = gradient + gamma*hi; // hi = gi + gamma*h(i-1)
		layer = (1.0f - pars.lambda)*layer + pars.eta*hi; // !!!! do the actual step
		gi_prev = gradient; // save negative gradient	
	}
	else { // something went wrong - reset.
		resetConjugate(gradient);
	}
}
void Stepper::doMomentumStep(MAT& layer, const MAT& gradient, const learnPars& pars) {
	//layer += pars.gamma*velocity; // NESTEROV 1
	// (2) apply the velocity step
	velocity = pars.gamma*velocity - pars.eta*gradient;
	layer = (1.0f - pars.lambda)*layer + velocity;
	//layer -= pars.gamma*velocity; // NESTEROV 2 - now we can calculate the next gradient
}
void Stepper::resetAdam() {
	beta1t = beta1;
	beta2t = beta2;
	mt.setZero();
	vt.setZero();
}
void Stepper::doAdamStep(MAT& layer, const MAT& gradient, const learnPars& pars) {
	//if (gradient.allFinite()) {
		beta1t *= beta1;
		beta2t *= beta2;

		mt = beta1*mt + (1 - beta1)*gradient;
		vt = beta2*vt + (1 - beta2)*gradient.unaryExpr(&norm);
		//if (mt.allFinite() && vt.allFinite())
			layer = (1.0f - pars.lambda)*layer - pars.eta* sqrt(1.0f - beta2t) / (1.0f - beta1t)*(mt.cwiseQuotient(vt.unaryExpr(&sqroot)) + epsilon);
		//else
		//	resetAdam();
	//} else {
	//	resetAdam();
	//}
}
/* Nesterov accelerated Momentum AND Conjugate Gradient in one
*/
void Stepper::stepLayer(MAT& layer, MAT& gradient, const learnPars& pars) {
	if (pars.conjugate) {
		/* Conjugate Gradient Method
		* T Masters P 104
		* * * * * * * * * * * * * *
		* Initialization
		* g_0:=  -gradient  [done in function resetConjugate]
		* h_0 := -gradient  [done in function resetConjugate]
		* * * * * * * * * * * * * *
		* For each step i in (1, ..., N)
		* gamma = (g_i-g_(i-1)).g_i/(g_(i-1).g_i)
		* h_i := g_i + gamma*h_(i-1)
		* Comments:
		* gi_prev -> g_(i-1).
		* hi -> h_i
		* -gradient -> g_i
		*/
		gradient *= -1; 

		if (!mode_conjugateGradient) { // user changed to conjugate gradient method
			mode_conjugateGradient = true;
			resetConjugate(gradient);
		} else {
			mode_adamStep = false;
			doConjugateStep(layer, gradient, pars);
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
			mode_adamStep = true;
			resetAdam();
		} else {
			mode_conjugateGradient = false;
			doAdamStep(layer, gradient, pars);
		}
	} else {
		mode_conjugateGradient = false;
		mode_adamStep = false;
		/* Nesterov Accelerated Momentum
		* v_t = gamma*v_(t-1) +eta*grad[ theta - gamma*v_(t-1) ]
		* theta = theta - v_t 
		*/
		// (1) The gradient was computed at  theta - gamma*v_t-1 so reset that
		doMomentumStep(layer, gradient, pars);
	}
}