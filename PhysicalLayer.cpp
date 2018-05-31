#include "stdafx.h"
#include "PhysicalLayer.h"

// pass constructor to base class
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN) : batchNormalizer(NIN), CNetLayer(_NOUT, _NIN) { };
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : batchNormalizer(NIN), CNetLayer(_NOUT, _NIN, type) { };
PhysicalLayer::PhysicalLayer(size_t _NOUT, actfunc_t type, CNetLayer& const lower) : batchNormalizer(lower.getNOUT()), CNetLayer(_NOUT, type, lower) {  };

void PhysicalLayer::NesterovParameterSetback(learnPars pars) {
	if(!pars.conjugate && vel.allFinite())
		layer -=  pars.gamma*vel;
}
void PhysicalLayer::NesterovParameterReset(learnPars pars) {
	if (!pars.conjugate && vel.allFinite())
		layer+= pars.gamma*vel;
}

void PhysicalLayer::copyLayer(fREAL* const toCopyTo) {
	copyToOut(layer.data(), toCopyTo, layer.size());
}

fREAL PhysicalLayer::applyUpdate(learnPars pars, MAT& const input) {
	fREAL gamma = 0;
	gradient -= grad(input);
	if (0 == pars.batch_update) {
		if (pars.conjugate) {
			// treat vel as g_(i-1)
			fREAL denom = vel.cwiseProduct(vel).sum(); // should be scalar
			gamma = gradient.cwiseProduct(gradient - vel).sum() / denom;
			if (!isnan(gamma)) {
				prevStep = gradient + gamma*prevStep; // save step
				layer = (1.0f - pars.lambda)*layer + pars.eta*gradient; // do the actual step
				vel = gradient; // save negative gradient
			}
			else {
				resetConjugate(input);
			}
		} else {
			NesterovParameterReset(pars);
			vel = pars.gamma*vel - pars.eta*gradient;
			if (vel.allFinite())
				layer = (1.0f - pars.lambda)*layer - vel; // this reverse call forces us to implement this function in the derived classes
			NesterovParameterSetback(pars); // now set layer back by gamma*v_t, such that future gradients are calculated using  
		}
		if (hierarchy != hierarchy_t::output) {
			above->applyUpdate(pars, input);
		}
		gradient.setZero(); // reset the gradient
	}
	return gamma;
}

void PhysicalLayer::resetConjugate(MAT& const input) {
	MAT gi = -grad(input);
	vel = gi;
	prevStep = gi;
	if (hierarchy != hierarchy_t::output)
		above->resetConjugate(input);
}

void PhysicalLayer::miniBatch_updateBuffer(MAT& input) {
	batchNormalizer.updateBuffer(input);
}
MAT& PhysicalLayer::miniBatch_normalize() {
	return batchNormalizer.passOnNormalized();
}

void PhysicalLayer::miniBatch_updateModel() {
	batchNormalizer.updateModel(); // build the model
}

MAT& PhysicalLayer::miniBatch_denormalize(MAT& toPassOn) {
	return batchNormalizer.deNormalize(toPassOn);
}

void PhysicalLayer::miniBatch_clearBuffer() {
	batchNormalizer.clearBuffer();
}

MAT& PhysicalLayer::miniBatch_passOnNormalized() {
	return batchNormalizer.passOnNormalized();
}
