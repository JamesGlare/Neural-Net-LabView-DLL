#pragma once

#include "defininitions.h"
#include "CNetLayer.h"

#ifndef CNET_PHYSICALLAYER
#define CNET_PHSICALLAYER

/* Abstract base class for a layer with weights.
* This means that this class sits in between nodes and forwards input or
* backpropagates deltas and updates its weights.
*/
class PhysicalLayer : public CNetLayer{
public:
	// constructors and initializers
	PhysicalLayer(size_t _NOUT, size_t _NIN);
	PhysicalLayer(size_t _NOUT, size_t _NIN, actfunc_t type);
	PhysicalLayer(size_t _NOUT, actfunc_t type, CNetLayer& const lower);
	virtual ~PhysicalLayer() {}; // purely abstract

	// forProp
	virtual void forProp(MAT& in, bool saveActivation) = 0; // recursive
															// backprop
	virtual MAT grad(MAT& const input) = 0;
	virtual void backPropDelta(MAT& const delta) = 0; // recursive
	fREAL applyUpdate(learnPars pars, MAT& const input); // recursive
	void NesterovParameterSetback(learnPars pars); 
	void NesterovParameterReset(learnPars pars);

	void resetConjugate(MAT& const input);
	// getters
	void copyLayer(fREAL* const toCopyTo);

protected:
	MAT layer; // actual layer
	MAT vel; // velocity for momentum OR previous gradient
	MAT prevStep; // this is needed for conjugate gradient method
	MAT gradient; // add gradients for minibatch before applying update.

	virtual void init()=0; // initialize all the weight matrices
};

#endif