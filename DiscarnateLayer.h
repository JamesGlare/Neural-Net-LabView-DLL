#pragma once
#include "defininitions.h"
#include "CNetLayer.h"

#ifndef CNET_DISCARNATELAYER
#define CNET_DISCARNATELAYER
/* Abstract base class for a layer without weights or adaptable parameters.
* Discarnate classes can, for instance, apply functions or change the structure of the data.
* They need only be able to do forward and backward passes. No updates.
*/
class DiscarnateLayer : public CNetLayer {
public:
	// constructors and initializers
	DiscarnateLayer(size_t _NOUT, size_t _NIN);
	DiscarnateLayer(size_t _NOUT, size_t _NIN, actfunc_t type);
	DiscarnateLayer(size_t _NOUT, actfunc_t type, CNetLayer& const lower);
	virtual ~DiscarnateLayer() {}; // purely abstract

	// type
	// forProp
	virtual void forProp(MAT& in, bool saveActivation) = 0; // recursive
	virtual void backPropDelta(MAT& const delta) = 0; // recursive
	fREAL applyUpdate(learnPars pars, MAT& const input); // recursive
	void resetConjugate(MAT& const input); // recursive;
};

#endif
