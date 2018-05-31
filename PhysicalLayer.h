#pragma once
#include "defininitions.h"
#include "CNetLayer.h"
#include "BatchNormalizer.h"

#ifndef CNET_PHYSICALLAYER
#define CNET_PHSICALLAYER

/* Abstract base class for a layer with weights.
* This means that this class sits in between nodes and forwards input or
* backpropagates deltas and updates its weights.
*/
class PhysicalLayer : public CNetLayer{
public:
	/* Constructors
	* Initialize Base Class (CNetLayer) and internal instance of MiniBatchNormalization.
	*/ 
	PhysicalLayer(size_t _NOUT, size_t _NIN);
	PhysicalLayer(size_t _NOUT, size_t _NIN, actfunc_t type);
	PhysicalLayer(size_t _NOUT, actfunc_t type, CNetLayer& const lower);
	virtual ~PhysicalLayer() {}; // purely abstract

	/* Forward Propagation 
	*  Implementation on child class level.
	*/
	virtual void forProp(MAT& in, learnPars& const pars, bool training) = 0; // recursive
															// backprop
	virtual MAT grad(MAT& const input) = 0;
	/* Backward propagation
	* Implementation on child class level.
	*/
	virtual void backPropDelta(MAT& const delta) = 0; // recursive
	fREAL applyUpdate(learnPars pars, MAT& const input); // recursive
	void resetConjugate(MAT& const input);

	// Read-only access to weight parameters.
	void copyLayer(fREAL* const toCopyTo);

protected:
	/* MiniBatch Normalization
	*/
	void miniBatch_updateBuffer(MAT& input);
	MAT& miniBatch_normalize();
	void miniBatch_updateModel();
	MAT& miniBatch_passOnNormalized();
	MAT& miniBatch_denormalize(MAT& toPassOn);
	void miniBatch_clearBuffer();
	inline bool miniBatch_stillToCome() { return batchNormalizer.stillToCome();};

	/* Internal weights and parameters
	*/
	MAT layer; // actual layer
	MAT vel; // velocity for momentum OR previous gradient
	MAT prevStep; // this is needed for conjugate gradient method
	MAT gradient; // add gradients for minibatch before applying update.

	virtual void init()=0; // initialize all the weight matrices
private:
	// Some auxiliary functions
	void NesterovParameterSetback(learnPars pars);
	void NesterovParameterReset(learnPars pars);
	// BatchNormalization instance
	BatchNormalizer batchNormalizer;
};

#endif