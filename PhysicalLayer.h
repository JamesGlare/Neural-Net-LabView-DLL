#pragma once
#include "defininitions.h"
#include "CNetLayer.h"
#include "BatchBuffer.h"
#include "Stepper.h"

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
	PhysicalLayer(size_t _NOUT, size_t _NIN, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex);
	PhysicalLayer(size_t _NOUT, size_t _NIN,  actfunc_t type, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex);
	PhysicalLayer(size_t _NOUT, actfunc_t type, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex, CNetLayer& lower);
	virtual ~PhysicalLayer() {}; // purely abstract

	/* Forward Propagation 
	*  Implementation on child class level.
	*/
	virtual void forProp(MAT& in, bool training, bool recursive) = 0; // recursive
															// backprop
	virtual MAT grad(MAT& input) = 0;
	/* Backward propagation
	* Implementation on child class level.
	*/
	virtual void backPropDelta(MAT& delta, bool recursive) = 0; // recursive
	void applyUpdate(const learnPars& pars, MAT& input, bool recursive); // recursive
	
	// Read-only access to weight parameters.
	void copyLayer(fREAL* const toCopyTo);

protected:
	/* MiniBatch Buffering
	*/
	void miniBatch_updateBuffer(MAT& input);
	MAT& miniBatch_normalize();
	void miniBatch_updateModel();
	inline bool miniBatch_stillToCome() { return batch.stillToCome();};
	/* Weight normalization
 	*/
	bool weightNormMode;
	MAT G; // as many as NOUT for FC layers
	MAT V; // store V and update W each time, keep V rowwise normalised
	virtual void updateW() = 0; // to W
	virtual void initV() = 0;
	virtual void initG() =0;
	virtual void normalizeV() = 0;
	virtual MAT inversVNorm() = 0;
	virtual MAT gGrad(MAT& grad) = 0; // gradient in g's
	virtual MAT vGrad(MAT& grad, MAT& ggrad) = 0; // gradient in V
	
	/* Internal weights and parameters
	*/
	MAT layer; // actual layer
	/* Subclasses 
	* initialized in derived classes
	*/
	Stepper stepper; // subclass - takes care of all the SDG stuff
	Stepper GStepper;
	Stepper VStepper;
	BatchBuffer batch; // BatchNormalization instance

	void init(); // initialize all the weight matrices
};

#endif