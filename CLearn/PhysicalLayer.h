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
	PhysicalLayer(size_t _NOUT, size_t _NIN,  MATIND _WIndex, MATIND _VIndex, MATIND _GIndex, MATIND _u1v1Index);
	PhysicalLayer(size_t _NOUT, size_t _NIN,  actfunc_t type, MATIND _WIndex, MATIND _VIndex, MATIND _GIndex, MATIND _u1v1Index);
	PhysicalLayer(size_t _NOUT,  actfunc_t type, MATIND _WIndex, MATIND _VIndex, MATIND _GIndex, MATIND _u1v1Index, CNetLayer& lower);
	virtual ~PhysicalLayer(); // purely abstract

	/* Forward Propagation 
	*  Implementation on child class level.
	*/
	virtual void forProp(MAT& in, bool training, bool recursive) = 0; // recursive
	virtual MAT w_grad(MAT& input) = 0;
	virtual MAT b_grad() = 0;
	
	/* Backward propagation
	* Implementation on child class level.
	*/
	virtual void backPropDelta(MAT& delta, bool recursive) = 0; // recursive
	void applyUpdate(const learnPars& pars, MAT& input, bool recursive); // recursive
	/* Initialization Routine
	*/
	virtual void constrainToMax(MAT& mues, MAT& max) = 0;
	/* Read-only access to weight parameters.
	*/
	MAT copyW() const;
	MAT copyb() const;
	// Set the layer weights (e.g. for initialization)
	void setW(const MAT& newLayer);
	MATIND WDimensions() const;
	void snorm_switchW();

protected:
	/* Weight normalization
 	*/
	bool weightNormMode;
	MAT G; // as many as NOUT for FC layers
	MAT V; // store V and update W each time
	MAT VInversNorm;
	Stepper wnorm_Vstepper;
	Stepper wnorm_Gstepper;

	virtual void wnorm_setW() = 0; // to W
	virtual void wnorm_initV() = 0;
	virtual void wnorm_initG() =0;
	virtual void wnorm_normalizeV() = 0;
	virtual void wnorm_inversVNorm() = 0;
	virtual MAT wnorm_gGrad(const MAT& grad) = 0; // gradient in g's
	virtual MAT wnorm_vGrad(const MAT& grad, MAT& ggrad) = 0; // gradient in V
	
	/* Spectral Normalization
	*/
	bool spectralNormMode;
	MAT u1;
	MAT v1;
	MAT W_temp;
	fREAL sigma;
	void snorm_initUV();
	virtual void snorm_setW()=0;
	virtual void snorm_updateUVs()=0;
	virtual MAT snorm_dWt(MAT& grad)=0; // neecds to be multiplied element-wise to 
	fREAL lambdaBatch;
	size_t lambdaCount;
	/* Internal weights and parameters
	*/
	MAT W; // actual layer
	MAT b;
	/* Subclasses 
	* initialized in derived classes
	*/
	Stepper w_stepper; // subclass - takes care of all the SDG stuff
	Stepper b_stepper; // subclass - takes care of all the SDG stuff

	BatchBuffer w_batch; // BatchNormalization instance
	BatchBuffer b_batch; // BatchNormalization instance

	void init(); // initialize all the weight matrices
};

#endif