#include "stdafx.h"
#include "PhysicalLayer.h"

// pass constructor to base class
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex) : 
	batch(_layerIndex, _NOUT, _NIN ), stepper(_layerIndex), VStepper(_VIndex),GStepper(_GIndex), layer(_layerIndex.rows, _layerIndex.cols), 
	V(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols), CNetLayer(_NOUT, _NIN) {
	init();
}
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN, actfunc_t type, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex) :
	batch(_layerIndex, _NOUT,_NIN), stepper(_layerIndex), VStepper(_VIndex), GStepper(_GIndex), layer(_layerIndex.rows, _layerIndex.cols),
	V(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols), CNetLayer(_NOUT, _NIN, type) {
	init();
}
PhysicalLayer::PhysicalLayer(size_t _NOUT, actfunc_t type, MATIND _layerIndex, MATIND _VIndex, MATIND _GIndex, CNetLayer& lower) :
	batch(_layerIndex, _NOUT, lower.getNOUT()), stepper(_layerIndex), VStepper(_VIndex), GStepper(_GIndex), layer(_layerIndex.rows, _layerIndex.cols),
	V(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols), CNetLayer(_NOUT, type, lower) { 
	init();
}
void PhysicalLayer::init() {
	layer.setRandom();
	G.setZero();
	V.setZero();
	weightNormMode = false;
}
void PhysicalLayer::copyLayer(fREAL* const toCopyTo) {
	copyToOut(layer.data(), toCopyTo, layer.size());
}
void PhysicalLayer::setLayer(fREAL* const copyFrom) {
	layer = MATMAP(copyFrom, layer.rows(), layer.cols());
}
// CHECK SIGNS!!
void PhysicalLayer::applyUpdate(const learnPars& pars, MAT& input, bool recursive) {

	/* new version of this function
	*/
	batch.swallowGradient(grad(input));
	if (0 == pars.batch_update) {
		// gradient reference can be accessed in batch.avgGradient() 
		if (pars.weight_normalization) {
			/* The step under weight normalization is slightly different and depends on the structure of the weights.
			* Therefore, it affects higher level operation.
			*/
			if (!weightNormMode) {
				initG();
				initV();
				weightNormMode = true;
			} 
			MAT Ggradient = gGrad(batch.avgGradient());

			GStepper.stepLayer(G, Ggradient, pars);
			VStepper.stepLayer(V, vGrad(batch.avgGradient(), Ggradient), pars);

			//normalizeV();
			updateW();
			
		} else {
			/* Standard step.
			*/
			weightNormMode = false;
			stepper.stepLayer(layer, batch.avgGradient(), pars);
		}
		batch.clearGradient();
	}
	// Recursion
	if (hierarchy != hierarchy_t::output && recursive) {
		above->applyUpdate(pars, input, true);
	}

}
MATIND PhysicalLayer::layerDimensions() const {
	size_t rows_ = layer.rows();
	size_t cols_ = layer.cols();
	return MATIND{rows_, cols_};
}
void PhysicalLayer::miniBatch_updateBuffer(MAT& input) {
	batch.updateBuffer(input);
}
MAT& PhysicalLayer::miniBatch_normalize() {
	return batch.passOnNormalized();
}

void PhysicalLayer::miniBatch_updateModel() {
	batch.updateModel(); // build the model
}