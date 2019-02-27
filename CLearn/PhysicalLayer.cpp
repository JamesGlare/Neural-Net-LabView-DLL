#include "stdafx.h"
#include "PhysicalLayer.h"

// pass constructor to base class
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN, MATIND _WIndex, MATIND _VIndex, MATIND _GIndex, MATIND _u1v1Index) :
	w_batch(_WIndex, _NOUT, _NIN), b_batch(MATIND{ _NOUT, 1 }, _NOUT, _NIN), w_stepper(_WIndex), b_stepper(MATIND{ _NOUT, 1 }), wnorm_Vstepper(_VIndex), wnorm_Gstepper(_GIndex), 
	W(_WIndex.rows, _WIndex.cols), b(_NOUT, (size_t)1), V(_VIndex.rows, _VIndex.cols), W_temp(_WIndex.rows, _WIndex.cols), u1(_u1v1Index.rows, 1), v1(_u1v1Index.cols, 1),
	VInversNorm(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols), CNetLayer(_NOUT, _NIN) {
	init();
}
PhysicalLayer::PhysicalLayer(size_t _NOUT, size_t _NIN, actfunc_t type, MATIND _WIndex, MATIND _VIndex, MATIND _GIndex, MATIND _u1v1Index) :
	w_batch(_WIndex, _NOUT, _NIN), b_batch(MATIND{ _NOUT, 1 }, _NOUT, _NIN), w_stepper(_WIndex), b_stepper(MATIND{ _NOUT, 1 }), wnorm_Vstepper(_VIndex), wnorm_Gstepper(_GIndex), 
	W(_WIndex.rows, _WIndex.cols), b(_NOUT, (size_t)1), V(_VIndex.rows, _VIndex.cols), W_temp(_WIndex.rows, _WIndex.cols), u1(_u1v1Index.rows, 1), v1(_u1v1Index.cols, 1),
	VInversNorm(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols), CNetLayer(_NOUT, _NIN, type) {
	init();
}
PhysicalLayer::PhysicalLayer(size_t _NOUT, actfunc_t type, MATIND _WIndex, MATIND _VIndex, MATIND _GIndex,  MATIND _u1v1Index, CNetLayer& lower) :
	w_batch(_WIndex, _NOUT, lower.getNOUT()), b_batch(MATIND{ _NOUT, 1 }, _NOUT, lower.getNOUT()), w_stepper(_WIndex), b_stepper(MATIND{ _NOUT, 1 }), wnorm_Vstepper(_VIndex),
	wnorm_Gstepper(_GIndex), W(_WIndex.rows, _WIndex.cols), b(_NOUT, (size_t)1),
	V(_VIndex.rows, _VIndex.cols), W_temp(_WIndex.rows, _WIndex.cols), u1(_u1v1Index.rows, 1), v1(_u1v1Index.cols, 1), VInversNorm(_VIndex.rows, _VIndex.cols), G(_GIndex.rows, _GIndex.cols),
	CNetLayer(_NOUT, type, lower) {
	init();
}
PhysicalLayer::~PhysicalLayer() {
}

void PhysicalLayer::init() {
	// Spectral normalization
	u1.setRandom();
	v1.setRandom();
	W_temp.setZero();
	b.setZero();
	fREAL initStddev =  1.0f/sqrt(getNIN()/4.0);
	for (size_t i = 0; i < W.cols(); ++i) {
		for (size_t j = 0; j < W.rows(); ++j) {
			W(j, i) = normal_dist(0.0f, initStddev);
		}
	}
	/*for (size_t i = 0; i < b.cols(); ++i) {
		for (size_t j = 0; j < b.rows(); ++j) {
			b(j,i) = normal_dist(0.0f, initStddev);
		}
	}*/
	G.setOnes();
	V.setZero();
	VInversNorm.setOnes();
	weightNormMode = false;
	spectralNormMode = false;

}
MAT PhysicalLayer::copyW() const {
	return MAT(W);
}
MAT PhysicalLayer::copyb() const {
	return MAT(b);
}
void PhysicalLayer::setW(const MAT& newW) {
	assert(W.rows() == newW.rows());
	assert(W.cols() == newW.cols());
	W = newW;
}
// initialize vectors
void PhysicalLayer::snorm_initUV() {
	u1.setRandom();
	v1.setRandom();
}

void PhysicalLayer::snorm_switchW() {
	swap(W, W_temp);
}

// CHECK SIGNS!!
void PhysicalLayer::applyUpdate(const learnPars& pars, MAT& input, bool recursive) {

	/* new version of this function
	*/
	// Get gradient
	bool isDense = whoAmI() == layer_t::fullyConnected;

	if (inRange(getLayerNumber(), pars.firstTrain, pars.lastTrain)) {
		if (pars.accept) {
			w_batch.swallowGradient(w_grad(input));
			b_batch.swallowGradient(b_grad());
		// TODO ------ -------- Put this abomination of a hack into order. 
			if ( pars.spectral_normalization) { // collect special batch information for spectral normalization
				lambdaBatch += (deltaSave.transpose()*(actSave - b)).sum(); // store this value
				++lambdaCount;
			}
		}
		if (0 == pars.batch_update) {
			// gradient reference can be accessed in batch.avgGradient()
			if (pars.weight_normalization) {
				/* The step under weight normalization is slightly different and depends on the structure of the weights.
				* Therefore, it affects higher level operation.
				*/
				if (!weightNormMode) {
					wnorm_initG();
					wnorm_initV();
					wnorm_inversVNorm();
					wnorm_setW();
				}
				weightNormMode = true;

				// (1) Update the InversVNorm matrix, since it is used several times
				MAT w_grad = w_batch.avgGradient();
				MAT Ggradient = wnorm_gGrad(w_grad);
				// (2) Perform stoch grad steps on G and V

				wnorm_Gstepper.stepLayer(G, Ggradient, pars);
				wnorm_Vstepper.stepLayer(V, wnorm_vGrad(w_grad, Ggradient), pars); // not compatible with RMSprop implementation
				b_stepper.stepLayer(b, b_batch.avgGradient(), pars);

				// (3) Recompute inversNorm and update the layer weight matrix
				wnorm_inversVNorm();
				wnorm_setW();
			} else if ( pars.spectral_normalization) {
				/* Spectral Normalization
				*/
				if (!spectralNormMode) {
					snorm_initUV(); // initialize the u'v
					W_temp = W; // save W
					snorm_updateUVs();
					snorm_setW();
					spectralNormMode = true;
					lambdaBatch = 0;
					lambdaCount = 0;
				} else {
					spectralNormMode = true;

					// We are in the business of spectral normalization
					w_stepper.stepLayer(W_temp, snorm_dWt(w_batch.avgGradient()), pars);
					b_stepper.stepLayer(b, b_batch.avgGradient(), pars);
					snorm_updateUVs();
					snorm_setW();
				}
	
			} else {
				/* Standard step.
				*/
				if (weightNormMode || spectralNormMode) {
					// We were in some normalization mode before.
					// So we have to reset all the velocity, vt, mt ...etc matrices.
					w_stepper.reset();
				} 
				
				weightNormMode = false;
				spectralNormMode = false;
				w_stepper.stepLayer(W, w_batch.avgGradient(), pars);
				b_stepper.stepLayer(b, b_batch.avgGradient(), pars);
			}
			w_batch.clearGradients();
			b_batch.clearGradients();
		}
	}
	// Recursion
	if (getHierachy() != hierarchy_t::output && recursive) {
		above->applyUpdate(pars, input, true);
	}

}
MATIND PhysicalLayer::WDimensions() const {
	size_t rows_ = W.rows();
	size_t cols_ = W.cols();
	return MATIND{ rows_, cols_ };
}