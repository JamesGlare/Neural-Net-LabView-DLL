#include "stdafx.h"
#include "FullyConnectedLayer.h"

// constructor - for input layers
FullyConnectedLayer::FullyConnectedLayer(size_t NOUT, size_t NIN, actfunc_t type) : PhysicalLayer(NOUT, NIN, type){
	 // declare matrix
	init();
}
// constructor - for hidden layers and output layers
FullyConnectedLayer::FullyConnectedLayer(size_t NOUT, actfunc_t type, CNetLayer& const lower) : PhysicalLayer(NOUT, type,lower) {
	// layer and velocity matrices
	init();
}
// Destructor
FullyConnectedLayer::~FullyConnectedLayer() {
	layer.resize(0, 0);
}

layer_t FullyConnectedLayer::whoAmI() const {
	return layer_t::fullyConnected;
}
// init
void FullyConnectedLayer::init() {

	layer = MAT(NOUT, NIN + 1);
	gradient = MAT(NOUT, NIN + 1);
	actSave = MAT(NIN, 1);
	deltaSave = MAT(NOUT,1);
	vel = MAT(layer.rows(), layer.cols());
	prevStep = MAT(layer.rows(), layer.cols());

	actSave.setConstant(0);
	vel.setConstant(0);
	deltaSave.setConstant(0);
	gradient.setConstant(0);
	layer.setRandom(); // in (-1,1)
	fREAL max = 1.0f / NIN;
	fREAL min = -1.0f / NIN;
	layer *= (max-min)/2.0f ; // (-max, max)
	layer = (layer.array() + (max+min)/2.0f).matrix(); // (min,  2+min)
}

void FullyConnectedLayer::forProp(MAT& inBelow, learnPars& const pars, bool training) {
	if (training) {
		if (pars.batch_normalization == 1) {
			/* (1) Batch normalization
			*	Transfer incoming training examples into batch buffer.
			*/
			miniBatch_updateBuffer(inBelow);
			if (pars.batch_update == 0) {
				miniBatch_updateModel();
				while (miniBatch_stillToCome() >0) {
					// (2) get normalized example out of buffer
					actSave = layer*appendOneInline(miniBatch_passOnNormalized()); 
					if (hierarchy != hierarchy_t::output) {
						// (3) denormalize the output.
						actSave = miniBatch_denormalize(actSave);
						inBelow = actSave.unaryExpr(act); // apply non-linearity now
						pars.batch_update = miniBatch_stillToCome() - 1; // refresh the batch_update counter
						//(4) pass to higher layers
						above->forProp(inBelow, pars, true); 
					} else {
						inBelow = actSave;
					}
				}
				//(5) clear the buffer 
				miniBatch_clearBuffer(); 
			}
		} else {
			/* normal training forward pass
			*/ 
			actSave = layer*appendOneInline(inBelow); // save the activations before non-linearity
			if (hierarchy != hierarchy_t::output) {
				inBelow = actSave.unaryExpr(act);
				above->forProp(inBelow, pars, true);
			} else {
				inBelow = actSave;
			}
		}
	} else {
		/* Non-training forward pass
		*/
		if (hierarchy != hierarchy_t::output) {
			inBelow = (layer*appendOneInline(inBelow)).unaryExpr(act);
			above->forProp(inBelow, pars, false);
		} else {
			inBelow = layer*appendOneInline(inBelow);
		}
	}
}

void FullyConnectedLayer::backPropDelta(MAT& const deltaAbove) {
	//DACT(inAct).cwiseProduct(hiddenLayers[0].leftCols(hiddenLayers[0].cols() - 1).transpose()*hiddenDeltas[0]);
	deltaSave = deltaAbove;
	if (hierarchy != hierarchy_t::input ) {
		deltaAbove = (below->getDACT()).cwiseProduct(layer.leftCols(NIN).transpose() * deltaAbove);
		below->backPropDelta(deltaAbove);
	}
}

MAT FullyConnectedLayer::grad(MAT& const input) {
	if (hierarchy == hierarchy_t::input) {
		return deltaSave*appendOneInline(input).transpose();
	}
	else {
		return deltaSave * appendOneInline(below->getACT()).transpose();
	}
}

void FullyConnectedLayer::saveToFile(ostream& os) const {
	os<< NOUT << "\t" << NIN <<endl; // header line 2
	os << layer;
}
void FullyConnectedLayer::loadFromFile(ifstream& in) {
	in >> NOUT;
	in >> NIN;
	
	for (size_t i = 0; i < NOUT; i++) {
		for (size_t j = 0; j < NIN+1; j++) {
			in >> layer(i, j);
		}
	}
}