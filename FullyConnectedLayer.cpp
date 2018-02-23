#include "stdafx.h"
#include "FullyConnectedLayer.h"

// constructor - for input layers
FullyConnectedLayer::FullyConnectedLayer(size_t NOUT, size_t NIN, actfunc_t type, fREAL min, fREAL max) : CNetLayer(NOUT, NIN, type){
	 // declare matrix
	init(min, max);
}
// constructor - for hidden layers and output layers
FullyConnectedLayer::FullyConnectedLayer(size_t NOUT, actfunc_t type, fREAL min, fREAL max, CNetLayer& const lower) :  CNetLayer(NOUT, type,lower) {
	// layer and velocity matrices
	init(min, max);
}
// Destructor
FullyConnectedLayer::~FullyConnectedLayer() {
	layer.resize(0, 0);
}

layer_t FullyConnectedLayer::whoAmI() const {
	return layer_t::fullyConnected;
}
// init
void FullyConnectedLayer::init(fREAL min, fREAL max) {

	layer = MAT(NOUT, NIN + 1);
	actSave = MAT(NIN, 1);
	deltaSave = MAT(NOUT,1);
	vel = MAT(layer.rows(), layer.cols());
	prevStep = MAT(layer.rows(), layer.cols());

	actSave.setConstant(0);
	vel.setConstant(0);
	deltaSave.setConstant(0);

	layer.setRandom(); // in (-1,1)
	layer *= (max-min)/2.0f ; // (-max, max)
	layer = (layer.array() + (max+min)/2.0f).matrix(); // (min,  2+min)
}

void FullyConnectedLayer::forProp(MAT& inBelow, bool saveActivation) {
	if (saveActivation) {
		actSave = layer*appendOneInline(inBelow);
		if (hierarchy != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			above->forProp(inBelow, true);
		}
		else {
			inBelow =  actSave;
		}
	} else {
		if (hierarchy != hierarchy_t::output) {
			inBelow = (layer*appendOneInline(inBelow)).unaryExpr(act);
			above->forProp(inBelow, false);
		}
		else {
			inBelow = (layer*appendOneInline(inBelow));
		}
	}
}
void FullyConnectedLayer::backPropDelta(MAT& const deltaAbove) {
	//DACT(inAct).cwiseProduct(hiddenLayers[0].leftCols(hiddenLayers[0].cols() - 1).transpose()*hiddenDeltas[0]);
	deltaSave = deltaAbove;
	if (hierarchy != hierarchy_t::input ) {
		MAT temp = (below->getDACT()).cwiseProduct(layer.leftCols(NIN).transpose() * deltaAbove);
		below->backPropDelta(temp);
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

fREAL FullyConnectedLayer::applyUpdate(learnPars pars, MAT& const input) {
	fREAL gamma = 0;
	fREAL denom = 0;
	if (pars.conjugate) {
		// treat vel as g_(i-1)
		denom = vel.cwiseProduct(vel).sum(); // should be scalar
		MAT gi = -grad(input);
		gamma = gi.cwiseProduct(gi - vel).sum() / denom;
		if (!isnan(gamma)) {
			prevStep = gi + gamma*prevStep; // save step
			layer = (1.0f - pars.lambda)*layer + pars.eta*gi; // do the actual step
			vel = gi; // save negative gradient
		}
		else {
			resetConjugate(input);
		}
	} else {
		prevStep = -grad(input);
		vel = pars.gamma*vel - pars.eta*prevStep;
		if(vel.allFinite())
			layer = (1.0f - pars.lambda)*layer - vel; // this reverse call forces us to implement this function in the derived classes
	}
	if (hierarchy != hierarchy_t::output) {
		above->applyUpdate(pars, input);
	}
	return gamma;
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