#include "stdafx.h"
#include "PassOnLayer.h"

PassOnLayer::PassOnLayer(size_t NOUT, size_t NIN, actfunc_t type) : DiscarnateLayer(NOUT, NIN, type) {

}
PassOnLayer::PassOnLayer(size_t NOUT, actfunc_t type, CNetLayer& const lower) : DiscarnateLayer(NOUT, type, lower) {

}
PassOnLayer::PassOnLayer(actfunc_t type, CNetLayer& const lower) : DiscarnateLayer(lower.getNOUT(), type, lower) {

}

layer_t PassOnLayer::whoAmI() const {
	return layer_t::passOn;
}

PassOnLayer::~PassOnLayer() {}
void PassOnLayer::init() {

	actSave = MAT(NOUT, 1);
	actSave.setConstant(0);
	deltaSave = MAT(1, 1);
	deltaSave.setConstant(0);
}
void PassOnLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	if (training) {
		actSave = inBelow;
	}
	inBelow = inBelow.unaryExpr(act);
	if (hierarchy != hierarchy_t::output && recursive) {
			above->forProp(inBelow, training, true);
	} 
}

void PassOnLayer::backPropDelta(MAT& const delta, bool recursive) {
	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.

		MAT temp = (below->getDACT()).cwiseProduct(delta);
		if (hierarchy == hierarchy_t::output) {
			temp.cwiseProduct(this->getDACT());
		}
		if(recursive)
			below->backPropDelta(temp, true);
	}
}

void PassOnLayer::saveToFile(ostream& os) const {
}
void PassOnLayer::loadFromFile(ifstream& in) {

}
