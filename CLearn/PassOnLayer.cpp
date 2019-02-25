#include "stdafx.h"
#include "PassOnLayer.h"

PassOnLayer::PassOnLayer(size_t NOUT, size_t NIN, actfunc_t type) : DiscarnateLayer(NOUT, NIN, type) {

}
PassOnLayer::PassOnLayer(size_t NOUT, actfunc_t type, CNetLayer& lower) : DiscarnateLayer(NOUT, type, lower) {

}
PassOnLayer::PassOnLayer(actfunc_t type, CNetLayer& lower) : DiscarnateLayer(lower.getNOUT(), type, lower) {

}

layer_t PassOnLayer::whoAmI() const {
	return layer_t::passOn;
}

PassOnLayer::~PassOnLayer() {}
void PassOnLayer::init() {

}
void PassOnLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	if (training) {
		actSave = inBelow;
	}
	inBelow = inBelow.unaryExpr(act);
	if (getHierachy() != hierarchy_t::output && recursive) {
			above->forProp(inBelow, training, true);
	} 
}

void PassOnLayer::backPropDelta(MAT& delta, bool recursive) {
	
	delta = (getDACT()).cwiseProduct(delta); // need this to collect delta
	deltaSave = delta;
	if(getHierachy() != hierarchy_t::input && recursive)
		below->backPropDelta(delta, true);
}

void PassOnLayer::saveToFile(ostream& os) const {
}
void PassOnLayer::loadFromFile(ifstream& in) {

}
