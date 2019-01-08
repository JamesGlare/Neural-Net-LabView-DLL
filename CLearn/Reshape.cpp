#include "stdafx.h"
#include "Reshape.h"

Reshape::Reshape(size_t NIN) : DiscarnateLayer(NIN, NIN, actfunc_t::NONE) {

}

Reshape::Reshape(CNetLayer& lower) : DiscarnateLayer(lower.getNOUT(), actfunc_t::NONE, lower) {

}

Reshape::~Reshape() {}

layer_t Reshape::whoAmI() const { return layer_t::reshape; }

void Reshape::forProp(MAT& in, bool saveActivation, bool recursive) {

	size_t squaresize = sqrt(in.size());
	in.resize(squaresize, squaresize);
	in.transposeInPlace();
	in.resize(in.size(), 1);
	if (getHierachy() != hierarchy_t::output && recursive)
		above->forProp(in, saveActivation, true);
}
void Reshape::backPropDelta(MAT& delta, bool recursive) {
	if (getHierachy() != hierarchy_t::input) {
		size_t squaresize = sqrt(delta.size());
		delta.resize(squaresize, squaresize);
		delta.transposeInPlace();
		delta.resize(delta.size(), 1);
		if (recursive)
			below->backPropDelta(delta, true);
	}
}

void Reshape::saveToFile(ostream& os) const {

}

void Reshape::loadFromFile(ifstream& in) {

}