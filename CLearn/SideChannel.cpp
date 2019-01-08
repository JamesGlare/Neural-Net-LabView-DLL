#include "stdafx.h"
#include "SideChannel.h"

SideChannel::SideChannel(CNetLayer& lower, size_t _sideChannelSize) 
	: sideChannelSize(_sideChannelSize), DiscarnateLayer(lower.getNOUT()+_sideChannelSize, actfunc_t::NONE, lower){
	sideChannelMatrix = MAT(_sideChannelSize, 1);
	sideChannelMatrix.setZero();
}

SideChannel::~SideChannel() {}

void SideChannel::forProp(MAT& in, bool saveActivation, bool recursive) {
	in.conservativeResize(in.size() + sideChannelSize,1);
	in.bottomRows(sideChannelSize) = sideChannelMatrix;
	if (recursive && getHierachy() != hierarchy_t::output)
		above->forProp(in, saveActivation, true);
}
void SideChannel::backPropDelta(MAT& delta, bool recursive) {
	delta.conservativeResize(delta.size() - sideChannelSize,1);
	if (recursive && getHierachy() != hierarchy_t::input)
		below->backPropDelta(delta, true);
}
layer_t SideChannel::whoAmI() const { return layer_t::sideChannel; }

void SideChannel::preFeed(const MAT& toStore) {
	assert(toStore.rows() == sideChannelMatrix.rows());
	assert(toStore.cols() == sideChannelMatrix.cols());
	sideChannelMatrix = toStore;
}

void SideChannel::saveToFile(ostream& os) const {
	os << sideChannelSize;
}

void SideChannel::loadFromFile(ifstream& in) {
	in >> sideChannelSize;
	sideChannelMatrix = MAT(sideChannelSize, 1);
	sideChannelMatrix.setZero();
}