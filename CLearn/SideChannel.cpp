#include "stdafx.h"
#include "SideChannel.h"


SideChannel::SideChannel(size_t NIN, size_t _sideChannelSize) : sideChannelSize(_sideChannelSize),
	DiscarnateLayer(NIN+_sideChannelSize, NIN, actfunc_t::NONE) {
	
	sideChannelMatrix = MAT(_sideChannelSize, 1);
	sideChannelMatrix.setZero();
	deltaSave = MAT(sideChannelSize, 1);
	deltaSave.setZero();
}

SideChannel::SideChannel(CNetLayer& lower, size_t _sideChannelSize) 
	: sideChannelSize(_sideChannelSize), DiscarnateLayer(lower.getNOUT()+_sideChannelSize, actfunc_t::NONE, lower){
	sideChannelMatrix = MAT(_sideChannelSize, 1);
	sideChannelMatrix.setZero();
	deltaSave = MAT(sideChannelSize, 1);
	deltaSave.setZero();
}

SideChannel::~SideChannel() {}

void SideChannel::forProp(MAT& in, bool saveActivation, bool recursive) {
	in.conservativeResize(getNIN() + sideChannelSize,1);
	in.bottomRows(sideChannelSize) = sideChannelMatrix;
	if (recursive && getHierachy() != hierarchy_t::output)
		above->forProp(in, saveActivation, true);
}
void SideChannel::backPropDelta(MAT& delta, bool recursive) {
	//// ATTENTION - WE ONLY COPY SIDE CHANNEL-SPECIFIC DELTAS
	//// THIS IS SPECIAL BEHAVIOUR TO CUT COMP COST
	deltaSave = delta.bottomRows(sideChannelSize);
	// proceed as normal
	delta.conservativeResize(getNOUT() - sideChannelSize,1);
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