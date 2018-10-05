#include "stdafx.h"
#include "CNetLayer.h"

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN) : NOUT(_NOUT), NIN(_NIN), actSave(_NOUT, 1), deltaSave(_NOUT, 1){
	assignActFunc(actfunc_t::NONE);
	actSave.setZero();
	deltaSave.setZero();
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
}

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : NOUT(_NOUT), NIN(_NIN), actSave(_NOUT, 1), deltaSave(_NOUT, 1) {
	assignActFunc(type);
	actSave.setZero();
	deltaSave.setZero();
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
}

CNetLayer::CNetLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower): NOUT(_NOUT), actSave(_NOUT, 1), deltaSave(_NOUT, 1) {
	actSave.setZero();
	deltaSave.setZero();
	assignActFunc(type);
	NIN = lower.getNOUT();
	below = &lower;
	below->connectAbove(this);
	hierarchy = hierarchy_t::output;
}

layer_t CNetLayer::whoAmI() const {
	return layer_t::cnet;
}

uint32_t CNetLayer::getFeatures() const {
	return 1;
}

MAT CNetLayer::getDACT() const {
	return {std::move(actSave.unaryExpr(dact))};
}

MAT CNetLayer::getACT() const {
	return {std::move(actSave.unaryExpr(act))};
}

void CNetLayer::connectAbove(CNetLayer* ptr) {
	above = ptr;
	if (hierarchy != hierarchy_t::input)
		hierarchy = hierarchy_t::hidden; // change from output to hidden
}

// assign the activation function
void CNetLayer::assignActFunc(actfunc_t type) {
	activationType = type; // save type
	switch (type) {
		case actfunc_t::RELU : 
			act = &ReLu;
			dact = &DReLu;
			break;
		case actfunc_t::SIG :
			act = &Sig;
			dact = &DSig;
			break;
		case actfunc_t::TANH:
			act = &Tanh;
			dact = &DTanh;
			break;
		case actfunc_t::NONE:
			act = &iden; // nothing
			dact = &DIden;
			break;
	}
}
// save to file
ostream& operator<<(ostream& os, const CNetLayer& toSave) {
	os << static_cast<int32_t>(toSave.whoAmI()) << endl; // first line - type identification - to be used in later versions
	toSave.saveMother(os);
	toSave.saveToFile(os);
	return os;
}

void CNetLayer::saveMother(ostream& os) const {
	os << static_cast<int32_t>(activationType) << "\t" << NOUT << "\t" << NIN << "\t"<< static_cast<int32_t>(hierarchy)<<endl;
}
ifstream& operator >> (ifstream& in, CNetLayer& toReconstruct) {
	int32_t type;
	in >> type;
	assert(type == static_cast<int32_t>(toReconstruct.whoAmI()));

	toReconstruct.reconstructMother(in);
	toReconstruct.loadFromFile(in);
	return in;
}

void CNetLayer::reconstructMother(ifstream& in) {
	
	int32_t actType;
	in >> actType;
	activationType = static_cast<actfunc_t>(actType);
	assignActFunc(activationType);
	in >> NOUT;
	in >> NIN;
	in >> actType;
	hierarchy = static_cast<hierarchy_t>(actType);
	actSave = MAT(NOUT, 1);
	deltaSave = MAT(NOUT, 1);
	actSave.setZero();
	deltaSave.setZero();
}
