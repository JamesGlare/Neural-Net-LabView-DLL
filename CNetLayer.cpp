#include "stdafx.h"
#include "CNetLayer.h"

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN) : NOUT(_NOUT), NIN(_NIN){
	assignActFunc(actfunc_t::NONE);
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
}

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : NOUT(_NOUT), NIN(_NIN) {
	assignActFunc(type);
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
};
CNetLayer::CNetLayer(size_t _NOUT, actfunc_t type, CNetLayer& const lower): NOUT(_NOUT) {
	assignActFunc(type);
	NIN = lower.getNOUT();
	below = &lower;
	below->connectAbove(this);
	hierarchy = hierarchy_t::output;
}
layer_t CNetLayer::whoAmI() const {
	return layer_t::cnet;
}

MAT CNetLayer::getDACT() const {
	if (activationType != actfunc_t::NONE) {
		return actSave.unaryExpr(dact);
	}
	else {
		return MAT::Constant(NOUT,1,1);
	}
}

MAT CNetLayer::getACT() const {
	if (activationType != actfunc_t::NONE) {
		return actSave.unaryExpr(act);
	}
	else {
		return actSave;
	}
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
	os << to_string(toSave.whoAmI()) << endl; // first line - type identification - to be used in later versions
	toSave.saveMother(os);
	toSave.saveToFile(os);
	return os;
}

void CNetLayer::saveMother(ostream& os) const {
	os << static_cast<int32_t>(activationType) << "\t" << NOUT << "\t" << NIN << "\t"<< static_cast<int32_t>(hierarchy)<<endl;
}
ifstream& operator >> (ifstream& in, CNetLayer& toReconstruct) {
	toReconstruct.reconstructMother(in);
	toReconstruct.loadFromFile(in);
	return in;
}

void CNetLayer::reconstructMother(istream& is) {
	
	int32_t temp;
	is >> temp;
	activationType = static_cast<actfunc_t>(temp);
	assignActFunc(activationType);
	is >> NOUT;
	is >> NIN;
	is >> temp;
	hierarchy = static_cast<hierarchy_t>(temp);
}
