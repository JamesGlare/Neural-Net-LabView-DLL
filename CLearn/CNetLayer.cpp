#include "stdafx.h"
#include "CNetLayer.h"

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN) : NOUT(_NOUT), NIN(_NIN), actSave(_NOUT, 1), deltaSave(_NOUT, 1){
	assignActFunc(actfunc_t::NONE);
	actSave.setZero();
	deltaSave.setZero();
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
	layerNumber = 0;
}

CNetLayer::CNetLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : NOUT(_NOUT), NIN(_NIN), actSave(_NOUT, 1), deltaSave(_NOUT, 1) {
	assignActFunc(type);
	actSave.setZero();
	deltaSave.setZero();
	below = NULL;
	above = NULL;
	hierarchy = hierarchy_t::input;
	layerNumber = 0;
}

CNetLayer::CNetLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower): NOUT(_NOUT), actSave(_NOUT, 1), deltaSave(_NOUT, 1) {
	actSave.setZero();
	deltaSave.setZero();
	assignActFunc(type);
	NIN = lower.getNOUT();
	below = &lower;
	below->connectAbove(this);
	layerNumber = below->getLayerNumber() + 1;
	hierarchy = hierarchy_t::output;
}

MAT CNetLayer::getDACT() const {
	return actSave.unaryExpr(dact);
}

MAT CNetLayer::getACT() const {
	return actSave.unaryExpr(act);
}

MAT CNetLayer::getDelta() const {
	return deltaSave;
}

void CNetLayer::connectAbove(CNetLayer* ptr) {
	above = ptr;
	//if (hierarchy != hierarchy_t::input)
	//	hierarchy = hierarchy_t::hidden; // change from output to hidden
}

void CNetLayer::connectBelow(CNetLayer* ptr) {
	below = ptr;
	//if(hierarchy !=output)
	//	hi
}
// Reset Hierarchy and LayerNumber
void CNetLayer::checkHierarchy(bool recursive) {
	if (below) {
		// either hidden or output
		layerNumber = below->getLayerNumber() + 1;
		if (above) {
			// definitely hidden
			hierarchy = hierarchy_t::hidden;
			if(recursive)
				above->checkHierarchy(true); 
		} else {
			// definitely output
			hierarchy = hierarchy_t::output;
		}
	} else {
		// definitely input or invalid single-layer network
		hierarchy = hierarchy_t::input;
		layerNumber = 0;
		if (above && recursive) {
			above->checkHierarchy(true);
		}
	}
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
		case actfunc_t::SOFTPLUS:
			act = &SoftPlus;
			dact = &DSoftPlus;
			break;
		case actfunc_t::LEAKYRELU:
			act = &LeakyReLu;
			dact = &DLeakyReLu;
			break;
		default:
			act = &ReLu;
			dact = &DReLu;
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
	os << static_cast<int32_t>(activationType) << " " << NOUT << " " << NIN << " "<< static_cast<int32_t>(hierarchy) << " " << layerNumber<<endl;
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
	
	int32_t temp;
	in >> temp;
	activationType = static_cast<actfunc_t>(temp);
	assignActFunc(activationType);
	in >> NOUT;
	in >> NIN;
	in >> temp;
	hierarchy = static_cast<hierarchy_t>(temp);
	// if we load partial networks or single layers from other chains,
	// the linkChain function will reset the network structure.
	in >> layerNumber; //... and also the layerNumber

	actSave = MAT(NOUT, 1);
	deltaSave = MAT(NOUT, 1);
	actSave.setZero();
	deltaSave.setZero();
}

void CNetLayer::changeActFunc( CNetLayer& changeMyActivation, actfunc_t type)
{
	changeMyActivation.assignActFunc(type);
}
