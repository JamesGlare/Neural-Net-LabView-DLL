#include "stdafx.h"
#include "ConvFeatureMap.h"

FeatureMap::FeatureMap(size_t featureNr, size_t NOUT, size_t NIN, actfunc_t type) : CNetLayer(NOUT, NIN, type){
	features = vector<PhysicalLayer*>();
	assert(NOUT % featureNr == 0); 
	assert(NIN % featureNr == 0);
	feature_NIN = NIN;
	feature_NOUT = NOUT / featureNr;
}

FeatureMap::FeatureMap(size_t featureNr, size_t NOUT, actfunc_t type, CNetLayer& const lower) : CNetLayer(NOUT, lower.getNOUT(), type ) {
	features = vector<PhysicalLayer*>();
	assert(NOUT% featureNr == 0);
	assert(NIN % featureNr == 0);
	feature_NIN = NIN;
	feature_NOUT = NOUT / featureNr;
}
/* Features think they are output layers unless they are input layers. They don't apply any nonlinearities.
*/

void FeatureMap::forProp(MAT& in, bool saveActivation, bool recursive) {
	MAT temp(feature_NIN, 1);
	MAT out(NOUT, 1);
	
	for (size_t i = 0; i < features.size(); i++) {
		temp = in; 
		features[i]->forProp(temp, saveActivation, false);
		out.block(i*feature_NOUT, 0, feature_NOUT, 1) = temp;
	}
	if (hierarchy != hierarchy_t::output) {
		if (saveActivation) {
			actSave = out;
		}
		in = out.unaryExpr(act);
		// recursion

		if (recursive) {
			above->forProp(in, saveActivation, true);
		}
	}	
}
void FeatureMap::backPropDelta(MAT& delta, bool recursive) {
	//deltaSave = delta; // deltas are stored in the features

	MAT temp(feature_NOUT, 1);
	MAT out(NIN, 1);
	out.setZero();
	for (size_t i = 0; i < features.size(); i++) {
		temp = delta.block(i*feature_NOUT, 0, feature_NOUT, 1);
		features[i]->backPropDelta(temp, false); // temp gets only changed when this is NOT an input layer to save time
		if(hierarchy != input)
			out += temp;
	}
	delta = out / features.size();
	// recursion
	if (hierarchy != input && recursive) {
		below->backPropDelta(delta, true);
	}
}
void FeatureMap::applyUpdate(learnPars& const pars, MAT& const input, bool recursive) {
	for (size_t i = 0; i < features.size(); i++) {
		features[i]->applyUpdate(pars, input, false);
	}
	if (hierarchy != hierarchy_t::output && recursive) {
		above->applyUpdate(pars, input, true);
	}
}

void FeatureMap::saveToFile(ostream& os) const {
	os << features.front()->whoAmI() << "\t" << features.size() << "\t" << feature_NOUT<< "\t" << feature_NIN<<std::endl;
	for (size_t i = 0; i < features.size(); i++) {
		os << *features[i];
	}
}
void FeatureMap::loadFromFile(ifstream& in) {
	uint32_t temp;
	in >> temp; // whoAmI
	in >> temp; // feature size
	features = vector<PhysicalLayer*>(temp);
	in >> feature_NOUT;
	in >> feature_NIN;

	for (size_t i = 0; i < features.size(); i++) {
		//features[i] = new ConvolutionalLayer((uint32_t) sqrt(feature_NOUT), sqrt(feature_NIN), );
		in >> *features[i];
	}
}
