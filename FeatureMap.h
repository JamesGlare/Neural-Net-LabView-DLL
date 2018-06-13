#pragma once
#include "defininitions.h"
#include "CNetLayer.h"
#include "PhysicalLayer.h"

#ifndef CNET_FEATUREMAP
#define CNET_FEATUREMAP
/* Virtual abstract class for feature maps
*/

class FeatureMap : public CNetLayer{
	public:
		FeatureMap(size_t featureNr, size_t NOUT, size_t NIN, actfunc_t type);
		FeatureMap(size_t featureNr, size_t NOUT,  actfunc_t type, CNetLayer& const lower);

		virtual ~FeatureMap() {};
		void forProp(MAT& in, bool saveActivation, bool recursive); // recursive
		void backPropDelta(MAT& delta, bool recursive); // recursive
		void applyUpdate(learnPars& const pars, MAT& const input, bool recursive); // recursive
	protected:
		void saveToFile(ostream& os) const;
		void loadFromFile(ifstream& in);
		vector<PhysicalLayer*> features;
		size_t feature_NIN;
		size_t feature_NOUT;
};
#endif