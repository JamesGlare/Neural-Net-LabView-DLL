#pragma once
#include "defininitions.h"
#include "CNetLayer.h"

#ifndef CNET_CONVFEATUREMAP
#define CNET_CONVFEATUREMAP

class ConvFeatureMap : public CNetLayer{
	public:
		ConvFeatureMap(size_t features, size_t NIN, size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type);
		ConvFeatureMap(size_t features, size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);

		~ConvFeatureMap();

		void forProp(MAT& in, bool saveActivation, bool recursive); // recursive
		void backPropDelta(MAT& const delta); // recursive
		void applyUpdate(learnPars pars, MAT& const input); // recursive
	private:
		void saveToFile(ostream& os) const;
		void loadFromFile(ifstream& in);
};
#endif