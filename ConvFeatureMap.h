#pragma once
#include "defininitions.h"
#include "FeatureMap.h"
#include "ConvolutionalLayer.h"

#ifndef CNET_CONVFEATUREMAP
#define CNET_CONVFEATUREMAP

class ConvFeatureMap : public FeatureMap{
	public:
		ConvFeatureMap(size_t featureNr, size_t feature_NOUTXY, size_t feature_NINXY, size_t feature_kernelXY, uint32_t feature_stride, actfunc_t type);
		ConvFeatureMap(size_t featureNr, size_t feature_NOUTXY, size_t feature_kernelXY, uint32_t feature_stride, actfunc_t type, CNetLayer& const lower);

		~ConvFeatureMap();
		inline layer_t whoAmI() const { return layer_t::convfeatureMap; };
};
#endif