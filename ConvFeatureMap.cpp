#include "stdafx.h"
#include "ConvFeatureMap.h"
ConvFeatureMap::ConvFeatureMap(size_t featureNr, size_t feature_NOUTXY, size_t feature_NINXY, size_t feature_kernelXY, uint32_t feature_stride, actfunc_t type) 
	: FeatureMap(featureNr, featureNr*feature_NOUTXY*feature_NOUTXY, feature_NINXY*feature_NINXY, type) {
	for (size_t i = 0; i < featureNr; i++) {
		// 		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
		features.push_back( new ConvolutionalLayer(feature_NOUTXY, feature_NINXY, feature_kernelXY, feature_stride,1, actfunc_t::NONE)); // thinks it's an input layer ... 
	}
}

ConvFeatureMap::ConvFeatureMap(size_t featureNr, size_t feature_NOUTXY, size_t feature_kernelXY, uint32_t feature_stride, actfunc_t type, CNetLayer& const lower)
	: FeatureMap(featureNr, featureNr*feature_NOUTXY*feature_NOUTXY, type, lower) {
	for (size_t i = 0; i < featureNr; i++) {
		// 		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
		features.push_back(new ConvolutionalLayer(feature_NOUTXY,  feature_kernelXY, feature_stride,1,  actfunc_t::NONE, lower)); // thinks it's an input layer ... 
	}
}

ConvFeatureMap::~ConvFeatureMap() {
	for (size_t i = 0; i < features.size(); i++) {
		// 		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
		((ConvolutionalLayer*)features[i])->~ConvolutionalLayer();
	}
}

