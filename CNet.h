#pragma once

#include "defininitions.h"
#include <memory>
#include "CNETLayer.h"

typedef std::unique_ptr<CNetLayer> layerPtr;
 

#ifndef CNET_CNET
#define CNET_CNET

class CNet {
	public:
		CNet(size_t NIN);
		~CNet();
		// add layers
		size_t addFullyConnectedLayer(size_t NOUT, actfunc_t type);
		size_t addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, actfunc_t type);
		size_t addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, actfunc_t type);
		size_t addPoolingLayer(size_t maxOverXY, pooling_t type);
		size_t addPassOnLayer(actfunc_t type);

		// forProp
		fREAL forProp(MAT& in, const learnPars& pars, const MAT& outDesired);
		// backprop
		fREAL backProp(MAT& in, MAT& outDesired, const learnPars& pars);
		// save to file
		void saveToFile(string filePath) const;
		void loadFromFile(string filePath);

		void debugMsg(fREAL* msg);
		// Getter functions
		inline size_t getLayerNumber() const { return layers.size(); };
		inline size_t getNIN() const { return NIN; };
		void copyNthLayer(uint32_t layer, fREAL* const toCopyTo);
		size_t getNOUT() const;
		inline CNetLayer* getLast() const { return layers.back(); };
		inline CNetLayer* getFirst() const { return layers.front(); };
	private:
		// error related functions
		MAT errorMatrix(const MAT& outPrediction, const MAT& outDesired);
		fREAL error(const MAT& diff);

		size_t NIN;
		vector<CNetLayer*> layers;

};

#endif 
