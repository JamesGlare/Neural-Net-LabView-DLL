#include "stdafx.h"
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "MaxPoolLayer.h"

CNet::CNet(size_t NIN) :  NIN(NIN) {
	layers = vector<CNetLayer*>(); // to be filled with layers
}

size_t CNet::addFullyConnectedLayer(size_t NOUT, actfunc_t type) {
	// now we need to check if there is a layer already
	if (getLayerNumber() > 0) { // .. so there is one
		FullyConnectedLayer* fcl =  new FullyConnectedLayer(NOUT,  type, -1.0/layers.back()->getNOUT(), 1.0 / layers.back()->getNOUT(), *(layers.back())); // don't want to forward declare this..
		layers.push_back(fcl);
	}
	else {
		FullyConnectedLayer* fcl = new FullyConnectedLayer(NOUT, NIN,type, -1.0/NIN, 1.0/NIN);
		layers.push_back(fcl);
	}
	return getLayerNumber();
}
size_t CNet::addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, actfunc_t type) {
	// At the moment, I only allow for square-shaped input.
	// this may need to change in the future.
	if (getLayerNumber() > 0) {
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, kernelXY, stride, type, *(layers.back()));
		layers.push_back(cl);
	} else {
		// then it's the input layer
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, sqrt(NIN), kernelXY, stride, type);
		layers.push_back(cl);
	}
	return getLayerNumber();
}

size_t CNet::addPoolingLayer(size_t maxOverXY, pooling_t type) {
	switch (type) {
		case pooling_t::max:
			if (getLayerNumber() > 0) {
				MaxPoolLayer* mpl = new MaxPoolLayer(maxOverXY, *(layers.back()));
				layers.push_back(mpl);
			} else {
				MaxPoolLayer* mpl = new MaxPoolLayer(sqrt(NIN), maxOverXY);
				layers.push_back(mpl);
			}
			break;
		case pooling_t::average:
			break;
	}
	return getLayerNumber();
}
// Destructor
CNet::~CNet() {
	for (size_t i = 0; i < getLayerNumber(); i++) {
		delete layers[i];
	}
}

size_t CNet::getNOUT() const {
	if (getLayerNumber() > 0) {
		return (*layers.back()).getNOUT();
	}
	else {
		return 0;
	}
}

void CNet::saveToFile(string filePath) const {
	for (size_t i = 0; i < getLayerNumber(); i++) {
		ofstream file(filePath+"CNetLayer_"+ to_string(i) + ".dat");
		if (file.is_open()) {
			file << (*layers[i]);
		}
		file.close();
	}
}
void CNet::loadFromFile(string filePath) {
	for(uint32_t i =0; i< getLayerNumber(); i++) {
		ifstream file(filePath + "CNetLayer_" + to_string(i) + ".dat");
		if (file.is_open()) {
			int type;
			file >> type; //First line has to be taken away - to be used in later versions
			file >> (*layers[i]);
		}
	}
}
// Simply output the network
fREAL CNet::forProp(MAT& in, MAT& const outDesired) {
	layers.front()->forProp(in, false);
	return error(errorMatrix(in, outDesired));
}
// Backpropagation 
fREAL CNet::backProp(MAT& const input, MAT& outDesired, learnPars pars) {
	// (1) for prop with saveActivations == true
	MAT outPredicted = input;
	(*layers.front()).forProp(outPredicted, true);
	// (2) calculate error matrix and error
	MAT diff = errorMatrix(outPredicted, outDesired);
	// (3) back propagate the deltas
	// (3.1) if Nesterov then set weights back now
	if (!pars.conjugate) {
		(*layers.front()).NesterovWeightSetback(pars);
	}
	// (3.2) back propagate
	(*layers.back()).backPropDelta(diff);
	// (4) Apply update
	if (diff.allFinite()) {
		fREAL gamma = (*layers.front()).applyUpdate(pars, input);
	}
	// ... DONE
	outDesired = outPredicted;

	return error(diff);
}
void CNet::resetConjugate(MAT& const input) {
	(*layers.front()).resetConjugate(input);
}


MAT CNet::errorMatrix(MAT& const outPrediction, MAT& const outDesired) {
	return outPrediction - outDesired;
}
fREAL CNet::error(MAT& const diff) {
	return 0.5*cumSum(matNorm(diff));
}


