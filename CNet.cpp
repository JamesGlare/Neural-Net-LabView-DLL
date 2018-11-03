#include "stdafx.h"
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "AntiConvolutionalLayer.h"
#include "MaxPoolLayer.h"
#include "PassOnLayer.h"
#include "DropoutLayer.h"

CNet::CNet(size_t NIN) :  NIN(NIN) {
	layers = vector<CNetLayer*>(); // to be filled with layers
	srand(42); // constant seed

	mixtureDensity = nullptr;
}
void CNet::debugMsg(fREAL* msg) {
	msg[0] = layers[0]->getNOUT();
	msg[1] = layers[0]->getNIN();
	msg[2] = layers[1]->getNOUT();
	msg[3] = layers[1]->getNIN();
}

size_t CNet::addFullyConnectedLayer(size_t NOUT, actfunc_t type) {
	// now we need to check if there is a layer already
	if (getLayerNumber() > 0) { // .. so there is one
		FullyConnectedLayer* fcl =  new FullyConnectedLayer(NOUT,  type,  *(layers.back())); // don't want to forward declare this..
		layers.push_back(fcl);
	}
	else {
		FullyConnectedLayer* fcl = new FullyConnectedLayer(NOUT, NIN,type);
		layers.push_back(fcl);
	}
	return getLayerNumber();
}
size_t CNet::addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t sideChannels, actfunc_t type) {
	// At the moment, I only allow for square-shaped input.
	// this may need to change in the future.
	if (getLayerNumber() > 0) {
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, kernelXY, stride, features, sideChannels, type, *(layers.back()));
		layers.push_back(cl);
	} else {
		// then it's the input layer
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, sqrt(NIN), kernelXY, stride, features, sideChannels, type);
		layers.push_back(cl);
	}
	return getLayerNumber();
}
size_t CNet::addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t sideChannels, actfunc_t type) {
	if (getLayerNumber() > 0) {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, kernelXY, stride, features, sideChannels, type, *(layers.back()));
		layers.push_back(acl);
	} else {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, sqrt(NIN/features), kernelXY, stride, features, sideChannels, type);
		layers.push_back(acl);
	}
	return getLayerNumber();
}

size_t CNet::addPassOnLayer( actfunc_t type) {
	if (getLayerNumber() > 0) {
		PassOnLayer* pol = new PassOnLayer(type, *(layers.back()));
		layers.push_back(pol);
	} else {
		PassOnLayer* pol = new PassOnLayer(NIN, NIN, type);
		layers.push_back(pol);
	}
	return getLayerNumber(); 
}
size_t CNet::addDropoutLayer(fREAL ratio) {
	if (getLayerNumber() > 0) {
		DropoutLayer* dl = new DropoutLayer(ratio, *(layers.back()) );
		layers.push_back(dl);
	} else {
		DropoutLayer* dl = new DropoutLayer(ratio, NIN);
		layers.push_back(dl);
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
	for (std::vector< CNetLayer* >::iterator it = layers.begin(); it != layers.end(); ++it) {
		delete (*it);
	}
	layers.clear();
	if (mixtureDensity)
		delete mixtureDensity;
}

size_t CNet::getNOUT() const {
	if (getLayerNumber() > 0) {
		if (!mixtureDensity)
			return layers.back()->getNOUT();
		else
			return mixtureDensity->getNOUT();
	}
	else {
		return 0;
	}
}

void CNet::saveToFile(string filePath) const {
	for (size_t i = 0; i < getLayerNumber(); ++i) {
		ofstream file(filePath+"\\CNetLayer_"+ to_string(i) + ".dat");
		if (file.is_open()) {
			file << (*layers[i]);
		}
		file.close();
	}
}
void CNet::loadFromFile(string filePath) {
	for(size_t i =0; i< getLayerNumber(); ++i) {
		ifstream file(filePath + "\\CNetLayer_" + to_string(i) + ".dat");
		if (file.is_open()) {
			file >> (*layers[i]);
		}
		file.close();
	}
}
// Simply output the network
fREAL CNet::forProp(MAT& in, const MAT& outDesired, const learnPars& pars) {
	layers.front()->forProp(in, false, true);
	if (!mixtureDensity)
		return l2_error(errorMatrix(in, outDesired));
	else {
		mixtureDensity->updateParameters(in); // resizes output
		//mixtureDensity->getParameters(in);
		//mixtureDensity->maxMixtureCoefficient(in); // turns a K*(L+2) matrix into a (L,1)
		mixtureDensity->conditionalMean(in);
		return mixtureDensity->negativeLogLikelihood(outDesired);
	}
		
}

// Backpropagation 
fREAL CNet::backProp(MAT& input, MAT& outDesired, const learnPars& pars) {
	// (0) Check in- & output
	if (!input.allFinite()
		|| !outDesired.allFinite()) {
		return 1; // just skip this sample 
	}
	MAT diffMatrix;
	fREAL errorOut = 0.0f;
	// (1) Propagate in forward direction (with saveActivations == true)
	MAT outPredicted(input);
	layers.front()->forProp(outPredicted, true, true);
	// (2) calculate error matrix and error
	if (mixtureDensity){
		mixtureDensity->updateParameters(outPredicted); // resizes output
		//mixtureDensity->getParameters(outPredicted);
		mixtureDensity->maxMixtureCoefficient(outPredicted);
		//mixtureDensity->conditionalMean(outPredicted); // this reduces the (K,L+2)-matrix to a (L,1)-sized matrix !
		//errorOut = mixtureDensity->negativeLogLikelihood(outDesired);
		errorOut = l2_error(errorMatrix(outPredicted, outDesired));
		diffMatrix = move(mixtureDensity->computeErrorGradient(outDesired));
	} else {
		diffMatrix = move(errorMatrix(outPredicted, outDesired));
		errorOut = l2_error(diffMatrix);
	}

	// (3) back propagate the deltas
	getLast()->backPropDelta(diffMatrix, true);
	// (4) Apply update
	getFirst()->applyUpdate(pars, input, true);
	// (5) Write predicted output to output matrix
	
	// ... DONE
	outDesired = outPredicted;
	return errorOut;
}
void CNet::addMixtureDensity(size_t K, size_t L, size_t Blocks) {
	assert(getNOUT() == (L + 2)*K);
	if(!mixtureDensity)
		this->mixtureDensity = new MixtureDensityModel(K, L, Blocks, getLast()->getNOUT());
}
void CNet::inquireDimensions(size_t layer, size_t& rows, size_t& cols) const {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			MATIND layerDimension = dynamic_cast<PhysicalLayer*>(layers[layer])->layerDimensions();
			rows = layerDimension.rows;
			cols = layerDimension.cols;
		}
	}
}
void CNet::copyNthLayer(size_t layer, fREAL* const toCopyTo) const {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			dynamic_cast<PhysicalLayer*>(layers[layer])->copyLayer(toCopyTo);
		}
	}
}
void CNet::setNthLayer(size_t layer, fREAL* const copyFrom) {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			dynamic_cast<PhysicalLayer*>(layers[layer])->setLayer(copyFrom);
		}
	}
}
MAT CNet::errorMatrix(const MAT& outPrediction, const MAT& outDesired) {
	return outPrediction - outDesired; // force Visual C++ to return without temporary - since RVO doesn't work ???!
}
fREAL CNet::l2_error(const MAT& diff) {
	fREAL sum = cumSum(matNorm(diff));
	//if (sum > 0.0f)
		return 0.5f*sqrt(sum); //  / sqrt(sum)
	//else
	//	return 0.0f;
}


