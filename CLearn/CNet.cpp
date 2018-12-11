#include "stdafx.h"
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "AntiConvolutionalLayer.h"
#include "MaxPoolLayer.h"
#include "PassOnLayer.h"
#include "DropoutLayer.h"
#include "Reshape.h"

CNet::CNet(size_t NIN) :  NIN(NIN) {
	layers = vector<CNetLayer*>(); // to be filled with layers
	srand(42); // constant seed
}
void CNet::debugMsg(fREAL* msg) {
	msg[0] = layers[0]->getNOUT();
	msg[1] = layers[0]->getNIN();
	msg[2] = layers[1]->getNOUT();
	msg[3] = layers[1]->getNIN();
}

void CNet::addFullyConnectedLayer(size_t NOUT, fREAL kappa, actfunc_t type) {
	// now we need to check if there is a layer already
	if (getLayerNumber() > 0) { // .. so there is one
		FullyConnectedLayer* fcl =  new FullyConnectedLayer(NOUT, kappa, type, *(getLast())); // don't want to forward declare this..
		layers.push_back(fcl);
	} else {
		// then it's the input layer

		FullyConnectedLayer* fcl = new FullyConnectedLayer(NOUT, NIN, kappa,type);
		layers.push_back(fcl);
	}
}
void CNet::addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t sideChannels, actfunc_t type) {
	// At the moment, I only allow for square-shaped input.
	// this may need to change in the future.
	if (getLayerNumber() > 0) {
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, kernelXY, stride, features, sideChannels, type, *(getLast()));
		layers.push_back(cl);
	} else {
		// then it's the input layer
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, sqrt(NIN-sideChannels), kernelXY, stride, features, sideChannels, type);
		layers.push_back(cl);
	}
}
void CNet::addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t outBoxes, size_t sideChannels, actfunc_t type) {
	if (getLayerNumber() > 0) {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, kernelXY, stride, features, outBoxes, sideChannels, type, *(getLast()));
		layers.push_back(acl);
	} else {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, sqrt(NIN/features), kernelXY, stride, features, outBoxes, sideChannels, type);
		layers.push_back(acl);
	}
}

void CNet::addPassOnLayer( actfunc_t type) {
	if (getLayerNumber() > 0) {
		PassOnLayer* pol = new PassOnLayer(type, *(getLast()));
		layers.push_back(pol);
	} else {
		PassOnLayer* pol = new PassOnLayer(NIN, NIN, type);
		layers.push_back(pol);
	}
}
void CNet::addReshape() {
	if (getLayerNumber() > 0) {
		Reshape* rs = new Reshape(*(getLast()));
		layers.push_back(rs);
	} else {
		Reshape* rs = new Reshape(NIN);
		layers.push_back(rs);
	}
}
void CNet::addDropoutLayer(fREAL ratio) {
	if (getLayerNumber() > 0) {
		DropoutLayer* dl = new DropoutLayer(ratio, *(getLast()));
		layers.push_back(dl);
	} else {
		DropoutLayer* dl = new DropoutLayer(ratio, NIN);
		layers.push_back(dl);
	}
}
void CNet::addPoolingLayer(size_t maxOverXY, pooling_t type) {
	switch (type) {
		case pooling_t::max:
			if (getLayerNumber() > 0) {
				MaxPoolLayer* mpl = new MaxPoolLayer(maxOverXY, *(getLast()));
				layers.push_back(mpl);
			} else {
				MaxPoolLayer* mpl = new MaxPoolLayer(sqrt(NIN), maxOverXY);
				layers.push_back(mpl);
			}
			break;
		case pooling_t::average: // not yet implemented
			break;
	}
}
void CNet::addMixtureDensity(size_t NOUT, size_t features, size_t BlockXY) {
	
	if (getLayerNumber() > 0 ) {
		MixtureDensityModel* mdm = new MixtureDensityModel(sqrt(NOUT), sqrt(NOUT), features, BlockXY, BlockXY, *(getLast()));
		layers.push_back(mdm);
	} else {
		// this is not really defined so don't do anything
	}
}

void CNet::shareLayers(CNet* const otherNet, uint32_t firstLayer, uint32_t lastLayer) {
	for (uint32_t i = firstLayer; i <= lastLayer; ++i) {
		if (i < otherNet->getLayerNumber()) {
		layers.push_back(otherNet->layers[i]);


			/*switch (layer->whoAmI()) {
				case layer_t::antiConvolutional:
					layers.push_back(dynamic_cast<AntiConvolutionalLayer*>(layer));
					break;
				case layer_t::convolutional: 
					layers.push_back(dynamic_cast<ConvolutionalLayer*>(layer));
					break;
				case layer_t::dropout:
					layers.push_back(dynamic_cast<DropoutLayer*>(layer));
					break;
				case layer_t::fullyConnected:
					layers.push_back(dynamic_cast<FullyConnectedLayer*>(layer));
					break;
				case layer_t::maxPooling:
					layers.push_back(dynamic_cast<MaxPoolLayer*>(layer));
					break;
				case layer_t::mixtureDensity:
					layers.push_back(dynamic_cast<MixtureDensityModel*>(layer));
					break;
				case layer_t::passOn:
					layers.push_back(dynamic_cast<PassOnLayer*>(layer));
					break;
				case layer_t::reshape:
					layers.push_back(dynamic_cast<Reshape*>(layer));
					break;
			}*/
		}
	}
}

void CNet::linkChain()
{
	if (getLayerNumber() > 1) {
		// set lowest and highest layer links
		getFirst()->connectBelow(NULL);
		getLast()->connectAbove(NULL);

		// rebuild connections 
		for (std::vector< CNetLayer* >::iterator it = layers.begin(); 
			it != layers.end() - 1; ++it) { // go until penultimate element
			(*it)->connectAbove(*(it + 1));
			(*(it + 1))->connectBelow(*it);
		}
		// make elements reset their hierarchy
		getFirst()->checkHierarchy(true);
	}
}

// Destructor
CNet::~CNet() {
	for (std::vector< CNetLayer* >::iterator it = layers.begin(); it != layers.end(); ++it) {
		delete (*it);
	}
	layers.clear();

}

size_t CNet::getNOUT() const {
	if (getLayerNumber() > 0) {
		return getLast()->getNOUT();
	} else {
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
		loadFromFile_layer(filePath, i);
	}
}
void CNet::loadFromFile_layer(string filePath, uint32_t layerNr) {
	ifstream file(filePath + "\\CNetLayer_" + to_string(layerNr) + ".dat");
	if (file.is_open()) {
		file >> (*layers[layerNr]);
	}
	file.close();
}
// Simply output the network
fREAL CNet::forProp(MAT& in, const MAT& outDesired, const learnPars& pars) {
	getFirst()->forProp(in, false, true);
	
	return l2_error(errorMatrix(in, outDesired));
}

// Backpropagation 
fREAL CNet::backProp(MAT& input, MAT& outDesired, const learnPars& pars) {

	// (0) Check in- & output
	/*if (!input.allFinite()
		|| !outDesired.allFinite()) {
		return 1; // just skip this sample 
	}*/
	// (0.5) Initialize error and difference matrix
	MAT diffMatrix;
	fREAL errorOut = 0.0f;

	// (1) Propagate in forward direction (with saveActivations == true)
	MAT outPredicted(input);
	getFirst()->forProp(outPredicted, true, true);
	
	// (2) calculate error matrix and error
	diffMatrix = move(errorMatrix(outPredicted, outDesired)); // delta =  estimate - target
	errorOut = l2_error(diffMatrix);

	// (3) back propagate the deltas
	getLast()->backPropDelta(diffMatrix, true);
	
	// (4) Apply update
	getFirst()->applyUpdate(pars, input, true);
	
	// (5) Write predicted output to output matrix
	outDesired = outPredicted;

	// DONE
	return errorOut;
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
void CNet::copyNthActivation(size_t layer, fREAL* const toCopyTo) const {
	if (layer < getLayerNumber()) {
		MAT temp = layers[layer]->getACT();
		temp.transposeInPlace();
		size_t rows = temp.rows();
		size_t cols = temp.cols();
		temp.resize(temp.size(), 1);
		copyToOut(temp.data(), toCopyTo, temp.size());
		
	}
}
void CNet::copyNthLayer(size_t layer, fREAL* const toCopyTo) const {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			MAT temp = dynamic_cast<PhysicalLayer*>(layers[layer])->copyLayer();
			temp.transposeInPlace();
			size_t rows = temp.rows();
			size_t cols = temp.cols();
			temp.resize(temp.size(), 1);
			copyToOut(temp.data(), toCopyTo, temp.size());
		}
	}
}
void CNet::setNthLayer(size_t layer, const MAT& newLayer) {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			dynamic_cast<PhysicalLayer*>(layers[layer])->setLayer(newLayer);
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


