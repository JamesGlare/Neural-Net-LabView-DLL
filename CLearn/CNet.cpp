#include "stdafx.h"
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "AntiConvolutionalLayer.h"
#include "MaxPoolLayer.h"
#include "PassOnLayer.h"
#include "DropoutLayer.h"
#include "Reshape.h"
#include "SideChannel.h"
#include "GaussianReparametrizationLayer.h"

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

void CNet::addFullyConnectedLayer(size_t NOUT, actfunc_t type) {
	// now we need to check if there is a layer already
	if (getLayerNumber() > 0) { // .. so there is one
		FullyConnectedLayer* fcl =  new FullyConnectedLayer(NOUT, type, *(getLast())); // don't want to forward declare this..
		layers.push_back(fcl);
	} else {
		// then it's the input layer
		FullyConnectedLayer* fcl = new FullyConnectedLayer(NOUT, NIN, type);
		layers.push_back(fcl);
	}
}

void CNet::addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride,  size_t outChannels, size_t inChannels, actfunc_t type) {
	// At the moment, I only allow for square-shaped input.
	// this may need to change in the future.
	if (getLayerNumber() > 0) {
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, kernelXY, stride,  outChannels, inChannels, type, *(getLast()));
		layers.push_back(cl);
	} else {
		// then it's the input layer
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, sqrt(NIN/inChannels), kernelXY, stride,  outChannels, inChannels, type);
		layers.push_back(cl);
	}
}

void CNet::addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride,  size_t outChannels, size_t inChannels, actfunc_t type) {
	if (getLayerNumber() > 0) {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, kernelXY, stride,  outChannels , inChannels, type, *(getLast()));
		layers.push_back(acl);
	} else {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, sqrt(NIN / inChannels), kernelXY, stride,  outChannels, inChannels, type);
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

void CNet::addSideChannel(size_t sideChannelSize) {
	if (getLayerNumber() > 0) {
		SideChannel* sc = new SideChannel(*(getLast()), sideChannelSize );
		layers.push_back(sc);
	} else {
		SideChannel* sc = new SideChannel(getNIN(), sideChannelSize);
		layers.push_back(sc);
	}
}

void CNet::addGaussianReparametrization()
{
	if (getLayerNumber() > 0) 
	{
		GaussianReparametrizationLayer* grpl = new GaussianReparametrizationLayer( *(getLast() ) );
		layers.push_back(grpl);
	}
	else
	{
		GaussianReparametrizationLayer* grpl = new GaussianReparametrizationLayer(getNIN());
		layers.push_back(grpl);
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

void CNet::addPoolingLayer(size_t maxOverXY, size_t channels, pooling_t type) {
	switch (type) {
		case pooling_t::max:
			if (getLayerNumber() > 0) {
				MaxPoolLayer* mpl = new MaxPoolLayer(maxOverXY, channels, *(getLast()));
				layers.push_back(mpl);
			} else {
				MaxPoolLayer* mpl = new MaxPoolLayer(sqrt(NIN), maxOverXY,  channels);
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
		// layer number mismatch will indicate missing MixtureDensity.
	}
}
/* Share layers in the range [firstLayer, lastLayer] with the CNet at ptr.
*/
void CNet::shareLayers(CNet* const otherNet, uint32_t firstLayer, uint32_t lastLayer) {
	for (uint32_t i = firstLayer; i <= lastLayer; ++i) {
		if (i < otherNet->getLayerNumber()) {
			layers.push_back(otherNet->layers[i]);
		}
	}
}
/* Dynamically relinks the chain at runtime.
*/
void CNet::linkChain(){
	if (getLayerNumber() > 1) {
		// set lowest and highest layer links
		getFirst()->connectBelow(NULL);
		getLast()->connectAbove(NULL);

		// rebuild connections 
		for (vector< CNetLayer* >::iterator it = layers.begin(); 
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
	for (vector< CNetLayer* >::iterator it = layers.begin(); it != layers.end(); ++it) {
		delete *it;
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

/* Initialization routine
*/
void CNet::initToUnitVariance(size_t batchSize) {

	// (1) For each physical layer
	for (uint32_t i = 0; i < getLayerNumber(); ++i) {
		if (isPhysical(i)) {
			
			PhysicalLayer* layer = dynamic_cast<PhysicalLayer*>(layers[i]);
			size_t nin = layer->getNIN();
			size_t nout = layer->getNOUT();
			BatchBuffer buffer(nout, nin);
			//(2) Draw Random numbers
			for (size_t k = 0; k < batchSize; ++k) {
				MAT in(nin, 1);
				in.setRandom();
				in.unaryExpr(&abs<fREAL>); // constrain noise to [0,1]
				layer->forProp(in, true, false);
				in = layer->getACT(); // don't apply non-linearity
				buffer.updateBuffer(in);
			}
			buffer.updateModel();
			layer->constrainToMax(buffer.batchMean(), buffer.batchMax());
		}

	}
}
size_t CNet::layerDimensionError() const{

	uint32_t wrongLayer = 0;
	// test first layer
	CNetLayer* firstLayer = layers[0];
	if (firstLayer->getNIN() != getNIN())
		return -1;

	while (wrongLayer < getLayerNumber()-1) {
		CNetLayer* layer = layers[wrongLayer];
		CNetLayer* nextLayer = layers[wrongLayer +1];

		if (layer->getNOUT() != nextLayer->getNIN()) {
			return wrongLayer;
		}
			++wrongLayer;
	}
	// last layer and NOUT by definition correct
	return 0;
}
// Simply output the network
fREAL CNet::forProp(MAT& in, const MAT& outDesired, bool saveAct) {

	// (1) Forward propagation
	getFirst()->forProp(in, saveAct, true);
	// (2) return the error 
	return l2_error(l2_errorMatrix(in- outDesired));
}

// Prefeeding function
void CNet::preFeedSideChannel(const MAT& sideChannelInput) {

	// Go through each layer - like this we don't need flags and
	// don't need to reroute the chain all the time when saved/restored.
	for (CNetLayer* layer : layers) {
		if (layer->whoAmI() == layer_t::sideChannel) {
			dynamic_cast<SideChannel*>(layer)->preFeed(sideChannelInput);
			break;
		}
	}

}

size_t CNet::getSideChannelSize() {
	size_t result = 0;
	for (CNetLayer* layer : layers) {
		if (layer->whoAmI() == layer_t::sideChannel) {
			result = dynamic_cast<SideChannel*>(layer)->getSidechannelSize();
			break;
		}
	}
	return result;
}

// Backpropagation 
fREAL CNet::backProp(MAT& input, MAT& outDesired, const learnPars& pars, bool deltaProvided) {

	// (0) Check if Input contains no NaN's or Infinities.
	// ... 
	// ...
	
	// (0.5) Initialize error and difference matrix
	MAT diffMatrix;
	fREAL errorOut = 0.0f;
	
	// (1) Propagate in forward direction (with saveActivations == true)
	MAT outPredicted(input);
	getFirst()->forProp(outPredicted, true, true);
	
	// (2) calculate error matrix and error
	if (!deltaProvided)
		diffMatrix = move(l2_errorMatrix(outPredicted - outDesired)); // delta =  estimate - target
	else
		diffMatrix = outDesired;
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
/* Train the Discriminator
*/
void CNet::train_GAN_D(MAT& in_copy, MAT &in, MAT& res, bool real, const learnPars& pars) {

	/*	The ICML 2017 workshop uses softplus as the cost .. see below.
	*	d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
	*	g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
	*/
	static bool softPlusLoss = false;

	static MAT labels(getNOUT(), 1);
	static MAT cross_entropy_gradient(getNOUT(), 1);

	getFirst()->forProp(in_copy, true, true);
	// check for NaN's & Infinities.
	if (!in_copy.allFinite())
		return; // protect the network.
	if (softPlusLoss) {
		if (real) {
			cross_entropy_gradient = (-in_copy).unaryExpr(&DSoftPlus);
			res(0, 0) = (-in_copy).unaryExpr(&SoftPlus).mean();

		} else {
			cross_entropy_gradient = (in_copy).unaryExpr(&DSoftPlus);
			res(0, 0) = (in_copy).unaryExpr(&SoftPlus).mean();
		}
	} else {

		if (real) {
			labels.setOnes();
		} else {
			labels.setZero();
		}

		cross_entropy_gradient = sigmoid_cross_entropy_errorMatrix(in_copy, labels); // delta =  estimate - target
		// calculate D-loss
		res(0, 0) = sigmoid_cross_entropy_with_logits(in_copy, labels);
	}

	// apply whatever gradient we just calculated
	if (cross_entropy_gradient.allFinite()) {
		getLast()->backPropDelta(cross_entropy_gradient, true);
		getFirst()->applyUpdate(pars, in, true);
	}
}
/* Prepare the training of the generator
*/
void CNet::train_GAN_G_D(MAT& in_copy,  MAT& res, const learnPars& pars) {
	/*	The ICML 2017 workshop uses softplus as the cost .. see below.
	*	d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
	*	g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
	*/
	static MAT labels(getNOUT(), 1);
	static MAT cross_entropy_gradient(getNOUT(), 1);
	static bool softPlusLoss = false;

	getFirst()->forProp(in_copy, true, true);

	// compute the gradient through Critic
	if (softPlusLoss) {
		// Calculate the generator loss
		res(0, 0) = (-in_copy).unaryExpr(&SoftPlus).mean();
		cross_entropy_gradient = (-in_copy).unaryExpr(&DSoftPlus);// Generator loss
	} else {
		labels.setOnes();
		// Calculate the generator loss
		res(0, 0) = (in_copy - in_copy.unaryExpr(&LogExp)).mean();
		cross_entropy_gradient = move(sigmoid_GEN_loss(labels, in_copy));// Generator loss
	}
	if (cross_entropy_gradient.allFinite())
		getLast()->backPropDelta(cross_entropy_gradient, true);// make sure the deltaSaves are updated
}

// Backpropagation 
void CNet::backProp_GAN_G(MAT& input, MAT& deltaMatrix, learnPars& pars) {

	// (0) Check if Input contains no NaN's or Infinities.
	if (!deltaMatrix.allFinite())
		return; // protect the network.
	// (2) Backpropagate deltas
	getLast()->backPropDelta(deltaMatrix, true);
	// (3) Apply update
	getFirst()->applyUpdate(pars, input, true);
}


void CNet::inquireDimensions(size_t layer, size_t& rows, size_t& cols) const {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			MATIND layerDimension = dynamic_cast<PhysicalLayer*>(layers[layer])->WDimensions();
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

void CNet::copyNthDelta(size_t layer, fREAL* const toCopyTo, int32_t size) const {
	if (layer < getLayerNumber()) {
		MAT temp = layers[layer]->getDelta();
		//temp.resize(temp.size(), 1); // unneccessary since delta's are vectors anyway
		assert(temp.size() == size);
		copyToOut(temp.data(), toCopyTo, size);
	}
}

void CNet::copyNthLayer(size_t layer, fREAL* const toCopyTo) const {
	if (layer < getLayerNumber()) {
		if (isPhysical(layer)) {
			MAT temp = dynamic_cast<PhysicalLayer*>(layers[layer])->copyW();
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
			dynamic_cast<PhysicalLayer*>(layers[layer])->setW(newLayer);
		}
	}
}
MAT CNet::l2_errorMatrix(const MAT& diff) {
	return diff; // force Visual C++ to return without temporary - since RVO doesn't work ???!
}
MAT CNet::l1_errorMatrix(const MAT& diff) {
	return (diff).unaryExpr(&sgn<fREAL>);
}
fREAL CNet::sigmoid_cross_entropy_with_logits(const MAT& logits, const MAT& labels) { 
	// obvsly logits.shape == labels.shape
	return (logits.unaryExpr(&ReLu) - logits.cwiseProduct(labels) + (logits).unaryExpr(&LogAbsExp)).mean();
}

MAT CNet::sigmoid_GEN_loss(const MAT& labels, const MAT& logits) {
	return logits.unaryExpr(&Sig) -labels; // only use if top level not sigma 
}
MAT CNet::sigmoid_cross_entropy_errorMatrix(const MAT& logits, const MAT& labels) {
	
	/*static MAT ones = MAT(getNOUT(), 1);
	ones.setOnes();
	return ones - labels - ((-1)*logits).unaryExpr(&Sig);*/
	return logits.unaryExpr(&DReLu) - labels + logits.unaryExpr(&DLogAbsExp);
}

MAT CNet::entropyRegularizer(const MAT& out) {
	fREAL sum = out.sum();
	static const fREAL eps = 1E-8;
	return  (out).unaryExpr(&logP1_fREAL);
}
fREAL CNet::l2_error(const MAT& diff) {
	fREAL sum = cumSum(matNorm(diff));
	//if (sum > 0.0f)
		return 0.5f*sqrt(sum); //  / sqrt(sum)
	//else
	//	return 0.0f;
}

fREAL CNet::l1_error(const MAT& diff) {
	return diff.cwiseAbs().sum();
}


