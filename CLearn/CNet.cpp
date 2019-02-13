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

void CNet::addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, actfunc_t type) {
	// At the moment, I only allow for square-shaped input.
	// this may need to change in the future.
	if (getLayerNumber() > 0) {
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, kernelXY, stride, features, type, *(getLast()));
		layers.push_back(cl);
	} else {
		// then it's the input layer
		ConvolutionalLayer* cl = new ConvolutionalLayer(NOUTXY, sqrt(NIN), kernelXY, stride, features,  type);
		layers.push_back(cl);
	}
}

void CNet::addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t outBoxes, actfunc_t type) {
	if (getLayerNumber() > 0) {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, kernelXY, stride, features, outBoxes, type, *(getLast()));
		layers.push_back(acl);
	} else {
		AntiConvolutionalLayer* acl = new AntiConvolutionalLayer(NOUTXY, sqrt(NIN/features), kernelXY, stride, features, outBoxes, type);
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

// Simply output the network
fREAL CNet::forProp(MAT& in, const MAT& outDesired, const learnPars& pars) {

	// (1) Forward propagation
	getFirst()->forProp(in, false, true);
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
		diffMatrix = move(outDesired);
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


// Backpropagation 
fREAL CNet::backProp_GAN_D(MAT& input, MAT& outPredicted, bool real, const learnPars& pars) {
	
	// (0) Check if Input contains no NaN's or Infinities.
	// ... 
	// ...
	// (0.5) Initialize error and difference matrix
	static MAT cross_entropy_gradient = MAT(getNOUT(), 1);
	bool errFlag = false;
	static bool real_flag = 0;
	static bool fake_flag = 0;
	static MAT cross_entropy_gradient_real(getNOUT(),1);
	static MAT cross_entropy_gradient_fake(getNOUT(), 1);
	
	bool batchIsDue = pars.batch_update == 0;
	learnPars newPars(pars); // we need to do this to hack the accept parameter
	newPars.accept = true;
	fREAL errorOut = 0.0f;

	// (1) Propagate in forward direction (with saveActivations == true)
	MAT logits(input);
	static MAT labels(getNOUT(), 1);

	getFirst()->forProp(logits, true, true);

	if (real) {
		labels.setOnes();
		cross_entropy_gradient_real = move(sigmoid_cross_entropy_errorMatrix(logits, labels)); // delta =  estimate - target
		// backprop the real gradient
		if (cross_entropy_gradient_real.allFinite())
			getLast()->backPropDelta(cross_entropy_gradient_real, true);
		else
			errFlag = true;
		// NOW - WE have to HACK our "accept" and the batch_update parameters
		// to make sure the gradients are not applied
		// but instead we wait for the fake gradients to come in
		if (batchIsDue) {
			newPars.batch_update = 1;
		}
		getFirst()->applyUpdate(newPars, input, true);

		real_flag = true;
	} else {
		labels.setZero();
		cross_entropy_gradient_fake = move(sigmoid_cross_entropy_errorMatrix(logits, labels)); // delta =  estimate - target
		if (cross_entropy_gradient_fake.allFinite())
			getLast()->backPropDelta(cross_entropy_gradient_fake, true);
		else
			errFlag = true;
		// NOW - HACK AGAIN
		if (batchIsDue ) {
			newPars.batch_update = 1;
		}
		getFirst()->applyUpdate(newPars, input, true);

		fake_flag = true;
	}

	// (2) calculate error matrix and error
	errorOut = sigmoid_cross_entropy_with_logits(logits, labels);

	// (3) Descent the combined gradient
	if (real_flag
		&& fake_flag) {
		// (3.1) set flags to false
		real_flag = false;
		fake_flag = false;
		//cross_entropy_gradient = cross_entropy_gradient_real + cross_entropy_gradient_fake;

		// (3.9) FINALLY Apply update
		if (batchIsDue) { // Go ahead and apply combined gradients
			newPars.batch_update = 0;
			newPars.accept = false; // don't add the gradient another time
			getFirst()->applyUpdate(newPars, input, true);
		}

		// (4) Now, we need to do one last thing - we need to 
		// somehow the delta for the generator (labels = 0, but fake input.
		// for this to work, the discriminator needs to be trained
		// with the real data first and then with the fake data.
		if (!real) { // we have the right input - the generator output
			//if (pars.spectral_normalization) {
			//	switchW_W_temp();
			//}
			errorOut = (logits - logits.unaryExpr(&LogExp)).mean();
			labels.setOnes();
			cross_entropy_gradient = move(sigmoid_GEN_loss(labels,logits));// Generator loss
			//cross_entropy_gradient = move(sigmoid_cross_entropy_errorMatrix(logits, labels));
			//MAT logits(input);
			//getFirst()->forProp(logits, true, true);
			if (cross_entropy_gradient.allFinite())
				getLast()->backPropDelta(cross_entropy_gradient, true);// make sure the deltaSaves are updated
			else
				errFlag = true;
			//if (pars.spectral_normalization) {
			//	switchW_W_temp();
			//}
		}

	}
	
	outPredicted = logits; // usually just a number
	// DONE
	if (errFlag)
		errorOut = -42;
	return errorOut;
}

// Backpropagation 
fREAL CNet::backProp_GAN_G(MAT& input, MAT& deltaMatrix, const learnPars& pars) {

	// (0) Check if Input contains no NaN's or Infinities.
	// ... 
	// ...

	fREAL errorOut = 0.0f;

	// (1) Propagate in forward direction (with saveActivations == true)
	MAT logits(input);

	getFirst()->forProp(logits, true, true);

	// (2) calculate error matrix and error
	// Check if we've computed error gradients 
	// for both fake and real datasets
	//deltaMatrix += pars.lambda*l1_errorMatrix(deltaMatrix); // allow for regularization... 
	// (3) back propagate the deltas
	getLast()->backPropDelta(deltaMatrix, true);
	// (4) Apply update
	getFirst()->applyUpdate(pars, input, true);

	deltaMatrix = logits; // overwrite the deltaMatrix with generator output
	// DONE
	return 0.0f;
}
/* Wasserstein GAN critic training
*/
fREAL CNet::backProp_WGAN_D(MAT& input, MAT& outPredicted, bool real, const learnPars& pars) {

	// (0) Check if Input contains no NaN's or Infinities.
	// ... 
	// ...
	// (0.5) Initialize error and difference matrix
	static MAT labels(1, 1);
	//static MAT alpha = MAT(1, 1);

	static bool real_flag = 0;
	static bool fake_flag = 0;
	//static MAT input_real = MAT(getNIN(), 1);
	//static MAT input_fake = MAT(getNIN(), 1);


	bool batchIsDue = pars.batch_update == 0;
	learnPars newPars = pars; // we need to do this to hack the accept parameter
	newPars.accept = true;
	static fREAL loss_critic = 0.0f;
	fREAL result = 0.0f;

	// (1) Propagate in forward direction (with saveActivations == true)
	MAT critic_out(input);

	getFirst()->forProp(critic_out, true, true);

	if (real) {
		//input_real = move(input); // x 
		result = critic_out.mean(); //- D_w(x)
		real_flag = true;

		// we need to abuse the batch buffer and store some gradients
		labels.setOnes();
		labels = (-1)*labels;

		getLast()->backPropDelta(labels, true);
		if (batchIsDue) {
			newPars.batch_update = 1;
		}
		getFirst()->applyUpdate(newPars, input, true);


	} else {
		result = -critic_out.mean();
		//input_fake = move(input);
		fake_flag = true;
		// we need to abuse the batch buffer and store some gradients
		labels.setOnes();

		getLast()->backPropDelta(labels, true);
		if (batchIsDue) {
			newPars.batch_update = 1;
		}
		getFirst()->applyUpdate(newPars, input, true);

	}

	// (3) Descent the combined gradient
	if (real_flag
		&& fake_flag) {
		// now we have real and fake inputs.
		// Draw random number for interpolating between  real and fake data
		//alpha.Random().unaryExpr(&ReLu);
		//MAT interp = alpha(0, 0)*input_real + (1.0f - alpha(0, 0))*input_fake;
		// Now, we need to built the gradient wRt this interpolated input
		//getFirst()->forProp(interp, true, true);
		//labels.setOnes();
		//getLast()->backPropDelta(labels, true);
		loss_critic = 0.0f;
		real_flag = false;
		fake_flag = false;
		//cross_entropy_gradient = cross_entropy_gradient_real + cross_entropy_gradient_fake;

		// (3.9) FINALLY Apply update
		if (batchIsDue) { // Go ahead and apply combined gradients
			newPars.batch_update = 0;
			newPars.accept = false; // don't add the gradient another time
			getFirst()->applyUpdate(newPars, input, true);
		}

		// (4) Now, we need to do one last thing - we need to 
		// somehow the delta for the generator (labels = 0, but fake input.
		// for this to work, the discriminator needs to be trained
		// with the real data first and then with the fake data.
		if (!real) { // we have the right input - the generator output
			labels.setOnes();
			labels = (-1)*labels;
			getLast()->backPropDelta(labels, true);// make sure the deltaSaves are updated
		}
	}

	return result;
}

// Backpropagation 
fREAL CNet::backProp_WGAN_G(MAT& input, MAT& deltaMatrix, const learnPars& pars) {

	fREAL errorOut = 0.0f;

	// (1) Propagate in forward direction (with saveActivations == true)
	MAT logits(input); 

	getFirst()->forProp(logits, true, true);
	// (2) calculate error matrix and error
	// Check if we've computed error gradients 
	// for both fake and real datasets

	// (3) back propagate the deltas
	getLast()->backPropDelta(deltaMatrix, true);
	// (4) Apply update
	getFirst()->applyUpdate(pars, input, true);

	deltaMatrix = logits; // overwrite the deltaMatrix with generator output
						  // DONE
	return 0.0f;
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
	return logits.unaryExpr(&Sig)-labels;
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


