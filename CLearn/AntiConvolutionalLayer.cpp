#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, 
	uint32_t _features, uint32_t _outBoxes, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), outBoxes(_outBoxes), sideChannels(_sideChannels),
	PhysicalLayer(_outBoxes*_NOUTX*_NOUTY + _sideChannels, _outBoxes*_features*_NINX*_NINY + _sideChannels, 0.0f, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX },
		MATIND{ 1,_features}) {
	
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY,  uint32_t _strideX, 
	uint32_t _features, uint32_t _outBoxes, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), outBoxes(_outBoxes), 
	sideChannels(_sideChannels), PhysicalLayer(_outBoxes*_NOUTX*_NOUTY+_sideChannels, 0.0f, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX },
		MATIND{ 1,_features }, lower) {
	
	init();
	assertGeometry();
}

// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _outBoxes, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),outBoxes(_outBoxes), sideChannels(_sideChannels),
	PhysicalLayer(_outBoxes*_NOUTXY*_NOUTXY + _sideChannels, _outBoxes*_features*_NINXY*_NINXY + _sideChannels, 0.0f, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY },
		MATIND{ 1,_features}) {
	
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _outBoxes, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT() / (_outBoxes*_features))), NINY(sqrt(lower.getNOUT() / (_outBoxes*_features))), kernelX(_kernelXY), kernelY(_kernelXY),
	strideY(_stride), strideX(_stride), features(_features), outBoxes(_outBoxes), sideChannels(_sideChannels),
	PhysicalLayer(_outBoxes*_NOUTXY*_NOUTXY + _sideChannels, 0.0f, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1, _features}, lower) {
	
	init();
	assertGeometry();
}

// destructor
AntiConvolutionalLayer::~AntiConvolutionalLayer() {
	layer.resize(0, 0);
}

layer_t AntiConvolutionalLayer::whoAmI() const {
	return layer_t::antiConvolutional;
}

void AntiConvolutionalLayer::assertGeometry() {
	assert(outBoxes*NOUTX*NOUTY+sideChannels == getNOUT());
	assert(outBoxes*features*NINX*NINY+sideChannels == getNIN());
	assert(antiConvoSize(NINY,kernelY, antiConvPad(NINY,strideY,kernelY,NOUTY),strideY) == NOUTY);
	assert(antiConvoSize(NINX, kernelX, antiConvPad(NINX, strideX, kernelX, NOUTX),strideX) == NOUTX);
}
void AntiConvolutionalLayer::init() {
	sideChannelBuffer = MAT(sideChannels, 1);
	sideChannelBuffer.setZero();
	//layer += MAT::Constant(layer.rows(), layer.cols(), 1.0f); // make everything positive
	layer *= ((fREAL)features) / layer.size();
}

// weight normalization reparametrize
void AntiConvolutionalLayer::updateW() {
	for (uint32_t i = 0; i < getFeatures(); i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		layer._FEAT(i) = G(0, i)*vInversEntry *V._FEAT(i);
	}
}

void AntiConvolutionalLayer::initV() {
	V = layer;
	//normalizeV();
}
void AntiConvolutionalLayer::normalizeV() {
	for (uint32_t i = 0; i< getFeatures(); ++i)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void AntiConvolutionalLayer::initG() {
	for (uint32_t i = 0; i< getFeatures(); ++i)
		G(0, i) = normSum(layer._FEAT(i));
}
void AntiConvolutionalLayer::inversVNorm() {
	
	VInversNorm.setOnes();
	for (uint32_t i = 0; i< getFeatures(); ++i)
		VInversNorm._FEAT(i) /= normSum(V._FEAT(i));
}
MAT AntiConvolutionalLayer::gGrad(const MAT& grad) {
	MAT ret(1, getFeatures());

	for (uint32_t i = 0; i < getFeatures(); ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*vInversEntry; //(1,1)
	}
	return ret;
}
MAT AntiConvolutionalLayer::vGrad(const MAT& grad, MAT& ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	for (uint32_t i = 0; i < getFeatures(); ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		out._FEAT(i) *= G(0, i)*vInversEntry;
		// (2) subtract 
		vInversEntry *= vInversEntry;
		out._FEAT(i) -= G(0, i)*ggrad(0, i)*V._FEAT(i)*vInversEntry;
	}
	return out;
}
uint32_t AntiConvolutionalLayer::getFeatures() const {
	// deconv layers restart the tree.
	// I assume there is at least one FC layer between 
	// a convolutional and an anticonvolutional layer
	return  features;
}

void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {


	// (1) Take care of the sidechannel inputs
	if (sideChannels > 0) {
		sideChannelBuffer = inBelow.bottomRows(sideChannels);
		inBelow.conservativeResize(getNIN()-sideChannels, 1); // drop bottom rows
	}

	// (2) Reshape the remaining input

	inBelow.resize(getNINY(), outBoxes*features*getNINX());
	inBelow =  antiConv_(inBelow, layer, getNOUTY(), getNOUTX(), strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), 
		antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes); // square convolution//
	inBelow.resize(getNOUT()-sideChannels, 1);

	if (training) {
		actSave = inBelow;
		if (getHierachy() != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			if (recursive) {
				if (sideChannels > 0) {
					inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
					inBelow.bottomRows(sideChannels) = sideChannelBuffer;
				}
				above->forProp(inBelow, true, true);
			}
		} else {
			if (sideChannels > 0) {
				inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
				inBelow.bottomRows(sideChannels) = sideChannelBuffer;
			}
		}
	} else {
		if (getHierachy() != hierarchy_t::output) {
			inBelow = inBelow.unaryExpr(act);
			if (sideChannels > 0) {
				inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
				inBelow.bottomRows(sideChannels) = sideChannelBuffer;
			}
			if(recursive)
				above->forProp(inBelow, false, true);
		} else {
			if (sideChannels > 0) {
				inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
				inBelow.bottomRows(sideChannels) = sideChannelBuffer;
			}
		}
	}
}
// backprop
void AntiConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	if (sideChannels > 0) {
		sideChannelBuffer = deltaAbove.bottomRows(sideChannels);
		deltaAbove.conservativeResize(getNOUT() - sideChannels, 1);
	}
	deltaSave = deltaAbove;

	if (getHierachy() != hierarchy_t::input) { // ... this is not an input layer.
		
		deltaSave.resize(getNOUTY(), outBoxes*getNOUTX()); // keep track of this!!!

		deltaAbove = conv_(deltaSave, layer,getNINY(), getNINX(), 
			strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes);
		deltaAbove.resize(getNIN()-sideChannels, 1);
		
		if (sideChannels > 0) {
			deltaAbove.conservativeResize(getNIN(), 1);
			deltaAbove.bottomRows(sideChannels) = sideChannelBuffer;
		}

		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj)
		deltaSave.resize(getNOUT(), 1); // resize back
		if (recursive) {
			below->backPropDelta(deltaAbove, true); // cascade...
		}
	}
}

// grad
MAT AntiConvolutionalLayer::grad(MAT& input) {
	deltaSave.resize(NOUTY, outBoxes*NOUTX);

	if (getHierachy() == hierarchy_t::input) {

		if (sideChannels > 0) {
			sideChannelBuffer = input.bottomRows(sideChannels);
			input.conservativeResize(getNIN()-sideChannels, 1); // drop sidechannel inputs - not relevant for gradient
		}

		input.resize(getNINY(), getNINX()*features*outBoxes);
		MAT convoluted = antiConvGrad_(deltaSave, input, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), 
			antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes); 

		deltaSave.resize(getNOUT() - sideChannels, 1); // make sure to resize to NOUT-sideChannels

		input.resize(getNIN() - sideChannels, 1);
		// Leave the dimensionality of input intact
		if (sideChannels > 0) {
			input.conservativeResize(getNIN(), 1);
			input.bottomRows(sideChannels) = sideChannelBuffer;
		}
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();

		if (sideChannels > 0) {
			sideChannelBuffer = fromBelow.bottomRows(sideChannels);
			fromBelow.conservativeResize(getNIN() - sideChannels, 1);
		}
		fromBelow.resize(getNINY(), getNINX()*features*outBoxes);
		MAT convoluted = antiConvGrad_(deltaSave, fromBelow, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), antiConvPad(getNINX(), strideX, kernelX, getNOUTX()),
			features, outBoxes); 

		deltaSave.resize(getNOUT() - sideChannels, 1);  // make sure to resize to NOUT-sideChannels
		return convoluted;
	}
}

void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << " " << NOUTX << " " << NINY << " " << NINX << " " << kernelY << " " << kernelX << " " << strideY << " " << strideX << " " << features << " " << outBoxes<< " "<< sideChannels<<endl;
	MAT temp = layer;
	temp.resize(layer.size(), 1);
	os << temp << endl;
}
// first line has been read already
void AntiConvolutionalLayer::loadFromFile(ifstream& in) {
	in >> NOUTY;
	in >> NOUTX;
	in >> NINY;
	in >> NINX;
	in >> kernelY;
	in >> kernelX;
	in >> strideY;
	in >> strideX;
	in >> features;
	in >> outBoxes;
	in >> sideChannels;

	layer = MAT(kernelY*features*kernelX, 1); // initialize as a column vector
	V= MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	V.setZero();
	G.setZero();
	stepper.reset();

	for (size_t i = 0; i < layer.size(); ++i) {
		in >> layer(i);
	}
	layer.resize(kernelY, features*kernelX);
}