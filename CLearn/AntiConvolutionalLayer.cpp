#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, 
	uint32_t _features, uint32_t _outBoxes,  actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), outBoxes(_outBoxes), 
	PhysicalLayer(_outBoxes*_NOUTX*_NOUTY , _outBoxes*_features*_NINX*_NINY ,  type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX },
		MATIND{ 1,_features}) {
	
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY,  uint32_t _strideX, 
	uint32_t _features, uint32_t _outBoxes, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), outBoxes(_outBoxes), 
	 PhysicalLayer(_outBoxes*_NOUTX*_NOUTY,  type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX },
		MATIND{ 1,_features }, lower) {
	
	init();
	assertGeometry();
}

// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _outBoxes,  actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),outBoxes(_outBoxes), 
	PhysicalLayer(_outBoxes*_NOUTXY*_NOUTXY , _outBoxes*_features*_NINXY*_NINXY ,  type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY },
		MATIND{ 1,_features}) {
	
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _outBoxes,  actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT() / (_outBoxes*_features))), NINY(sqrt(lower.getNOUT() / (_outBoxes*_features))), kernelX(_kernelXY), kernelY(_kernelXY),
	strideY(_stride), strideX(_stride), features(_features), outBoxes(_outBoxes), 
	PhysicalLayer(_outBoxes*_NOUTXY*_NOUTXY,  type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1, _features}, lower) {
	
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
	assert(outBoxes*NOUTX*NOUTY == getNOUT());
	assert(outBoxes*features*NINX*NINY == getNIN());
	assert(antiConvoSize(NINY,kernelY, antiConvPad(NINY,strideY,kernelY,NOUTY),strideY) == NOUTY);
	assert(antiConvoSize(NINX, kernelX, antiConvPad(NINX, strideX, kernelX, NOUTX),strideX) == NOUTX);
}
void AntiConvolutionalLayer::init() {
	//layer += MAT::Constant(layer.rows(), layer.cols(), 1.0f); // make everything positive
	layer *= ((fREAL)features) / layer.size();
}

// weight normalization reparametrize
void AntiConvolutionalLayer::updateW() {
	for (uint32_t i = 0; i < features; i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		layer._FEAT(i) = G(0, i)*vInversEntry *V._FEAT(i);
	}
}

void AntiConvolutionalLayer::initV() {
	V = layer;
	//normalizeV();
}
void AntiConvolutionalLayer::normalizeV() {
	for (uint32_t i = 0; i< features; ++i)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void AntiConvolutionalLayer::initG() {
	for (uint32_t i = 0; i< features; ++i)
		G(0, i) = normSum(layer._FEAT(i));
}
void AntiConvolutionalLayer::inversVNorm() {
	
	VInversNorm.setOnes();
	for (uint32_t i = 0; i<features; ++i)
		VInversNorm._FEAT(i) /= normSum(V._FEAT(i));
}
MAT AntiConvolutionalLayer::gGrad(const MAT& grad) {
	MAT ret(1, features);

	for (uint32_t i = 0; i <features; ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*vInversEntry; //(1,1)
	}
	return ret;
}
MAT AntiConvolutionalLayer::vGrad(const MAT& grad, MAT& ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	for (uint32_t i = 0; i <features; ++i) {
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
	return outBoxes;
}

void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {

	// (1) Reshape the remaining input

	inBelow.resize(getNINY(), outBoxes*features*getNINX());
	inBelow =  antiConv_(inBelow, layer, getNOUTY(), getNOUTX(), strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), 
		antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes); // square convolution//
	inBelow.resize(getNOUT(), 1);

	if (training) {
		actSave = inBelow;
		if (getHierachy() != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			if (recursive) {
				above->forProp(inBelow, true, true);
			}
		} 
	} else {
		if (getHierachy() != hierarchy_t::output) {
			inBelow = inBelow.unaryExpr(act);
			
			if(recursive)
				above->forProp(inBelow, false, true);
		} 
	}
}
// backprop
void AntiConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	
	deltaSave = deltaAbove;

	if (getHierachy() != hierarchy_t::input) { // ... this is not an input layer.
		
		deltaSave.resize(getNOUTY(), outBoxes*getNOUTX()); // keep track of this!!!

		deltaAbove = conv_(deltaSave, layer,getNINY(), getNINX(), 
			strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes);
		deltaAbove.resize(getNIN(), 1);

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

		input.resize(getNINY(), getNINX()*features*outBoxes);
		MAT convoluted = antiConvGrad_(deltaSave, input, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), 
			antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), features, outBoxes); 

		deltaSave.resize(getNOUT(), 1); // make sure to resize to NOUT-sideChannels
		// Leave the dimensionality of input intact
		input.resize(getNIN(), 1);

		return convoluted;
	} else {
		MAT fromBelow = below->getACT();

		fromBelow.resize(getNINY(), getNINX()*features*outBoxes);
		MAT convoluted = antiConvGrad_(deltaSave, fromBelow, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), antiConvPad(getNINX(), strideX, kernelX, getNOUTX()),
			features, outBoxes); 

		deltaSave.resize(getNOUT(), 1);  // make sure to resize to NOUT-sideChannels
		return convoluted;
	}
}

void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << " " << NOUTX << " " << NINY << " " << NINX << " " << kernelY << " " << kernelX << " " << strideY << " " << strideX << " " << features << " " << outBoxes<< endl;
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
	//in >> sideChannels;

	layer = MAT(kernelY*features*kernelX, 1); // initialize as a column vector
	V= MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	V.setZero();
	G.setZero();
	stepper.reset();

	for (size_t i = 0; i < layer.size(); ++i) {
		in >> layer(i,0);
	}
	layer.resize(kernelY, features*kernelX);
}