#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX,
	 uint32_t _outChannels, uint32_t _inChannels, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_inChannels*_outChannels), inChannels(_inChannels),
	outChannels(_outChannels), PhysicalLayer(_outChannels*_NOUTX*_NOUTY, _inChannels*_NINX*_NINY, type, MATIND{ _kernelY, features*_kernelX }, MATIND{ _kernelY, features*_kernelX },
		MATIND{ 1,features }, MATIND{_kernelY, features*_kernelX}) {

	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX,
	 uint32_t _outChannels, uint32_t _inChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_inChannels*_outChannels), inChannels(_inChannels), 
	outChannels(_outChannels), PhysicalLayer(_outChannels*_NOUTX*_NOUTY, type, MATIND{ _kernelY, features*_kernelX }, MATIND{ _kernelY, features*_kernelX },
		MATIND{ 1, features }, MATIND{_kernelY, features*_kernelX }, lower) {

	init();
	assertGeometry();
}

// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride,  uint32_t _outChannels, uint32_t _inChannels, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_inChannels*_outChannels), outChannels(_outChannels),
	PhysicalLayer(_outChannels*_NOUTXY*_NOUTXY, _inChannels*_NINXY*_NINXY, type, MATIND{ _kernelXY, _inChannels*_outChannels*_kernelXY }, MATIND{ _kernelXY, _inChannels*_outChannels*_kernelXY },
		MATIND{ 1, _inChannels*_outChannels }, MATIND{ _kernelXY, _inChannels*_outChannels*_kernelXY }) {

	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _outChannels, uint32_t _inChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT() / (_inChannels))), NINY(sqrt(lower.getNOUT() / (_inChannels))), kernelX(_kernelXY), kernelY(_kernelXY),
	strideY(_stride), strideX(_stride), features(_inChannels*_outChannels), inChannels(_inChannels), outChannels(_outChannels),
	PhysicalLayer(_outChannels*_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _inChannels*_outChannels*_kernelXY }, MATIND{ _kernelXY, _inChannels*_outChannels*_kernelXY },
		MATIND{ 1,_inChannels*_outChannels }, MATIND{_kernelXY, _inChannels*_outChannels*_kernelXY }, lower) {

	init();
	assertGeometry();
}

// destructor
AntiConvolutionalLayer::~AntiConvolutionalLayer() {}

layer_t AntiConvolutionalLayer::whoAmI() const {
	return layer_t::antiConvolutional;
}

void AntiConvolutionalLayer::assertGeometry() {
	assert(outChannels*NOUTX*NOUTY == getNOUT());
	assert(inChannels*NINX*NINY == getNIN());
	assert(antiConvoSize(NINY, kernelY, antiConvPad(NINY, strideY, kernelY, NOUTY), strideY) == NOUTY);
	assert(antiConvoSize(NINX, kernelX, antiConvPad(NINX, strideX, kernelX, NOUTX), strideX) == NOUTX);
}
void AntiConvolutionalLayer::init() {
	W.setRandom();
	W.unaryExpr(&SoftPlus); // make evrythng positive
	W *= ((fREAL)features) / W.size();
}

/* Weight Normalization Functions
*/
void AntiConvolutionalLayer::wnorm_setW() {
	for (uint32_t i = 0; i < features; i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		W._FEAT(i) = G(0, i)*vInversEntry *V._FEAT(i);
	}
}

void AntiConvolutionalLayer::wnorm_initV() {
	V = W;
	//normalizeV();
}
void AntiConvolutionalLayer::wnorm_normalizeV() {
	for (uint32_t i = 0; i< features; ++i)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void AntiConvolutionalLayer::wnorm_initG() {
	for (uint32_t i = 0; i< features; ++i)
		G(0, i) = normSum(W._FEAT(i));
}
void AntiConvolutionalLayer::wnorm_inversVNorm() {

	VInversNorm.setOnes();
	for (uint32_t i = 0; i<features; ++i)
		VInversNorm._FEAT(i) /= normSum(V._FEAT(i));
}
MAT AntiConvolutionalLayer::wnorm_gGrad(const MAT& grad) {
	MAT ret(1, features);

	for (uint32_t i = 0; i <features; ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*vInversEntry; //(1,1)
	}
	return ret;
}
MAT AntiConvolutionalLayer::wnorm_vGrad(const MAT& grad, MAT& ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	for (uint32_t i = 0; i < features; ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		out._FEAT(i) *= G(0, i)*vInversEntry;
		// (2) subtract 
		vInversEntry *= vInversEntry;
		out._FEAT(i) -= G(0, i)*ggrad(0, i)*V._FEAT(i)*vInversEntry;
	}
	return out;
}
/* Spectral Normalization Functions
*/
void AntiConvolutionalLayer::snorm_setW() {
	W = W_temp / spectralNorm(W_temp, u1, v1);

	/*for(size_t i=0; i< features; ++i)
		W._FEAT(i) = W_temp._FEAT(i)/spectralNorm(W._FEAT(i),u1.block(i*kernelY,0, kernelY,1), v1.block(i*kernelX, 0, kernelX, 1));*/
}
void AntiConvolutionalLayer::snorm_updateUVs() {
	updateSingularVectors(W_temp, u1, v1, 1);
	/*
	MAT u_temp = u1.block(0, 0, kernelY, 1);
	MAT v_temp = v1.block(0, 0, kernelX, 1);

	for (size_t i = 0; i < features; ++i) {
		// TODO Do this differently - without the copies here
		u_temp = u1.block(i*kernelY, 0, kernelY, 1);
		v_temp = v1.block(i*kernelX, 0, kernelX, 1);
		updateSingularVectors(W_temp._FEAT(i), u_temp, v_temp, 1);
		u1.block(i*kernelY, 0, kernelY, 1) = u_temp;
		v1.block(i*kernelX, 0, kernelX, 1) = v_temp;
	}*/
}
MAT AntiConvolutionalLayer::snorm_dWt(MAT& grad) {
	if (lambdaCount>0)
		grad -= lambdaBatch / lambdaCount*(u1*v1.transpose());

	grad /= spectralNorm(W_temp, u1, v1);
	lambdaBatch = 0; // reset this
	lambdaCount = 0;
	return grad;
	/*
	if (lambdaCount > 0) {
		fREAL ratio = lambdaBatch / lambdaCount;
		for (size_t i = 0; i < features; ++i) {
			grad._FEAT(i) -= ratio*(u1.block(i*kernelY, 0, kernelY, 1)*(v1.block(i*kernelX, 0, kernelX, 1)).transpose());

			grad._FEAT(i) /= spectralNorm(W_temp._FEAT(i), u1.block(i*kernelY, 0, kernelY, 1), v1.block(i*kernelX, 0, kernelX, 1));

		}
		lambdaBatch = 0; // reset this
		lambdaCount = 0;
	}
	return grad; */
}
uint32_t AntiConvolutionalLayer::getOutChannels() const {
	// deconv layers restart the tree.
	// I assume there is at least one FC layer between 
	// a convolutional and an anticonvolutional layer
	return outChannels;
}

void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {

	// (1) Reshape the remaining input
	inBelow.resize(getNINY(), inChannels*getNINX());
	inBelow = antiConv_(inBelow, W, getNOUTY(), getNOUTX(), strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()),
		antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), outChannels, inChannels); // square convolution//
	inBelow.resize(getNOUT(), 1);
	inBelow += b; // add bias term

	if (training) {
		actSave = inBelow;
		inBelow = move(this->getACT());
		if (recursive && getHierachy() != hierarchy_t::output) {
			above->forProp(inBelow, true, true);
		}
	} else {
		inBelow = inBelow.unaryExpr(act);

		if (recursive && getHierachy() != hierarchy_t::output)
			above->forProp(inBelow, false, true);
	}
}
// backprop
void AntiConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	if (getHierachy() == hierarchy_t::output)
		deltaAbove = deltaAbove.cwiseProduct(this->getDACT());

	deltaSave = deltaAbove;

	if (getHierachy() != hierarchy_t::input) { // ... this is not an input layer.
		

		deltaAbove.resize(getNOUTY(), outChannels*getNOUTX()); // keep track of this!!!

		deltaAbove = conv_(deltaAbove, W, getNINY(), getNINX(),
			strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), inChannels, outChannels);
		deltaAbove.resize(getNIN(), 1);

		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj)
		//deltaSave.resize(getNOUT(), 1); // resize back
		if (recursive) {
			below->backPropDelta(deltaAbove, true); // cascade...
		}
	}
}

// grad
MAT AntiConvolutionalLayer::w_grad(MAT& input) {
	deltaSave.resize(NOUTY, outChannels*NOUTX);

	if (getHierachy() == hierarchy_t::input) {

		input.resize(getNINY(), getNINX()*inChannels);
		MAT convoluted = antiConvGrad_(deltaSave, input, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()),
			antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), outChannels, inChannels);

		deltaSave.resize(getNOUT(), 1); // make sure to resize to NOUT-sideChannels
										// Leave the dimensionality of input intact
		input.resize(getNIN(), 1);

		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(getNINY(), getNINX()*inChannels);

		MAT convoluted = antiConvGrad_(deltaSave, fromBelow, kernelY, kernelX, strideY, strideX, antiConvPad(getNINY(), strideY, kernelY, getNOUTY()), 
			antiConvPad(getNINX(), strideX, kernelX, getNOUTX()), outChannels, inChannels);

		deltaSave.resize(getNOUT(), 1);  // make sure to resize to NOUT-sideChannels
		return convoluted;
	}
}
// b_grad
MAT AntiConvolutionalLayer::b_grad() {
	return deltaSave;
}
void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << " " << NOUTX << " " << NINY << " " << NINX << " " << kernelY << " " << kernelX << " " << strideY << " " << strideX << " " << outChannels << " " << inChannels << endl;
	os << spectralNormMode << " " << weightNormMode << endl;

	MAT temp = W;
	temp.resize(W.size(), 1);
	os << temp << endl;
	os << b << endl;
	os << spectralNormMode << " " << weightNormMode << endl;
	if (weightNormMode) {
		temp = V;
		temp.resize(V.size(), 1);
		os << V << endl;
		temp = G;
		temp.resize(G.size(), 1);
		os << temp << endl;
	}
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
	in >> outChannels;
	in >> inChannels;
	features = inChannels * outChannels;
	
	// Load normalization settings
	in >> spectralNormMode;
	in >> weightNormMode;
	
	// Set up matrices.
	W = MAT(kernelY*features*kernelX, 1); // initialize as a column vector
	V = MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	b = MAT(getNOUT(), 1);
	V.setZero();
	G.setZero();
	b.setZero();
	w_stepper.reset();
	b_stepper.reset();

	// Load matrices
	for (size_t i = 0; i < W.size(); ++i) {
		in >> W(i, 0);
	}
	for (size_t i = 0; i < b.size(); ++i) {
		in >> b(i, 0);
	}

	W.resize(kernelY, features*kernelX);
	if (weightNormMode) {
		V.resize(V.size(), 1);
		for (size_t i = 0; i < V.size(); ++i) {
			in >> V(i, 0);
		}
		V.resize(kernelY, features*kernelX);
		for (size_t i = 0; i < G.size(); ++i) {
			in >> G(i, 0);
		}
	}
}