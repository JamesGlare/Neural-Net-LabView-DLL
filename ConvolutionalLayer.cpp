#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "BatchBuffer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), 
	PhysicalLayer(_NOUTX*_NOUTY, _NINX*_NINY, type, MATIND{ _kernelY, _kernelX }, MATIND{ _kernelY, _kernelX }, MATIND{ 1,1 }) {
	// the layer matrix will act as convolutional kernel
	init();
}

ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride , actfunc_t type, CNetLayer& const lower)
: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), 
PhysicalLayer(_NOUTX*_NOUTY,  type, MATIND{ _kernelY, _kernelX }, MATIND{ _kernelY, _kernelX }, MATIND{ 1,1 }, lower) {
	init();
	assertGeometry();
}
// second most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, actfunc_t type) 
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), stride(_stride),
	PhysicalLayer(_NOUTXY*_NOUTXY, _NINXY*_NINXY,  type, MATIND{ _kernelXY, _kernelXY }, MATIND{ _kernelXY, _kernelXY }, MATIND{ 1,1 }) {
	init();
	assertGeometry();
}

// most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT())), NINY(sqrt(lower.getNOUT())), kernelX(_kernelXY), kernelY(_kernelXY), stride(_stride),
	PhysicalLayer(_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _kernelXY }, MATIND{ _kernelXY, _kernelXY }, MATIND{ 1,1 }, lower) {
	init();
	assertGeometry();
}
// destructor
ConvolutionalLayer::~ConvolutionalLayer() {
	layer.resize(0, 0);
}

layer_t ConvolutionalLayer::whoAmI() const {
	return layer_t::convolutional;
}
void ConvolutionalLayer::assertGeometry() {
	assert(NOUTX*NOUTY == getNOUT());
	assert(NINX*NINY == getNIN());
	assert((stride*NOUTX - stride - NINX + kernelX) % 2 == 0);
	assert((stride*NOUTY - stride - NINY + kernelY) % 2 == 0);
}
void ConvolutionalLayer::init() {

	layer += MAT::Constant(layer.rows(), layer.cols(), 1);
	layer *= 1 / layer.size();
}
// weight normalization reparametrize
void ConvolutionalLayer::updateW() {
	layer = G(0,0)* inversVNorm()(0, 0) *V;
}

void ConvolutionalLayer::initV() {
	V = layer;
	normalizeV();
}
void ConvolutionalLayer::normalizeV() {
	V *= 1.0f / normSum(V);
}
void ConvolutionalLayer::initG() {
	G(0, 0) = normSum(layer);
}
MAT ConvolutionalLayer::inversVNorm() {
	MAT out(1, 1);
	out.setOnes();
	out /= normSum(V);
	return out;
}
MAT ConvolutionalLayer::gGrad(MAT& const grad) {
	MAT ret(1, 1);
	ret(0,0)=grad.cwiseProduct(V).sum()*inversVNorm()(0, 0); //(1,1)
	return ret;
}
MAT ConvolutionalLayer::vGrad(MAT& const grad, MAT& const ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	fREAL inversV = inversVNorm()(0, 0);
	out *= G(0, 0)*inversV;
	// (2) subtract 
	out -= G(0, 0)*ggrad(0, 0)*V*inversV*inversV;
	return out;
}
void ConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	inBelow.resize(NINY, NINX);
	inBelow = conv(inBelow, layer, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride)); // square convolution
	inBelow.resize(NOUTX*NOUTY, 1);
	if (training) {
		actSave = inBelow;
		if (hierarchy != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			if(recursive)
				above->forProp(inBelow, true, true);
		} 
	} else {
		if (hierarchy != hierarchy_t::output) {
			inBelow = inBelow.unaryExpr(act);
			if(recursive)
				above->forProp(inBelow, false, true);
		}
	}
}
// backprop
void ConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, NOUTX); // keep track of this!!!
		deltaAbove = antiConv(deltaSave, layer, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		//MAT convoluted = conv(deltaSave, layer, stride, padSize(NINY, NOUTY, kernelY, stride), padSize(NINX, NOUTX, kernelX, stride));
		deltaAbove.resize(NIN, 1);
		//assert(below->getDACT().size() == NIN);
		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj)
		deltaSave.resize(NOUT, 1); // resize back
		if(recursive)
			below->backPropDelta(deltaAbove, true); // cascade...
	}
}
// grad
MAT ConvolutionalLayer::grad(MAT& const input) {
	deltaSave.resize(NOUTY, NOUTX);
	if (hierarchy == hierarchy_t::input) {
		input.resize(NINY, NINX);
		//MAT convoluted = conv(deltaSave, input, stride, padSize(kernelY,  NOUTY, NINY, stride), padSize(kernelX,  NOUTX, NINX, stride)).reverse();
		MAT convoluted = convGrad(deltaSave, input, stride, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		
		//MAT convoluted = deltaActConv(deltaSave, input, kernelY, kernelX, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		deltaSave.resize(NOUTY*NOUTX, 1);
		input.resize(NIN, 1);
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, NINX);
		//MAT convoluted = conv( deltaSave, fromBelow, stride, padSize(kernelY,NOUTY, NINY, stride), padSize(kernelX,  NOUTX, NINX, stride)).reverse();
		MAT convoluted = convGrad( deltaSave, fromBelow, stride, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		//MAT convoluted = deltaActConv(deltaSave, fromBelow, kernelY, kernelX, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		deltaSave.resize(NOUT, 1);
		return convoluted;
	}
}
void ConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << "\t" << NOUTX << "\t" << NINY << "\t" << NINX << "\t" << kernelY << "\t" << kernelX << "\t"  << endl;
	os << layer;
}
// first line has been read already
void ConvolutionalLayer::loadFromFile(ifstream& in) {
	in >> NOUTY;
	in >> NOUTX;
	in >> NINY;
	in >> NINX;
	in >> kernelY;
	in >> kernelX;
	layer = MAT(kernelY, kernelX);
	V = MAT(kernelY, kernelX);
	G = MAT(1, 1);
	V.setZero();
	G.setZero();

	for (size_t i = 0; i < kernelY; i++) {
		for (size_t j = 0; j < kernelX; j++) {
			in >> layer(i, j);
		}
	}
}