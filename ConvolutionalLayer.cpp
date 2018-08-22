#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "BatchBuffer.h"


/* Convolutional layer Constructors
* - Layer weights for other features are stored in x-direction (second index) in the MAT layer variable.
* - NOUTX, NOUTY only refers to the dimensions of a single feature.
* - The CNetLayer::NOUT variable contains information about the number of features.
*/
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, uint32_t _features, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features),
	PhysicalLayer(_features*_NOUTX*_NOUTY, _NINX*_NINY, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }) {
	// the layer matrix will act as convolutional kernel
	init();
}

ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, uint32_t _features, actfunc_t type, CNetLayer& const lower)
: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features),
PhysicalLayer(_features*_NOUTX*_NOUTY,  type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }, lower) {
	init();
	assertGeometry();
}
// second most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(_features*_NOUTXY*_NOUTXY, _NINXY*_NINXY,  type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }) {
	init();
	assertGeometry();
}

// most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT())), NINY(sqrt(lower.getNOUT())), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(_features*_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }, lower) {
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
	assert(features*NOUTX*NOUTY == getNOUT());
	assert(NINX*NINY == getNIN());
	// Feature dimensions
	assert((strideX*NOUTX - strideX - NINX + kernelX) % 2 == 0);
	assert((strideY*NOUTY - strideY - NINY + kernelY) % 2 == 0);
}
void ConvolutionalLayer::init() {

	//layer += MAT::Constant(layer.rows(), layer.cols(), 1);
	//layer *= features / layer.size();
}
// Select submatrix of ith feature
const MAT& ConvolutionalLayer::getIthFeature(size_t i) {
	return layer._FEAT(i);
}
// weight normalization reparametrize
void ConvolutionalLayer::updateW() {
	MAT inversV = inversVNorm();
	for(uint32_t i=0; i< features; i++)
		layer._FEAT(i) = G(0,i)* inversV(0, i) *V._FEAT(i);
}

void ConvolutionalLayer::initV() {
	V = layer;
	normalizeV();
}
void ConvolutionalLayer::normalizeV() {
	for (uint32_t i = 0; i< features; i++)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void ConvolutionalLayer::initG() {
	for (uint32_t i = 0; i< features; i++)
		G(0, i) = normSum(layer._FEAT(i));
}
MAT ConvolutionalLayer::inversVNorm() {
	MAT out(1, features);
	out.setOnes();
	for (uint32_t i = 0; i< features; i++)
		out(0,i) /= normSum(V._FEAT(i));
	return out;
}
MAT ConvolutionalLayer::gGrad(MAT& const grad) {
	MAT ret(1, features);
	MAT inversV = inversVNorm();
	for (uint32_t i = 0; i < features; i++)
		ret(0,i)=(grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*inversV(0, i); //(1,1)
	return ret;
}
MAT ConvolutionalLayer::vGrad(MAT& const grad, MAT& const ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	MAT inversV = inversVNorm();
	for (uint32_t i = 0; i < features; i++) {
		out._FEAT(i) *= G(0, i)*inversV(0, i);
		// (2) subtract 
		out._FEAT(i) -= G(0, i)*ggrad(0, i)*V._FEAT(i)*inversV(0,i)*inversV(0,i);
	}
	return out;
}
void ConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	inBelow.resize(NINY, NINX);
	inBelow = conv(inBelow, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features); // square convolution
	inBelow.resize(features*NOUTX*NOUTY, 1);
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
void ConvolutionalLayer::backPropDelta(MAT& const deltaAbove, bool recursive) {
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, features*NOUTX); // keep track of this!!!
		deltaAbove = antiConv(deltaSave, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		//MAT convoluted = conv(deltaSave, layer, stride, padSize(NINY, NOUTY, kernelY, stride), padSize(NINX, NOUTX, kernelX, stride));
		deltaAbove.resize(NIN, 1);
		//assert(below->getDACT().size() == NIN);
		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj), we dont need eval.
		deltaSave.resize(NOUT, 1); // resize back
		if(recursive)
			below->backPropDelta(deltaAbove, true); // cascade...
	}
}
// grad
MAT ConvolutionalLayer::grad(MAT& const input) {
	deltaSave.resize(NOUTY, features*NOUTX);
	if (hierarchy == hierarchy_t::input) {
		input.resize(NINY, NINX);
		//MAT convoluted = conv(deltaSave, input, stride, padSize(kernelY,  NOUTY, NINY, stride), padSize(kernelX,  NOUTX, NINX, stride)).reverse();
		MAT convoluted = convGrad(deltaSave, input, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		
		//MAT convoluted = deltaActConv(deltaSave, input, kernelY, kernelX, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		deltaSave.resize(NOUT, 1);
		input.resize(NIN, 1);
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, NINX);
		//MAT convoluted = conv( deltaSave, fromBelow, stride, padSize(kernelY,NOUTY, NINY, stride), padSize(kernelX,  NOUTX, NINX, stride)).reverse();
		MAT convoluted = convGrad(deltaSave, fromBelow, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		//MAT convoluted = deltaActConv(deltaSave, fromBelow, kernelY, kernelX, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		deltaSave.resize(NOUT, 1);
		return convoluted;
	}
}
void ConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << "\t" << NOUTX << "\t" << NINY << "\t" << NINX << "\t" << kernelY << "\t" << kernelX << "\t"  << strideY<< "\t" << strideX<< "\t" << features<<endl;
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
	in >> strideY;
	in >> strideX;
	in >> features;
	layer = MAT(kernelY, features*kernelX);
	V = MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	V.setZero();
	G.setZero();

	for (size_t i = 0; i < kernelY; i++) {
		for (size_t j = 0; j < features*kernelX; j++) {
			in >> layer(i, j);
		}
	}
}