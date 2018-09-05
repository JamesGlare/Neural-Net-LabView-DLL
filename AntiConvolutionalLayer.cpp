#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, uint32_t _features, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features),
	PhysicalLayer(_NOUTX*_NOUTY, features*_NINX*_NINY, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }) {
	// the layer matrix will act as convolutional kernel
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY,  uint32_t _strideX, uint32_t _features, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features),
	PhysicalLayer(_NOUTX*_NOUTY, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features },lower) {
	init();
	assertGeometry();
}
// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(_NOUTXY*_NOUTXY, _features*_NINXY*_NINXY, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }) {
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT() / _features)), NINY(sqrt(lower.getNOUT() / _features)), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features}, lower) {
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
	assert(NOUTX*NOUTY == getNOUT());
	assert(features*NINX*NINY == getNIN());
	assert(antiConvoSize(NINY,kernelY, antiConvPad(NINY,strideY,kernelY,NOUTY),strideY)==NOUTY);
	assert(antiConvoSize(NINX, kernelX, antiConvPad(NINX, strideX, kernelX, NOUTX),strideX) == NOUTX);
}
void AntiConvolutionalLayer::init() {
	MAT gauss_init = MAT::Constant(kernelY, kernelX, 1.0f);
	gauss(gauss_init); // make a gaussian
	for (size_t f = 0; f < features; ++f) {
		layer._FEAT(f) = gauss_init + 1.0f/(kernelY*kernelX)*MAT::Random(kernelY, kernelX);
	}
		
}
// weight normalization reparametrize
// weight normalization reparametrize
void AntiConvolutionalLayer::updateW() {
	MAT inversV = inversVNorm();
	for (uint32_t i = 0; i< features; i++)
		layer._FEAT(i) = G(0, i)*inversV(0, i) *V._FEAT(i);
}

void AntiConvolutionalLayer::initV() {
	V = layer;
	//normalizeV();
}
void AntiConvolutionalLayer::normalizeV() {
	for (uint32_t i = 0; i< features; i++)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void AntiConvolutionalLayer::initG() {
	for (uint32_t i = 0; i< features; i++)
		G(0, i) = normSum(layer._FEAT(i));
}
MAT AntiConvolutionalLayer::inversVNorm() {
	MAT out(1, features);
	out.setOnes();
	for (uint32_t i = 0; i< features; i++)
		out(0, i) /= normSum(V._FEAT(i));
	return out;
}
MAT AntiConvolutionalLayer::gGrad(MAT& const grad) {
	MAT ret(1, features);
	MAT inversV = inversVNorm();
	for (uint32_t i = 0; i< features; i++)
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*inversV(0, i); //(1,1)
	return ret;
}
MAT AntiConvolutionalLayer::vGrad(MAT& const grad, MAT& const ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	MAT inversV = inversVNorm();
	for (uint32_t i = 0; i < features; i++) {
		out._FEAT(i) *= G(0, i)*inversV(0, i);
		// (2) subtract 
		out._FEAT(i) -= G(0, i)*ggrad(0, i)*V._FEAT(i)*inversV(0, i)*inversV(0, i);
	}
	return out;
}
void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	inBelow.resize(NINY, features*NINX);
	inBelow =  antiConv(inBelow, layer, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); // square convolution//
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
			if (recursive)
				above->forProp(inBelow, false, true);
		}
	}
}
// backprop
void AntiConvolutionalLayer::backPropDelta(MAT& const deltaAbove, bool recursive) {
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, NOUTX); // keep track of this!!!
		deltaAbove = conv(deltaSave, layer, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features);
		deltaAbove.resize(NIN, 1);
		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj)
		deltaSave.resize(NOUT, 1); // resize back
		if(recursive)
			below->backPropDelta(deltaAbove, true); // cascade...
	}
}

// grad
MAT AntiConvolutionalLayer::grad(MAT& const input) {
	deltaSave.resize(NOUTY, NOUTX);

	if (hierarchy == hierarchy_t::input) {
		input.resize(NINY, NINX*features);
		MAT convoluted = antiConvGrad(deltaSave, input, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); //MAT::Constant(kernelY, kernelX,2 );//
		deltaSave.resize(NOUT, 1);
		input.resize(NIN, 1);
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, NINX*features);
		MAT convoluted = antiConvGrad(deltaSave, fromBelow, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); //MAT::Constant(kernelY, kernelX,2 );//
		deltaSave.resize(NOUT, 1);
		return convoluted;
	}
}

void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << "\t" << NOUTX << "\t" << NINY << "\t" << NINX << "\t" << kernelY << "\t" << kernelX << "\t" << strideY << "\t" << strideX << "\t"<<features<<endl;
	os << layer;
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
	layer = MAT(kernelY, features*kernelX);
	V= MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	V.setZero();
	G.setZero();

	for (size_t i = 0; i < kernelY; i++) {
		for (size_t j = 0; j < features*kernelX; j++) {
			in >> layer(i, j);
		}
	}
}