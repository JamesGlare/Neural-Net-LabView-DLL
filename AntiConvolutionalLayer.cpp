#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), 
	PhysicalLayer(_NOUTX*_NOUTY, _NINX*_NINY, type, MATIND{ _kernelY, _kernelX }, MATIND{ _kernelY, _kernelX }, MATIND{ 1,1 }) {
	// the layer matrix will act as convolutional kernel
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), 
	PhysicalLayer(_NOUTX*_NOUTY, type, MATIND{ _kernelY, _kernelX }, MATIND{ _kernelY, _kernelX }, MATIND{ 1,1 },lower) {
	init();
	assertGeometry();
}
// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), stride(_stride),
	PhysicalLayer(_NOUTXY*_NOUTXY, _NINXY*_NINXY, type, MATIND{ _kernelXY, _kernelXY }, MATIND{ _kernelXY, _kernelXY }, MATIND{ 1,1 }) {
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT())), NINY(sqrt(lower.getNOUT())), kernelX(_kernelXY), kernelY(_kernelXY), stride(_stride),
	PhysicalLayer(_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _kernelXY }, MATIND{ _kernelXY, _kernelXY }, MATIND{ 1,1 }, lower) {
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
	assert(NINX*NINY == getNIN());
	assert(antiConvoSize(NINY,kernelY, antiConvPad(NINY,stride,kernelY,NOUTY),stride)==NOUTY);
	assert(antiConvoSize(NINX, kernelX, antiConvPad(NINX, stride, kernelX, NOUTX),stride) == NOUTX);
}
void AntiConvolutionalLayer::init() {

}
// weight normalization reparametrize
void AntiConvolutionalLayer::updateW() {
	layer = G(0, 0)* inversVNorm()(0, 0) *V;
}

void AntiConvolutionalLayer::initV() {
	V = layer;
	normalizeV();
}
MAT AntiConvolutionalLayer::inversVNorm() {
	MAT out(1, 1);
	out.setOnes();
	out /= normSum(V);
	return out;
}
void AntiConvolutionalLayer::normalizeV() {
 		V *= 1.0f / normSum(V);
}
void AntiConvolutionalLayer::initG() {
	G(0,0) = normSum(layer);
}
MAT AntiConvolutionalLayer::gGrad(MAT& const grad) {
	MAT ret(1, 1);
	ret(0, 0) = grad.cwiseProduct(V).sum()*inversVNorm()(0, 0); //(1,1)
	return ret;
}
MAT AntiConvolutionalLayer::vGrad(MAT& const grad, MAT& const ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	fREAL inversV = inversVNorm()(0, 0);
	out *= G(0, 0)*inversV;
	// (2) subtract 
	out -= G(0, 0)*ggrad(0, 0)*V*inversV*inversV;
	return out;
}
void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	inBelow.resize(NINY, NINX);
	inBelow =  antiConv(inBelow, layer, stride, antiConvPad(NINY, stride, kernelY, NOUTY), antiConvPad(NINX, stride, kernelX, NOUTX)); // square convolution//
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
void AntiConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, NOUTX); // keep track of this!!!
		deltaAbove = conv(deltaSave, layer, stride, antiConvPad(NINY, stride, kernelY, NOUTY), antiConvPad(NINX, stride, kernelX, NOUTX));
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
		input.resize(NINY, NINX);
		MAT convoluted = antiConvGrad(deltaSave, input, stride, antiConvPad(NINY, stride, kernelY, NOUTY), antiConvPad(NINX, stride, kernelX, NOUTX)); //MAT::Constant(kernelY, kernelX,2 );//
		deltaSave.resize(NOUT, 1);
		input.resize(NIN, 1);
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, NINX);
		MAT convoluted = antiConvGrad(deltaSave, fromBelow, stride, antiConvPad(NINY, stride, kernelY, NOUTY), antiConvPad(NINX, stride, kernelX, NOUTX));
		deltaSave.resize(NOUT, 1);
		return convoluted;
	}
}

void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << "\t" << NOUTX << "\t" << NINY << "\t" << NINX << "\t" << kernelY << "\t" << kernelX << "\t" << endl;
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
	layer = MAT(kernelY, kernelX);
	V= MAT(kernelY, kernelX);
	G = MAT(1, 1);
	V.setZero();
	G.setZero();

	for (size_t i = 0; i < kernelY; i++) {
		for (size_t j = 0; j < kernelX; j++) {
			in >> layer(i, j);
		}
	}
}