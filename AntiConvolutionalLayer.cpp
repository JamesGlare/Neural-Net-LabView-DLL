#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), CNetLayer(_NOUTX*_NOUTY, _NINX*_NINY, type) {
	// the layer matrix will act as convolutional kernel
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _stride, actfunc_t type, CNetLayer& const lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), stride(_stride), CNetLayer(_NOUTX*_NOUTY, type, lower) {
	init();
	assertGeometry();
}
// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t _stride, actfunc_t type)
	: NOUTX(NOUTXY), NOUTY(NOUTXY), NINX(NINXY), NINY(NINXY), kernelX(kernelXY), kernelY(kernelXY), stride(_stride), CNetLayer(NOUTXY*NOUTXY, NINXY*NINXY, type) {
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t _stride, actfunc_t type, CNetLayer& const lower)
	: NOUTX(NOUTXY), NOUTY(NOUTXY), NINX(sqrt(lower.getNOUT())), NINY(sqrt(lower.getNOUT())), kernelX(kernelXY), kernelY(kernelXY), stride(_stride), CNetLayer(NOUTXY*NOUTXY, type, lower) {
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
	assert(antiConvoSize(NINY,kernelY,0,stride)==NOUTY);
	assert(antiConvoSize(NINX, kernelX, 0,stride) == NOUTX);
}
void AntiConvolutionalLayer::init() {
	layer = MAT(kernelY, kernelX);
	vel = MAT(kernelY, kernelX);
	actSave = MAT(NOUT, 1);
	deltaSave = MAT(NOUT, 1);
	prevStep = MAT(layer.rows(), layer.cols());

	prevStep.setConstant(0);
	gauss(layer);
	vel.setConstant(0);
	actSave.setConstant(0);
	deltaSave.setConstant(0);
}

void AntiConvolutionalLayer::forProp(MAT& inBelow, bool saveActivation) {
	inBelow.resize(NINY, NINX);
	MAT convoluted = antiConv(inBelow, layer, stride, 0, 0 ); // square convolution
	convoluted.resize(NOUTX*NOUTY, 1);
	if (saveActivation) {
		actSave = convoluted;
		if (hierarchy != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			above->forProp(inBelow, true);
		}
		else {
			inBelow = actSave;
		}
	}
	else {
		if (hierarchy != hierarchy_t::output) {
			inBelow = convoluted.unaryExpr(act);
			above->forProp(inBelow, false);
		}
		else {
			inBelow = convoluted;
		}
	}
}
// backprop
void AntiConvolutionalLayer::backPropDelta(MAT& const deltaAbove) {
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, NOUTX); // keep track of this!!!
		MAT convoluted = conv(deltaSave, layer, stride, 1,0,0);
		convoluted.resize(NIN, 1);
		convoluted = convoluted.cwiseProduct(below->getDACT()); // multiply with h'(aj)
		deltaSave.resize(NOUT, 1); // resize back
		below->backPropDelta(convoluted); // cascade...
	}
}
// grad
MAT AntiConvolutionalLayer::grad(MAT& const input) {
	deltaSave.resize(NOUTY, NOUTX);

	if (hierarchy == hierarchy_t::input) {
		input.resize(NINY, NINX);
		MAT convoluted = conv(deltaSave, input, stride, 1,0, 0);
		deltaSave.resize(NOUT, 1);
		input.resize(NIN, 1);
		return convoluted;
	}
	else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, NINX);
		MAT convoluted = conv(deltaSave, fromBelow, stride, 1, 0, 0);
		deltaSave.resize(NOUT, 1);
		return convoluted;
	}
}
fREAL AntiConvolutionalLayer::applyUpdate(learnPars pars, MAT& const input) {
	fREAL gamma = 0;
	fREAL denom = 0;
	if (pars.conjugate) {
		// treat vel as g_(i-1)
		denom = vel.cwiseProduct(vel).sum(); // should be scalar
		MAT gi = -grad(input);
		gamma = gi.cwiseProduct(gi - vel).sum() / denom;
		if (!isnan(gamma)) {
			prevStep = gi + gamma*prevStep; // save step
			layer = (1.0f - pars.lambda)*layer + pars.eta*gi; // do the actual step
			vel = gi; // save negative gradient
		}
		else {
			resetConjugate(input);
		}
	} else {
		prevStep = -grad(input);
		vel = pars.gamma*vel - pars.eta*prevStep;
		if (vel.allFinite())
			layer = (1.0f - pars.lambda)*layer - vel; // this reverse call forces us to implement this function in the derived classes
	}
	if (hierarchy != hierarchy_t::output) {
		above->applyUpdate(pars, input);
	}
	return gamma;
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

	for (size_t i = 0; i < kernelY; i++) {
		for (size_t j = 0; j < kernelX; j++) {
			in >> layer(i, j);
		}
	}
}