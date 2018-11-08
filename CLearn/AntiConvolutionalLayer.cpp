#include "stdafx.h"
#include "AntiConvolutionalLayer.h"

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, uint32_t _features, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(_NOUTX*_NOUTY+ _sideChannels, features*_NINX*_NINY + _sideChannels, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }) {
	// the layer matrix will act as convolutional kernel
	
	inFeatures = 1;
	init();
}

AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY,  uint32_t _strideX, uint32_t _features, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(_NOUTX*_NOUTY+_sideChannels, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features },lower) {
	
	//TODO
	inFeatures = 1;
	init();
	assertGeometry();
}
// second most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(_NOUTXY*_NOUTXY + _sideChannels, _features*_NINXY*_NINXY + _sideChannels, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }) {
	
	inFeatures = 1;
	init();
	assertGeometry();
}

// most convenient constructor
AntiConvolutionalLayer::AntiConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(sqrt(lower.getNOUT() / _features)), NINY(sqrt(lower.getNOUT() / _features)), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), sideChannels(_sideChannels), features(_features),
	PhysicalLayer(_NOUTXY*_NOUTXY + _sideChannels, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features}, lower) {
	
	//TODO
	inFeatures = 1;
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
	assert(NOUTX*NOUTY+sideChannels == getNOUT());
	assert(features*NINX*NINY+sideChannels == getNIN());
	assert(antiConvoSize(NINY,kernelY, antiConvPad(NINY,strideY,kernelY,NOUTY),strideY)==NOUTY);
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
	for (uint32_t i = 0; i< features; ++i)
		VInversNorm._FEAT(i) /= normSum(V._FEAT(i));
}
MAT AntiConvolutionalLayer::gGrad(const MAT& grad) {
	MAT ret(1, features);

	for (uint32_t i = 0; i < features; ++i) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*vInversEntry; //(1,1)
	}
	return ret;
}
MAT AntiConvolutionalLayer::vGrad(const MAT& grad, MAT& ggrad) {
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
uint32_t AntiConvolutionalLayer::getFeatures() const {
	uint32_t tree_feature_product = features;
	if (getHierachy() != hierarchy_t::input) {
		tree_feature_product *= below->getFeatures();
	}
	return tree_feature_product;
}
void AntiConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {


	// (1) Take care of the sidechannel inputs
	if (sideChannels > 0) {
		sideChannelBuffer = inBelow.bottomRows(sideChannels);
		inBelow.conservativeResize(getNIN()-sideChannels, 1); // drop bottom rows
	}
	// (2) Reshape the remaining input

	inBelow.resize(NINY, features*NINX);
	inBelow =  antiConv(inBelow, layer, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); // square convolution//
	inBelow.resize(NOUTX*NOUTY, 1);
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
		deltaSave.resize(NOUTY, NOUTX); // keep track of this!!!
		deltaAbove = conv_(deltaSave, layer,NINY, NINX, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features,1);
		deltaAbove.resize(getNIN()-sideChannels, 1);
		if (sideChannels > 0) {
			deltaAbove.conservativeResize(getNOUT(), 1);
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
	deltaSave.resize(NOUTY, NOUTX);

	if (getHierachy() == hierarchy_t::input) {

		if (sideChannels > 0) {
			sideChannelBuffer = input.bottomRows(sideChannels);
			input.conservativeResize(NINY*NINX*features, 1); // drop sidechannel inputs - not relevant for gradient
		}

		input.resize(NINY, NINX*features);
		MAT convoluted = antiConvGrad(deltaSave, input, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); //MAT::Constant(kernelY, kernelX,2 );//

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
		fromBelow.resize(NINY, NINX*features);
		MAT convoluted = antiConvGrad(deltaSave, fromBelow, strideY, strideX, antiConvPad(NINY, strideY, kernelY, NOUTY), antiConvPad(NINX, strideX, kernelX, NOUTX), features); //MAT::Constant(kernelY, kernelX,2 );//
		deltaSave.resize(getNOUT() - sideChannels, 1);  // make sure to resize to NOUT-sideChannels
		return convoluted;
	}
}

void AntiConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << " " << NOUTX << " " << NINY << " " << NINX << " " << kernelY << " " << kernelX << " " << strideY << " " << strideX << " " << features << " " << inFeatures << " "<< sideChannels<<endl;
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
	in >> inFeatures;
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