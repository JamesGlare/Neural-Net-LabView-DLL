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
	inFeatures = 1;
	init();
	assertGeometry();
}

ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, uint32_t _features, actfunc_t type, CNetLayer& lower)
: NOUTX(_NOUTX), NOUTY(_NOUTY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features),
PhysicalLayer(lower.getFeatures()*_features*_NOUTX*_NOUTY,  type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }, lower) {
	
	// We need to know how to interpre the inputs geometrically. Thus, we request number of features.
	inFeatures = lower.getFeatures();
	NINX = _NINX / inFeatures; //TODO 
	NINY = _NINY;

	init();
	assertGeometry();
}
// second most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(_features*_NOUTXY*_NOUTXY, _NINXY*_NINXY,  type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }) {
	inFeatures = 1;
	init();
	assertGeometry();
}

// most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features),
	PhysicalLayer(lower.getFeatures()*_features*_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }, lower) {
	
	// We need to know how to interpre the inputs geometrically. Thus, we request number of features.
	inFeatures = lower.getFeatures(); // product of all features so far
	NINX = sqrt(lower.getNOUT() / inFeatures); // sqrt(2* 5*5/2) = 5 
	NINY = NINX; // this is only for squares.
	
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
	assert(inFeatures*features*NOUTX*NOUTY == getNOUT());
	assert(NINX*NINY*inFeatures == getNIN());
	// Feature dimensions
	assert((strideX*NOUTX - strideX - NINX + kernelX) % 2 == 0);
	assert((strideY*NOUTY - strideY - NINY + kernelY) % 2 == 0);
}
void ConvolutionalLayer::init() {

	//layer *= ((fREAL) features) / layer.size();
}
// Select submatrix of ith feature
const MAT& ConvolutionalLayer::getIthFeature(size_t i) {
	return layer._FEAT(i);
}
// weight normalization reparametrize
void ConvolutionalLayer::updateW() {
	for (uint32_t i = 0; i < features; i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		layer._FEAT(i) = G(0, i)*vInversEntry *V._FEAT(i);
	}
}

void ConvolutionalLayer::initV() {
	V = layer;
	//normalizeV();
}
void ConvolutionalLayer::normalizeV() {
	for (uint32_t i = 0; i< features; i++)
		V._FEAT(i) *= 1.0f / normSum(V._FEAT(i));
}
void ConvolutionalLayer::initG() {
	for (uint32_t i = 0; i< features; i++)
		G(0, i) = normSum(layer._FEAT(i));
	//G.setOnes();
}
void ConvolutionalLayer::inversVNorm() {

	VInversNorm.setOnes();
	for (uint32_t i = 0; i< features; i++)
		VInversNorm._FEAT(i) /= normSum(V._FEAT(i));
}
MAT ConvolutionalLayer::gGrad(const MAT& grad) {
	MAT ret(1, features);

	for (uint32_t i = 0; i < features; i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0); // take any entry
		ret(0, i) = (grad._FEAT(i)).cwiseProduct(V._FEAT(i)).sum()*vInversEntry; //(1,1)
	}
	return ret;
}
MAT ConvolutionalLayer::vGrad(const MAT& grad, MAT& ggrad) {
	MAT out = grad; // same dimensions as grad
					// (1) multiply rows of grad with G's
	for (uint32_t i = 0; i < features; i++) {
		fREAL vInversEntry = (VInversNorm._FEAT(i))(0, 0);
		out._FEAT(i) *= G(0, i)*vInversEntry;
		// (2) subtract 
		vInversEntry *= vInversEntry;
		out._FEAT(i) -= G(0, i)*ggrad(0, i)*V._FEAT(i)*vInversEntry;
	}
	return out;
}
void ConvolutionalLayer::forProp(MAT& inBelow, bool training, bool recursive) {
	inBelow.resize(NINY, inFeatures*NINX);
	
	//inBelow = conv(inBelow, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features); // square convolution
	inBelow = conv_(inBelow, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
	inBelow.resize(inFeatures*features*NOUTX*NOUTY, 1);
	if (training) {
		actSave = inBelow;
		if (getHierachy() != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act);
			if(recursive)
				above->forProp(inBelow, true, true);
		} 
	} else {
		if (getHierachy() != hierarchy_t::output) {
			inBelow = inBelow.unaryExpr(act);
			if(recursive)
				above->forProp(inBelow, false, true);
		}
	}
}
uint32_t ConvolutionalLayer::getFeatures() const {
	uint32_t tree_feature_product = features;
	if (getHierachy() != hierarchy_t::input) {
		tree_feature_product *= below->getFeatures();
	}
	return tree_feature_product;
}
// backprop
void ConvolutionalLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	deltaSave = deltaAbove;

	if (getHierachy() != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, inFeatures*features*NOUTX); // keep track of this!!!
		deltaAbove = backPropConv_(deltaSave, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		//deltaAbove = antiConv(deltaSave, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		deltaAbove.resize(getNIN(), 1);
		//assert(below->getDACT().size() == NIN);
		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj), we dont need eval.
		deltaSave.resize(getNOUT(), 1); // resize back
		if(recursive)
			below->backPropDelta(deltaAbove, true); // cascade...
	}
}
// grad
MAT ConvolutionalLayer::grad(MAT& input) {
	deltaSave.resize(NOUTY, inFeatures*features*NOUTX);
	if (getHierachy() == hierarchy_t::input) {
		input.resize(NINY, inFeatures*NINX);
		//MAT convoluted = convGrad(deltaSave, input, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		MAT convoluted = convGrad_(input, deltaSave, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		deltaSave.resize(getNOUT(), 1);
		input.resize(getNIN(), 1);
		return convoluted;
	} else {
		MAT fromBelow = below->getACT();
		fromBelow.resize(NINY, inFeatures*NINX);
		//MAT convoluted = conv( deltaSave, fromBelow, stride, padSize(kernelY,NOUTY, NINY, stride), padSize(kernelX,  NOUTX, NINX, stride)).reverse();
		MAT convoluted = convGrad_(fromBelow, deltaSave, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		//MAT convoluted = deltaActConv(deltaSave, fromBelow, kernelY, kernelX, stride, padSize(NOUTY, NINY, kernelY, stride), padSize(NOUTX, NINX, kernelX, stride));
		deltaSave.resize(getNOUT(), 1);
		return convoluted;
	}
}
void ConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << "\t" << NOUTX << "\t" << NINY << "\t" << NINX << "\t" << kernelY << "\t" << kernelX << "\t"  << strideY<< "\t" << strideX<< "\t" << features<<  "\t" <<inFeatures<<endl;
	for (size_t f = 0; f < features; ++f) {
		for (size_t i = 0; i < kernelY; ++i) {
			for (size_t j = 0; j < kernelX; ++j) {
				os << layer(i, j + f*kernelX) << "\t";
			}
		}
	}
	os << endl;
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
	in >> inFeatures;
	layer = MAT(kernelY, features*kernelX);
	V = MAT(kernelY, features*kernelX);
	G = MAT(1, features);
	V.setZero();
	G.setZero();
	stepper.reset();

	for (size_t f = 0; f < features; ++f) {
		for (size_t i = 0; i < kernelY; ++i) {
			for (size_t j = 0; j < kernelX; ++j) {
				in >> layer(i, j + f*kernelX);
			}
		}
	}
}	