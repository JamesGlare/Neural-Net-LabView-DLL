#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "BatchBuffer.h"


/* Convolutional layer Constructors
* - Layer weights for other features are stored in x-direction (second index) in the MAT layer variable.
* - NOUTX, NOUTY only refers to the dimensions of a single feature.
* - In addition to some square or rectangular structure to be convolved over, a specified number of inputs can be given to the network
    which bypass the convolution. I refer to these input channels as "sidechannels". 
	The convention is that the sideChannel inputs are forward as the LAST ENTRIES in the input matrix.
	
	They are intended to be passed on to some dense layer where they can be incorporated in the stream of information.
* - The CNetLayer::NOUT variable contains information about the number of features.
*/
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, 
	uint32_t _features, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTX), NOUTY(_NOUTY), NINX(_NINX), NINY(_NINY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(_features*_NOUTX*_NOUTY+_sideChannels, _NINX*_NINY+_sideChannels, type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }) {
	// the layer matrix will act as convolutional kernel
	inFeatures = 1;
	init();
	assertGeometry();
}

ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTX, size_t _NOUTY, size_t _NINX, size_t _NINY, size_t _kernelX, size_t _kernelY, uint32_t _strideY, uint32_t _strideX, 
	uint32_t _features, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
: NOUTX(_NOUTX), NOUTY(_NOUTY), kernelX(_kernelX), kernelY(_kernelY), strideY(_strideY), strideX(_strideX), features(_features), sideChannels(_sideChannels),
PhysicalLayer(lower.getFeatures()*_features*_NOUTX*_NOUTY+_sideChannels,  type, MATIND{ _kernelY, _features*_kernelX }, MATIND{ _kernelY, _features*_kernelX }, MATIND{ 1,_features }, lower) {
	
	// We need to know how to interpre the inputs geometrically. Thus, we request number of features.
	inFeatures = lower.getFeatures();
	NINX = _NINX / inFeatures; //TODO 
	NINY = _NINY;

	init();
	assertGeometry();
}
// second most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _NINXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _sideChannels, actfunc_t type)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), NINX(_NINXY), NINY(_NINXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(_features*_NOUTXY*_NOUTXY+_sideChannels, _NINXY*_NINXY+_sideChannels,  type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }) {
	inFeatures = 1;
	init();
	assertGeometry();
}

// most convenient constructor
ConvolutionalLayer::ConvolutionalLayer(size_t _NOUTXY, size_t _kernelXY, uint32_t _stride, uint32_t _features, uint32_t _sideChannels, actfunc_t type, CNetLayer& lower)
	: NOUTX(_NOUTXY), NOUTY(_NOUTXY), kernelX(_kernelXY), kernelY(_kernelXY), strideY(_stride), strideX(_stride), features(_features), sideChannels(_sideChannels),
	PhysicalLayer(lower.getFeatures()*_features*_NOUTXY*_NOUTXY, type, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ _kernelXY, _features*_kernelXY }, MATIND{ 1,_features }, lower) {
	
	// We need to know how to interpre the inputs geometrically. Thus, we request number of features.
	inFeatures = lower.getFeatures(); // product of all features so far
	NINX = sqrt((lower.getNOUT()-sideChannels) / inFeatures); // sqrt(2* 5*5/2) = 5 
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
	assert(inFeatures*features*NOUTX*NOUTY + sideChannels == getNOUT());
	assert(NINX*NINY*inFeatures+sideChannels == getNIN());
	// Feature dimensions
	assert((strideX*NOUTX - strideX - NINX + kernelX) % 2 == 0);
	assert((strideY*NOUTY - strideY - NINY + kernelY) % 2 == 0);
}
void ConvolutionalLayer::init() {

	sideChannelBuffer = MAT(sideChannels, 1);
	sideChannelBuffer.setZero();
	deltaSave = MAT(getNOUT() - sideChannels, 1);
	deltaSave.setZero();
	actSave = MAT(getNOUT() - sideChannels, 1);
	actSave.setZero();
	//layer += MAT::Constant(layer.rows(), layer.cols(), 1.0f); // make everything positive
	layer *= ((fREAL) features) / layer.size();
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
	
	// (1) Take care of the sidechannel inputs
	if (sideChannels > 0) {
		sideChannelBuffer = inBelow.bottomRows(sideChannels);
		inBelow.conservativeResize(getNIN()-sideChannels,1); // drop bottom rows
	}
	// (2) Reshape the remaining input
	
	inBelow.resize(NINY, inFeatures*NINX);
	
	//inBelow = conv(inBelow, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features); // square convolution
	inBelow = conv_(inBelow, layer, NOUTY, NOUTX, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
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
			if (recursive) {
				if (sideChannels > 0) {
					inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
					inBelow.bottomRows(sideChannels) = sideChannelBuffer;
				}
				above->forProp(inBelow, false, true);
			}
		}	else {
			if (sideChannels > 0) {
				inBelow.conservativeResize(getNOUT(), 1); // append sideChannel-rows
				inBelow.bottomRows(sideChannels) = sideChannelBuffer;
			}
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
	if (sideChannels > 0) {
		sideChannelBuffer = deltaAbove.bottomRows(sideChannels);
		deltaAbove.conservativeResize(getNOUT() - sideChannels, 1);
	}
	deltaSave = deltaAbove; // just overwrite this matrix with an (NOUT-sideChannel)-sized vector

	if (getHierachy() != hierarchy_t::input) { // ... this is not an input layer.
		deltaSave.resize(NOUTY, inFeatures*features*NOUTX); // keep track of this!!!
		//deltaAbove = backPropConv_(deltaSave, layer, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		deltaAbove = antiConv_(deltaSave, layer, NINY, NINX, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		deltaAbove.resize(getNIN()-sideChannels, 1);
		
		if (sideChannels > 0) {
			deltaAbove.conservativeResize(getNIN(), 1);
			deltaAbove.bottomRows(sideChannels) = sideChannelBuffer;
		}
		deltaAbove = deltaAbove.cwiseProduct(below->getDACT()); // multiply with h'(aj), we dont need eval.
		deltaSave.resize(getNOUT() - sideChannels, 1); // resize back
		if (recursive) {
			below->backPropDelta(deltaAbove, true); // cascade...
		}
	}
}
// grad
MAT ConvolutionalLayer::grad(MAT& input) { // deltaSave: (NOUT-sideChannel) sized vector
	deltaSave.resize(NOUTY, inFeatures*features*NOUTX);
	if (getHierachy() == hierarchy_t::input) {
		if (sideChannels > 0) {
			sideChannelBuffer = input.bottomRows(sideChannels);
			input.conservativeResize(getNIN()-sideChannels, 1); // drop sidechannel inputs - not relevant for gradient
		}
		input.resize(NINY, inFeatures*NINX);

		MAT convoluted = convGrad_(input, deltaSave, kernelY, kernelX, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		//MAT convoluted = convGrad(input, deltaSave, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);

		deltaSave.resize(getNOUT()-sideChannels, 1); // make sure to resize to NOUT-sideChannels

		input.resize(getNIN()-sideChannels, 1);
		
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
		fromBelow.resize(NINY, inFeatures*NINX);
		
		MAT convoluted = convGrad_(fromBelow, deltaSave, kernelY, kernelX, strideY, strideX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features, inFeatures);
		//MAT convoluted = convGrad(fromBelow, deltaSave, strideY, strideX, kernelY, kernelX, padSize(NOUTY, NINY, kernelY, strideY), padSize(NOUTX, NINX, kernelX, strideX), features);
		
		deltaSave.resize(getNOUT()-sideChannels, 1);  // make sure to resize to NOUT-sideChannels
		if (sideChannels > 0) {
			fromBelow.conservativeResize(getNIN(), 1);
			fromBelow.bottomRows(sideChannels) = sideChannelBuffer;
		}
		return convoluted;
	}
}
void ConvolutionalLayer::saveToFile(ostream& os) const {
	os << NOUTY << " " << NOUTX << " " << NINY << " " << NINX << " " << kernelY << " " << kernelX << " " << strideY<< " " << strideX<< " " << features<< " " << inFeatures <<" " << sideChannels <<endl;
	MAT temp = layer;
	temp.resize(layer.size(), 1);
	os << temp<< endl;
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
	in >> sideChannels;

	layer = MAT(kernelY*features*kernelX,1); // initialize as a column vector
	V = MAT(kernelY, features*kernelX);
	G = MAT(1, features);

	V.setZero();
	G.setZero();
	stepper.reset();

	for (size_t i = 0; i < layer.size(); ++i) {
		in >> layer(i);
	}
	layer.resize(kernelY, features*kernelX);

}	