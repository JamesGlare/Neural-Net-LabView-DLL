#include "stdafx.h"
#include "DropoutLayer.h"
#include <algorithm>

// Constructors
DropoutLayer::DropoutLayer(fREAL _ratio, size_t NIN ) : ratio(_ratio), DiscarnateLayer(NIN, NIN, actfunc_t::NONE) {
	init();
	assertGeometry();
}

DropoutLayer::DropoutLayer(fREAL _ratio, CNetLayer& lower) : ratio(_ratio), DiscarnateLayer(lower.getNOUT(), actfunc_t::NONE, lower) {
	init();
	assertGeometry();
}

void DropoutLayer::init() {
	/* This matrix type is automatically initialized.
	*/
	permut = PermutationMatrix<Dynamic, Dynamic>(getNIN());
	permut.setIdentity();
}
layer_t DropoutLayer::whoAmI() const {
	return layer_t::dropout;
}
void DropoutLayer::forProp(MAT& in, bool saveActivation, bool recursive) {
	// (1) shuffle indices
	shuffleIndices();
	//(2) Set stuff to zero
	setZeroAtIndex(in, permut.indices().cast<size_t>(), (size_t) ratio*getNIN());
	//(3) Resize down to NOUT (conserves top rows)
	in *= (1.0f + ratio);
	// (4) Pass on
	if (saveActivation)
		actSave = in;
	if (hierarchy != hierarchy_t::output && recursive)
		above->forProp(in, saveActivation, recursive);
}

void DropoutLayer::backPropDelta(MAT& deltaAbove, bool recursive) {

	if (hierarchy != hierarchy_t::input || !recursive) {
		// we make use of a C++11 construct below
		// unaryExpr takes functors which can be matrices
		// the output should have the same dimensionality
		// as the matrix permut.
		//deltaSave = permut.indices().unaryExpr(deltaAbove).cast<fREAL>();
		/////// DOESN'T Work with VC
		setZeroAtIndex(deltaAbove, permut.indices().cast<size_t>(), (size_t)ratio*getNIN());
		deltaAbove /=(1.0f+ratio); 
		
		if (recursive) {
			below->backPropDelta(deltaAbove, true);
		}
	}
}

void DropoutLayer::shuffleIndices() {
	// use raw pointers of the underlying array to create begin and end pointers
	random_shuffle(permut.indices().data(), permut.indices().data()+ permut.indices().size());
}
void DropoutLayer::assertGeometry() {
	assert(ratio<=1.0f);
	assert(ratio >= 0.0f);
}
// File functions - we don't need to save anything
void DropoutLayer::saveToFile(ostream& os) const {}
void DropoutLayer::loadFromFile(ifstream& in) {}
// Destructor
DropoutLayer::~DropoutLayer(){}