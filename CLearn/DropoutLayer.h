#pragma once
#include "defininitions.h"
#include "DiscarnateLayer.h"

#ifndef CNET_DROPOU
#define CNET_DROPOUT
/*	Implements a dropout operation.
*	Sets a certain proportion of inputs to zero,
*	but maintains the geometry (i.e. number of outputs).
*/
class DropoutLayer : public DiscarnateLayer {
public:
	DropoutLayer(fREAL ratio, size_t NIN);
	DropoutLayer(fREAL ratio, CNetLayer& lower);

	~DropoutLayer();
	layer_t whoAmI() const;

	// propagation
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive); // recursive
	inline fREAL getRatio() const { return ratio; };

private:
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
	void init();
	//void shuffleIndices();
	void randomize();
	void round();
	void assertGeometry();
	fREAL ratio;
	/* Keep a list of indices 
	* and shuffle before every forward pass
	* then, only forward the first NOUT indices.
	*/
	MAT zeroOne;
	//PermutationMatrix<Dynamic,Dynamic> permut;
};
#endif