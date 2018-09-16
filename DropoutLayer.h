#pragma once
#include "defininitions.h"
#include "DiscarnateLayer.h"

#ifndef CNET_DROPOU
#define CNET_DROPOUT

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
	void shuffleIndices();
	void assertGeometry();
	fREAL ratio;

	/* Keep a list of indices 
	* and shuffle before every forward pass
	* then, only forward the first NOUT indices.
	*/
	PermutationMatrix<Dynamic,Dynamic> permut;
};
#endif

/*
class MaxPoolLayer : public DiscarnateLayer {

public:
	MaxPoolLayer(size_t NINXY, size_t maxOver);
	MaxPoolLayer(size_t maxOver, CNetLayer& lower);
	MaxPoolLayer(CNetLayer& lower);

	~MaxPoolLayer();
	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool saveActivation, bool recursive); // recursive
	void backPropDelta(MAT& delta, bool recursive); // recursive

	inline size_t getMaxOverX() { return maxOverX; }
	inline size_t getMaxOverY() { return maxOverY; }
	inline size_t getNINX() { return NINX; };
	inline size_t getNINY() { return NINY; };
	inline size_t getNOUTX() { return NOUTX; };
	inline size_t getNOUTY() { return NOUTY; };


private:
	size_t maxOverX;
	size_t maxOverY;
	size_t NINX;
	size_t NINY;
	size_t NOUTX;
	size_t NOUTY;
	MATINDEX indexX;
	MATINDEX indexY;

	MAT maxPool(const MAT& in, bool saveIndices);
	void assertGeometry();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
	void init();
}; */