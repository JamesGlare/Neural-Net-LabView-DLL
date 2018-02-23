#pragma once
#include "CNetLayer.h"

#ifndef CMAXPOOL
#define CMAXPOOL


class MaxPoolLayer : public CNetLayer {

public:
	MaxPoolLayer(size_t NINXY, size_t maxOver);
	MaxPoolLayer(size_t maxOver, CNetLayer& const lower);
	MaxPoolLayer(CNetLayer& const lower);

	~MaxPoolLayer();
	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool saveActivation); // recursive
	// backprop
	MAT grad(MAT& const input);
	void backPropDelta(MAT& const delta); // recursive
	fREAL applyUpdate(learnPars pars, MAT& const input); // recursive

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
};

#endif