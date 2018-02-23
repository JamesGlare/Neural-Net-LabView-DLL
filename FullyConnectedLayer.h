#pragma once

#include "CNetLayer.h"

#ifndef FULLYCONNECTEDLAYER
#define FULLYCONNECTEDLAYER

class FullyConnectedLayer : public CNetLayer {
	public:
		// Constructors
		FullyConnectedLayer(size_t NOUT, size_t NIN, actfunc_t type, fREAL min, fREAL max); // specified in .cpp
		FullyConnectedLayer(size_t NOUT, actfunc_t type, fREAL min, fREAL max, CNetLayer& const lower);
		~FullyConnectedLayer();
		layer_t whoAmI() const;

		// propagation 
		// forProp
		void forProp(MAT& in, bool saveActivation);
		// backprop
		MAT grad(MAT& const input);
		fREAL applyUpdate(learnPars pars, MAT& const input); // recursive
		void backPropDelta(MAT& const delta);

private:
	// initialization
	void init(fREAL min, fREAL max);
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif