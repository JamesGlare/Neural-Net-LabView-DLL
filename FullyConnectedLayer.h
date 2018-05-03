#pragma once
#include "PhysicalLayer.h"

#ifndef CNET_FULLYCONNECTEDLAYER
#define CNET_FULLYCONNECTEDLAYER

class FullyConnectedLayer : public PhysicalLayer {
	public:
		// Constructors
		FullyConnectedLayer(size_t NOUT, size_t NIN, actfunc_t type); // specified in .cpp
		FullyConnectedLayer(size_t NOUT, actfunc_t type, CNetLayer& const lower);
		~FullyConnectedLayer();
		layer_t whoAmI() const;

		// propagation 
		// forProp
		void forProp(MAT& in, bool saveActivation);
		// backprop
		MAT grad(MAT& const input);
		void backPropDelta(MAT& const delta);

private:
	// initialization
	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif