#pragma once
#include "PhysicalLayer.h"

#ifndef CNET_FULLYCONNECTEDLAYER
#define CNET_FULLYCONNECTEDLAYER

class FullyConnectedLayer : public PhysicalLayer {
	public:
		// Constructors
		FullyConnectedLayer(size_t NOUT, size_t NIN, actfunc_t type); // specified in .cpp
		FullyConnectedLayer(size_t NOUT,  actfunc_t type, CNetLayer& lower);
		~FullyConnectedLayer();
		layer_t whoAmI() const;

		// forProp
		void forProp(MAT& in, bool training, bool recursive);

		// backprop
		void backPropDelta(MAT& delta, bool recursive);
		MAT grad(MAT& input);
		uint32_t getFeatures() const;

private:
	/* Weight normalization functions
	*/
	void updateW();
	void normalizeV();
	void inversVNorm();
	MAT gGrad(const MAT& grad); // gradient in g's
	MAT vGrad(const MAT& grad, MAT& ggrad); // gradient in V
	void initG();
	void initV();

	// initialization
	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif