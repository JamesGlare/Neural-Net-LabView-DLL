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
		MAT w_grad(MAT& input);
		MAT b_grad();

		// Initialization Routine
		void constrainToMax(MAT& mues, MAT& sigma);

private:
	/* Weight normalization functions
	*/
	void wnorm_setW(); // to W
	void wnorm_initV();
	void wnorm_initG();
	void wnorm_normalizeV();
	void wnorm_inversVNorm();
	MAT wnorm_gGrad(const MAT& grad); // gradient in g's
	MAT wnorm_vGrad(const MAT& grad, MAT& ggrad); // gradient in V
	/* spectral normalization
	*/
	void snorm_setW();
	void snorm_updateUVs();
	MAT snorm_dWt(MAT& grad); // neecds to be multiplied element-wise to 
	// initialization
	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif