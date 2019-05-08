#pragma once
#include "defininitions.h"
#include "DiscarnateLayer.h"


#ifndef CNET_GAUSSIANREPARAMETRIZATIONLAYER
#define CNET_GAUSSIANREPARAMETRIZATIONLAYER

class GaussianReparametrizationLayer : public DiscarnateLayer {

public:
	GaussianReparametrizationLayer(size_t NOUT);
	GaussianReparametrizationLayer(CNetLayer& lower);

	~GaussianReparametrizationLayer();
	layer_t whoAmI() const;

	// propagation
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive); // recursive

private:
	void init();
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
	MAT eps;
	MAT ones;
};

#endif