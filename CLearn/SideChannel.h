#pragma once
#include "defininitions.h"
#include "DiscarnateLayer.h"

#ifndef CNET_SIDECHANNEL
#define CNET_SIDECHANNEL
/* SIDECHANNEL Class
This allows us to add some input to layers other than the input layer.
For instance, if one wishes to sneak some additional information past a convolutional layer
it's best done using this sidechannel class
*/
class SideChannel : public DiscarnateLayer {
public:
	SideChannel(CNetLayer& lower, size_t _sidechannelSize);

	~SideChannel();
	layer_t whoAmI() const;
	void preFeed(const MAT& sideChannelMatrix);

	// propagation
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive); // recursive
	inline size_t getSidechannelSize() const { return sideChannelSize; };

private:
	size_t sideChannelSize;
	MAT sideChannelMatrix; // here we store the sidechannel INPUTS that get fed into the chain
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif