#pragma once
#include "defininitions.h"
#include "DiscarnateLayer.h"

#ifndef CNET_RESHAPE
#define CNET_RESHAPE
/* Reshape Class
   Implements a simple transpose operation at the moment.
   But it's meant as a placeholder for the future, in case I need more difficult reshaping operations.
*/
class Reshape : public DiscarnateLayer {
public:
	Reshape(size_t NIN);
	Reshape(CNetLayer& lower);

	~Reshape();
	layer_t whoAmI() const;

	// propagation
	void forProp(MAT& in, bool saveActivation, bool recursive);
	void backPropDelta(MAT& delta, bool recursive); // recursive
	
private:
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);
};
#endif