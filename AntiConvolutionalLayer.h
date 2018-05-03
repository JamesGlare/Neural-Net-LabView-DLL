#pragma once
#include "defininitions.h"
#include "PhysicalLayer.h"
#ifndef CNET_ANTICONVOLAYER
#define CNET_ANTICONVOLAYER

class AntiConvolutionalLayer : public PhysicalLayer {
public:
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t stride, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
	AntiConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
	~AntiConvolutionalLayer();

	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool saveActivation);
	MAT grad(MAT& const input);
	void backPropDelta(MAT& const delta);

	inline size_t getNOUTX() const { return NOUTX; };
	inline size_t getNOUTY() const { return NOUTY; };
	inline size_t getNINX() const { return NINX; };
	inline size_t getNINY() const { return NINY; };
	inline size_t getKernelX() const { return kernelX; };
	inline size_t getKernelY() const { return kernelY; };


private:
	// backprop
	size_t NOUTX;
	size_t NOUTY;
	size_t NINX;
	size_t NINY;
	size_t kernelX;
	size_t kernelY;
	uint32_t stride;
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);

	void assertGeometry();
	void init();
};
#endif