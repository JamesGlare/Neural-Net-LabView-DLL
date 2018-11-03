#pragma once
#include "defininitions.h"
#include "PhysicalLayer.h"
#ifndef CNET_ANTICONVOLAYER
#define CNET_ANTICONVOLAYER

class AntiConvolutionalLayer : public PhysicalLayer {
public:
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, uint32_t features, uint32_t sideChannels, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, uint32_t features, uint32_t sideChannels, actfunc_t type, CNetLayer& lower);
	AntiConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride, uint32_t features, uint32_t sideChannels, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, uint32_t features, uint32_t sideChannels, actfunc_t type, CNetLayer& lower);
	~AntiConvolutionalLayer();

	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool training, bool recursive);
	MAT grad(MAT& input);
	void backPropDelta(MAT& delta, bool recursive);
	// Getter Function
	inline size_t getNOUTX() const { return NOUTX; };
	inline size_t getNOUTY() const { return NOUTY; };
	inline size_t getNINX() const { return NINX; };
	inline size_t getNINY() const { return NINY; };
	inline size_t getKernelX() const { return kernelX; };
	inline size_t getKernelY() const { return kernelY; };
	uint32_t getFeatures() const ;

private:
	// Weight normalization functions
	void updateW();
	void normalizeV();
	void inversVNorm();
	MAT gGrad(const MAT& grad);
	MAT vGrad(const MAT& grad, MAT& ggrad);
	void initG();
	void initV();

	// Geometry
	size_t NOUTX;
	size_t NOUTY;
	size_t NINX;
	size_t NINY;
	size_t kernelX;
	size_t kernelY;
	size_t strideY;
	size_t strideX;
	size_t inFeatures;
	uint32_t features;
	uint32_t sideChannels;
	void assertGeometry();
	MAT sideChannelBuffer; // buffer for sidechannel inputs

	// File function
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);

	void init();
};
#endif