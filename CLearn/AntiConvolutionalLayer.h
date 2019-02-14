#pragma once
#include "defininitions.h"
#include "PhysicalLayer.h"
#ifndef CNET_ANTICONVOLAYER
#define CNET_ANTICONVOLAYER
/* Deconvolution Layer
*	Supports multiple features (which are currently collapsed onto the single feature).
*	In addition, it supports side-channels.
*/
class AntiConvolutionalLayer : public PhysicalLayer {
public:
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, 
		uint32_t features, uint32_t outBoxes, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, 
		uint32_t features, uint32_t outBoxes, actfunc_t type, CNetLayer& lower);
	AntiConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride, uint32_t features, uint32_t outBoxes, actfunc_t type);
	AntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, uint32_t features, uint32_t outBoxes,  actfunc_t type, CNetLayer& lower);
	~AntiConvolutionalLayer();

	layer_t whoAmI() const;
	// forProp
	void forProp(MAT& in, bool training, bool recursive);
	MAT w_grad(MAT& input);
	MAT b_grad();
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
	/* Weight normalization functions
	*/ 
	void wnorm_setW();
	void wnorm_initV();
	void wnorm_initG();
	void wnorm_normalizeV();
	void wnorm_inversVNorm();
	MAT wnorm_gGrad(const MAT& grad); // gradient in g's
	MAT wnorm_vGrad(const MAT& grad, MAT& ggrad); // gradient in V
	
	/* Spectral Normalization
	*/
	void snorm_setW();
	void snorm_updateUVs();
	MAT snorm_dWt(MAT& grad);

	// Geometry
	size_t NOUTX;
	size_t NOUTY;
	size_t NINX;
	size_t NINY;
	size_t kernelX;
	size_t kernelY;
	size_t strideY;
	size_t strideX;
	size_t features;
	uint32_t outBoxes;
	void assertGeometry();

	// File function
	void saveToFile(ostream& os) const;
	void loadFromFile(ifstream& in);

	void init();
};
#endif