#pragma once
#include "defininitions.h"
#include "PhysicalLayer.h"

#ifndef CNET_CONVOLAYER
#define CNET_CONVOLAYER

class ConvolutionalLayer : public PhysicalLayer{
	public:
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, uint32_t features, actfunc_t type);
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, uint32_t features, actfunc_t type,CNetLayer& const lower);
		ConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride, uint32_t features, actfunc_t type);
		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, uint32_t features, actfunc_t type,  CNetLayer& const lower);
		~ConvolutionalLayer();
		
		layer_t whoAmI() const;
		// propagation 
		// forProp
		void forProp(MAT& in, bool training, bool recursive);
		MAT grad(MAT& const input);
		void backPropDelta(MAT& const delta, bool recursive);

		inline size_t getNOUTX() const { return NOUTX; };
		inline size_t getNOUTY() const { return NOUTY; };
		inline size_t getNINX() const { return NINX; };
		inline size_t getNINY() const { return NINY; };
		inline size_t getKernelX() const { return kernelX; };
		inline size_t getKernelY() const { return kernelY; };

	private:
		/* Weight normalization functions
		*/
		void updateW();
		void normalizeV();
		MAT inversVNorm();
		MAT gGrad(MAT& const grad);
		MAT vGrad(MAT& const grad, MAT& const ggrad);
		void initG();
		void initV();
		// Auxiliary feature-selection functions
		const MAT& getIthFeature(size_t i);
		// sizes
		size_t NOUTX;
		size_t NOUTY;
		size_t NINX;
		size_t NINY;
		size_t kernelX;
		size_t kernelY;
		size_t strideX;
		size_t strideY;
		size_t features;

		void saveToFile(ostream& os) const;
		void loadFromFile(ifstream& in);

		void assertGeometry();
		void init();
};
#endif