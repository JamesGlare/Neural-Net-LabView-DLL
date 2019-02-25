#pragma once
#include "defininitions.h"
#include "PhysicalLayer.h"

#ifndef CNET_CONVOLAYER
#define CNET_CONVOLAYER
/* Convolutional layer
*	Supports multiple features and sidechannels (inputs thar are simply passed on).
*/

class ConvolutionalLayer : public PhysicalLayer{
	public:
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX, 
			 uint32_t outChannels, uint32_t inChannels, actfunc_t type);
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t strideY, uint32_t strideX,  
			 uint32_t outChannels, uint32_t inChannels, actfunc_t type, CNetLayer& lower);
		ConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride,  uint32_t outChannels, uint32_t inChannels, actfunc_t type);
		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride,  uint32_t outChannels, uint32_t inChannels, actfunc_t type, CNetLayer& lower);
		~ConvolutionalLayer();
		
		layer_t whoAmI() const;
		// propagation 
		// forProp
		void forProp(MAT& in, bool training, bool recursive);
		MAT w_grad(MAT& input);
		MAT b_grad();
		void backPropDelta(MAT& delta, bool recursive);

		inline size_t getNOUTX() const { return NOUTX; };
		inline size_t getNOUTY() const { return NOUTY; };
		inline size_t getNINX() const { return NINX; };
		inline size_t getNINY() const { return NINY; };
		inline size_t getKernelX() const { return kernelX; };
		inline size_t getKernelY() const { return kernelY; };
		uint32_t getOutChannels() const;
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

		/* Spectral Normalization
		*/
		void snorm_setW();
		void snorm_updateUVs();
		MAT snorm_dWt(MAT& grad);
		// Auxiliary feature-selection functions
		const MAT& getIthFeature(size_t i);
		
		// Geometry
		size_t NOUTX;
		size_t NOUTY;
		size_t NINX;
		size_t NINY;
		size_t kernelX;
		size_t kernelY;
		uint32_t strideX;
		uint32_t strideY;
		uint32_t inChannels;
		uint32_t outChannels;
		uint32_t features;
		void assertGeometry();

		// File functions
		void saveToFile(ostream& os) const;
		void loadFromFile(ifstream& in);

		void init();
};
#endif