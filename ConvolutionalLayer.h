#pragma once
#include "defininitions.h"
#include "CNetLayer.h"

#ifndef CONVOLAYER
#define CONVOLAYER

class ConvolutionalLayer : public CNetLayer{
	public:
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t stride, actfunc_t type);
		ConvolutionalLayer(size_t NOUTX, size_t NOUTY, size_t NINX, size_t NINY, size_t kernelX, size_t kernelY, uint32_t stride, actfunc_t type,CNetLayer& const lower);
		ConvolutionalLayer(size_t NOUTXY, size_t NINXY, size_t kernelXY, uint32_t stride, actfunc_t type);
		ConvolutionalLayer(size_t NOUTXY, size_t kernelXY, uint32_t stride, actfunc_t type, CNetLayer& const lower);
		~ConvolutionalLayer();
		
		layer_t whoAmI() const;


		// propagation 
		// forProp
		void forProp(MAT& in, bool saveActivation);
		
		MAT grad(MAT& const input);
		fREAL applyUpdate(learnPars pars, MAT& const input) ; // recursive
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