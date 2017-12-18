#pragma once
#include "defininitions.h"

#ifdef CHOLONET_EXPORTS  
#define CHOLONET_API__declspec(dllexport)   
#else  
#define CHOLONET_API__declspec(dllimport)   
#endif  

class CHoloNet
{
public:
	CHoloNet(uint32_t, uint32_t, uint32_t);
	~CHoloNet();
	// Training functions
	fREAL backProp(const MAT&, MAT&, learnPars); // returns cum error and overwrites the second argument with the prediction
	MAT forProp(const MAT&, bool);
	fREAL l2Error(const MAT&);

	// Getter functions
	uint32_t get_NIN();
	uint32_t get_NOUT();
	uint32_t get_NNODE(uint32_t);


private:
	// private parameters
	uint32_t NINXY;

	uint32_t NOUTXY;

	uint32_t NNODES;

	uint32_t kernelSize;
	uint32_t stride;
	uint32_t padding;

	MAT inLayer; // weights for fourier layer
	MAT inAct;
	MAT inDelta; 
	MAT hiddenLayer1; // all to all layer
	MAT hiddenAct1;
	MAT hiddenDelta1;
	MAT hiddenLayer2;
	MAT hiddenAct2;
	MAT hiddenDelta2;
	MAT kernel; // kernel for convolution of hiddenLayer

	// activation functions
	static MAT ACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer
	static MAT DACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer
	// convolution functions
	MAT conv(const MAT& in);
	MAT fourier(const MAT& in);
};

