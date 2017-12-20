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
	MAT fourier(const MAT& );
	MAT conv( const MAT&, const MAT&);
	MAT antiConv(const MAT&, const MAT&);
	MAT getKernel() const;
	// Getter functions
	uint32_t get_NIN();
	uint32_t get_NOUT();
	uint32_t get_NNODES();

private:
	// private parameters
	uint32_t NINXY;
	uint32_t NOUTXY;
	uint32_t NNODES;

	int32_t kernelSize;
	int32_t stride;
	int32_t padding;

	MAT inFourier; // store fourier trafo of input
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
	static fREAL l2Error(const MAT&);
};

