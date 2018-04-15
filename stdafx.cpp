// stdafx.cpp : source file that includes just the standard includes
// CLearn.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"
#include <memory>
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "MaxPoolLayer.h"
#include "CLearn.h"
#include "HoloNet.h"
// TODO: reference any additional headers you need in STDAFX.H
// and not in this file
typedef std::shared_ptr<CNet> CNETPTR;

__declspec(dllexport) void __stdcall initializeCNet(CNet** ptr, uint32_t NIN){
	*ptr = new CNet(NIN);
}

__declspec(dllexport) uint32_t __stdcall addFullyConnectedLayer(CNet* ptr, uint32_t NOUT) {
	return ptr->addFullyConnectedLayer(NOUT, actfunc_t::RELU);
}
__declspec(dllexport) uint32_t __stdcall addConvolutionalLayer(CNet* ptr, uint32_t NOUTXY, uint32_t kernelXY, uint32_t stride) {
	return ptr->addConvolutionalLayer(NOUTXY, kernelXY, stride, actfunc_t::RELU);
}
__declspec(dllexport) uint32_t __stdcall addMaxPoolLayer(CNet* ptr, uint32_t maxOverXY) {
	return ptr->addPoolingLayer(maxOverXY, pooling_t::max);
}

__declspec(dllexport) fREAL __stdcall forwardCNet(CNet* ptr, fREAL* const input, fREAL* const output) {
	MAT inputMatrix = MATMAP(input, ptr->getNIN(), 1); // (NIN, 1) Matrix
	MAT outputDesiredMatrix = MATMAP(output, ptr->getNOUT(), 1); // (NIN, 1) Matrix

	fREAL error = ptr->forProp(inputMatrix, outputDesiredMatrix);

	copyToOut(inputMatrix.data(), output, ptr->getNOUT());
	return error;
}
__declspec(dllexport) fREAL __stdcall backPropCNet(CNet* ptr, fREAL* const input, fREAL* const outputDesired, fREAL* const eta, 
	fREAL* const metaEta, fREAL* const gamma, fREAL* const lambda, uint32_t* const conjugate ) {
	
	learnPars pars = { *eta, *metaEta, *gamma, *lambda, *conjugate };

	MAT inputMatrix = MATMAP(input, ptr->getNIN(), 1); // (NIN, 1) Matrix
	MAT outputDesiredMatrix = MATMAP(outputDesired, ptr->getNOUT(), 1);
	fREAL error = ptr->backProp(inputMatrix, outputDesiredMatrix, pars);
	copyToOut(outputDesiredMatrix.data(), outputDesired, ptr->getNOUT());
	return error;
}
__declspec(dllexport) void __stdcall resetConjugate(CNet* ptr, fREAL* const input) {
	MAT inputMatrix = MATMAP(input, ptr->getNIN(), 1); // (NIN, 1) Matrix
	ptr->resetConjugate(inputMatrix);
}

__declspec(dllexport) void __stdcall saveCNet(CNet* ptr) {
	string filePath = "C:\\Jannes\\learnSamples\\";
	ptr->saveToFile(filePath);
}
__declspec(dllexport) void __stdcall destroyCNet(CNet* ptr) {
	ptr->~CNet();
}
__declspec(dllexport) void __stdcall writeLayer(CNet* ptr, uint32_t layer, fREAL* const toCopyTo) {
	if(layer < ptr->getLayerNumber())
		ptr->copyNthLayer(layer, toCopyTo);
}
__declspec(dllexport) uint32_t __stdcall test() {
	FullyConnectedLayer fc(16, 16, actfunc_t::RELU,0,1);
	MaxPoolLayer mpl(2, fc);
	return mpl.getNOUT();
}
/* HOLONET Stuff **************************************************************************************************************
*/

__declspec(dllexport) int __stdcall initCHoloNet(CHoloNet** ptr, uint32_t NINXY, uint32_t NOUTXY, uint32_t NNODES) {
	*ptr = new CHoloNet(NINXY, NOUTXY, NNODES);
	return 1;
}

__declspec(dllexport) void __stdcall testFourier(CHoloNet* ptr, fREAL* const img) {
	MAT in = MATMAP(img, ptr->get_NIN(), ptr->get_NIN()); // (NIN, 1) Matrix
	MAT out = ptr->fourier(in);
	in.setConstant(out.size());
	copyToOut(in.data(), img, in.size());
	copyToOut(out.data(), img, out.size());
}

__declspec(dllexport) void __stdcall testConvolution(uint32_t NINX, uint32_t kernelX, fREAL* const img, fREAL* const kernelIN) {
	MAT in = MATMAP(img, NINX, NINX); // (NIN, 1) Matrix
	MAT kernel = MATMAP(kernelIN, kernelX, kernelX);

	MAT out(antiConvoSize(NINX,kernelX,0,1), antiConvoSize(NINX, kernelX, 0, 1));
	out = antiConv(in, kernel, 1,0,0);
	copyToOut(out.data(), img, out.size());
}

/* test the conv layer backprop
* it should converge to the identity function.
*/
__declspec(dllexport) fREAL __stdcall testConvLayerBackProp(CHoloNet* ptr, fREAL* const img, fREAL eta) {
	MAT in = MATMAP(img, ptr->get_NIN(), ptr->get_NIN()); // (NINXY, NINXY) Matrix
	MAT out(in.rows(), in.cols());
	MAT kernel = ptr->getKernel();
	size_t kernelSize = kernel.rows();
	out = ptr->conv(in, kernel, 1, padSizeForEqualConv(in.rows(), kernelSize, 1)); // (NOUTXY, NOUTXY)
																				   // NINXY == NOUTXY
	MAT delta = out - 2 * in;

	MAT grad = ptr->conv(delta, in, 1, (kernelSize - 1) / 2).reverse(); // .reverse()
	ptr->getKernel() = ptr->getKernel().reverse() - eta*grad;
	copyToOut(out.data(), img, out.size());
	return delta.cwiseProduct(delta).sum();
}

__declspec(dllexport) void getKernel(CHoloNet* ptr, fREAL* const kernel) {
	copyToOut(ptr->getKernel().data(), kernel, ptr->getKernel().size());
}

__declspec(dllexport) void __stdcall testForward(CHoloNet* ptr, fREAL* const img) {
	MAT in = MATMAP(img, ptr->get_NIN(), ptr->get_NIN()); // (NIN, 1) Matrix
	MAT out = ptr->forProp(in, false);
	copyToOut(out.data(), img, out.size());
}
__declspec(dllexport) fREAL __stdcall train(CHoloNet* ptr, fREAL* const in, fREAL* const dOut, fREAL* const eta, int32_t forwardOnly) {
	MAT inMat = MATMAP(in, ptr->get_NIN(), ptr->get_NIN()); // (NINXY, NINXY) Matrix
	MAT dOutMat = MATMAP(dOut, ptr->get_NOUT(), ptr->get_NOUT()); // (NOUTXY, NOUTXY)
	learnPars pars = { *eta, 0,0,0, false };
	fREAL error = 0;
	if (1 == forwardOnly) {
		dOutMat = ptr->forProp(inMat, false);
	}
	else {
		error = ptr->backProp(inMat, dOutMat, pars); // doutMat contains prediction
	}
	copyToOut(dOutMat.data(), dOut, dOutMat.size());

	return error;
}
/* CLearn Stuff ***************************************************************************************************************
*/

__declspec(dllexport) int __stdcall saveNetwork(CLearn* ptr) {
	return ptr->saveToFile();
}
__declspec(dllexport) int __stdcall loadNetwork(CLearn* ptr) {
	return ptr->initializeFromFiles();
}


__declspec(dllexport) int __stdcall initClass(CLearn** ptr, uint32_t NIN, uint32_t NOUT, uint32_t NHIDDEN, uint32_t* const NNODES) {
	*ptr = new CLearn(NIN, NOUT, NHIDDEN, NNODES);
	return 1;
}

__declspec(dllexport) int __stdcall callClass(CLearn* ptr, uint8_t* const SLM, uint8_t* const out, int32_t kx, int32_t ky, double val) {
	int sx = 100;
	int sy = 100;
	for (int i = 0; i < sx; i++) {
		for (int j = 0; j < sy; j++) {
			SLM[i*sy + j] = 127 * val*(sin((double)(ky*i) / sy)*sin((double)(kx*j) / sx) + 1);
		}
	}
	return 1;
}
__declspec(dllexport) fREAL __stdcall forward(CLearn* ptr, fREAL* const SLM, fREAL* const image, fREAL* const eta, fREAL* const metaEta, fREAL* const gamma, fREAL* const lambda, uint32_t* const nesterov, int* const validate) {
	// remember not to use .size() on a matrix. It's not numpy ;)
	learnPars pars = { *eta, *metaEta, *gamma, *lambda, *nesterov };

	MAT in = MATMAP(SLM, ptr->get_NIN(), 1); // (NIN, 1) Matrix
	MAT dOut = MATMAP(image, ptr->get_NOUT(), 1);
	fREAL error = 0.0;
	if (*validate == 0) {
		error = ptr->backProp(in, dOut, pars); // overwrites dOut with prediction
	}
	else {
		dOut = ptr->forProp(in, false);
	}
	copyToOut(dOut.data(), image, ptr->get_NOUT()); // copy data into pointer
													//MAT SLMSub = MATMAP(SLM, ptr->get_NIN(), 1);
													//SLMSub.setConstant(128);
													//copyToOut(SLMSub.data(), SLM, ptr->get_NIN()); // copy data into pointer
	return error;
}

__declspec(dllexport) void __stdcall getINWeights(CLearn* ptr, fREAL* const inWeights) {
	ptr->copy_inWeights(inWeights);
}
__declspec(dllexport) void __stdcall getHiddenWeights(CLearn* ptr, fREAL* const hiddenWeights, uint32_t layer) {
	ptr->copy_hiddenWeights(hiddenWeights, layer);
}
__declspec(dllexport) void __stdcall getOutWeights(CLearn* ptr, fREAL* const outWeights) {
	ptr->copy_outWeights(outWeights);
}
__declspec(dllexport) int __stdcall terminateClass(CLearn* ptr) {
	delete (ptr);
	return 1;
}
