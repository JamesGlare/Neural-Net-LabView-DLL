// stdafx.cpp : source file that includes just the standard includes
// CLearn.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"
#include <memory>
#include "CNet.h"
#include "FullyConnectedLayer.h"
#include "MaxPoolLayer.h"
#include "PassOnLayer.h"

/* CNet LIBRARY FUNCTIONS 
*/

bool sameCNet(CNet* ptr) {
	// save ptr
	static CNet* ptrCache = NULL;

	if (ptr != ptrCache) {
		ptrCache = ptr;
		return false;
	} else {
		return true;
	}
	
}
typedef std::shared_ptr<CNet> CNETPTR;
__declspec(dllexport) void __stdcall initializeCNet(CNet** ptr, uint32_t NIN){
	*ptr = new CNet(NIN);
}

__declspec(dllexport) void __stdcall addFullyConnectedLayer(CNet* ptr, uint32_t NOUT, uint32_t func) {
	ptr->addFullyConnectedLayer(NOUT, static_cast<actfunc_t>(func));
}
__declspec(dllexport) void __stdcall addConvolutionalLayer(CNet* ptr, uint32_t NOUTXY, uint32_t kernelXY, uint32_t stride, uint32_t inChannels, uint32_t outChannels, uint32_t func) {
	ptr->addConvolutionalLayer(NOUTXY, kernelXY, stride, inChannels, outChannels, static_cast<actfunc_t>(func));
}
__declspec(dllexport) void __stdcall addAntiConvolutionalLayer(CNet* ptr, uint32_t NOUTXY, uint32_t kernelXY, uint32_t stride, uint32_t inChannels, uint32_t outChannels, uint32_t func) {
	ptr->addAntiConvolutionalLayer(NOUTXY, kernelXY, stride, inChannels, outChannels, static_cast<actfunc_t>(func));
}
__declspec(dllexport) void __stdcall addMaxPoolLayer(CNet* ptr, uint32_t maxOverXY, uint32_t channels) {
	ptr->addPoolingLayer(maxOverXY, channels, pooling_t::max);
}
__declspec(dllexport) void __stdcall addPassOnLayer(CNet* ptr, uint32_t function) {
	ptr->addPassOnLayer(static_cast<actfunc_t>(function));
} 
__declspec(dllexport) void __stdcall addReshapeLayer(CNet* ptr) {
	ptr->addReshape();
}
__declspec(dllexport) void __stdcall addSideChannel(CNet* ptr, uint32_t sideChannelSize) {
	ptr->addSideChannel(sideChannelSize);
}
__declspec(dllexport) void __stdcall addDropoutLayer(CNet* ptr, fREAL ratio) {
	 ptr->addDropoutLayer(ratio);
}
__declspec(dllexport) fREAL __stdcall forwardCNet(CNet* ptr, fREAL* const input, fREAL* const output, int32_t* const inFormat, int32_t* const outFormat) {
	// if change of CNet instance, relink the chain

	if (!sameCNet(ptr)) {
		ptr->linkChain();
	}


	learnPars pars;
	assert(ptr->getNOUT() == outFormat[0]*outFormat[1]);
	assert(ptr->getNIN() == inFormat[0]*inFormat[1]);

	MAT inputMatrix = MATMAP_ROWMAJOR(input,  inFormat[0], inFormat[1]); 
	MAT outputDesiredMatrix = MATMAP_ROWMAJOR(output, outFormat[0], outFormat[1]); 
	outputDesiredMatrix.resize(ptr->getNOUT(), 1); // (NIN, 1) Matrix
	inputMatrix.resize(ptr->getNIN(), 1); // (NIN, 1) Matrix
	
	fREAL error = ptr->forProp(inputMatrix, outputDesiredMatrix,  false, pars);
	inputMatrix.resize(outFormat[0], outFormat[1]);
	inputMatrix.transposeInPlace(); // Go back to Row-major format
	inputMatrix.resize(ptr->getNOUT(), 1);
	copyToOut(inputMatrix.data(), output, ptr->getNOUT());
	return error;
}
/* BACK
*/
__declspec(dllexport) fREAL __stdcall backPropCNet(CNet* ptr, fREAL* const input, fREAL* const output, fREAL* const eta, 
	fREAL* const clip, fREAL* const gamma, fREAL* const lambda, uint32_t* const rmsprop, uint32_t* const adam ,uint32_t* const batch_update,
	uint32_t* const weight_norm, uint32_t* const spectral_norm, uint32_t* const firstTrain, uint32_t* const lastTrain, int32_t* const inFormat, 
	int32_t* const outFormat, uint32_t* const deltaProvided) {
	
	// if change of CNet instance, relink the chain
	if (! sameCNet(ptr)) {
		ptr->linkChain();
	}
	bool deltaProvided_bool = false;
	if (*deltaProvided == 1) {
		deltaProvided_bool = true;
	}

	learnPars pars( *eta, *clip, *gamma, *lambda, *rmsprop, *adam, *batch_update,  *weight_norm, *spectral_norm, *firstTrain, *lastTrain, true);

	assert(ptr->getNOUT() == outFormat[0] * outFormat[1]);
	assert(ptr->getNIN() == inFormat[0] * inFormat[1]);

	MAT inputMatrix = MATMAP_ROWMAJOR(input, inFormat[0], inFormat[1]);
	inputMatrix.resize(ptr->getNIN(), 1);// (NIN, 1) Matrix
	MAT outputDesiredMatrix = MATMAP_ROWMAJOR(output, outFormat[0], outFormat[1]); 
	outputDesiredMatrix.resize(ptr->getNOUT(), 1);// (NOUT,1) Matrix
	fREAL error = ptr->backProp(inputMatrix, outputDesiredMatrix, pars, deltaProvided_bool);

	// Resize the output matrix & copy into outgoing array.
	outputDesiredMatrix.resize(outFormat[0], outFormat[1]);
	outputDesiredMatrix.transposeInPlace(); // Go back to Row-major format
	outputDesiredMatrix.resize(ptr->getNOUT(), 1);
	copyToOut(outputDesiredMatrix.data(), output, ptr->getNOUT());
	
	return error;
}

__declspec(dllexport) void __stdcall trainConGan(CNet* ptr_D, CNet* ptr_G, fREAL* const X, fREAL* const Y, 
	uint32_t prepGen, uint32_t handOverLayer, fREAL* D_REAL_ERR, fREAL* D_FAKE_ERR, fREAL eta_D, fREAL eta_G, fREAL clip, fREAL gamma, fREAL lambda, uint32_t rmsprop, uint32_t adam, uint32_t batch_update,
	uint32_t weight_norm, uint32_t spectral_norm, uint32_t firstTrain, uint32_t lastTrain, int32_t* const xFormat, int32_t* const yFormat) {

	// Major GAN training function
	learnPars pars(eta_D, clip, gamma, lambda, rmsprop, adam, batch_update, weight_norm, spectral_norm, firstTrain, lastTrain, true);
	assert(ptr_D->getNIN() == yFormat[0] * yFormat[1]); 
	assert(ptr_G->getNOUT() == xFormat[0] * xFormat[1]);
	assert(ptr_D->getSideChannelSize() == ptr_G->getNOUT());
	// Format the matrices
	MAT x_matrix = MATMAP_ROWMAJOR(X, xFormat[0], xFormat[1]);
	x_matrix.resize(ptr_G->getNOUT(), 1);// (NIN, 1) Matrix
	MAT y_matrix = MATMAP_ROWMAJOR(Y, yFormat[0], yFormat[1]);
	y_matrix.resize(ptr_D->getNIN(), 1);// (NOUT,1) Matrix

	// (0) change a few settings for the discriminator
	pars.weight_normalization = false; // only used for generator
	bool batchIsDue = pars.batch_update == 0;

	// (1) Train D real =====================================================================
	MAT D_REAL_Y(y_matrix); // Y copy - expensive
	MAT D_REAL_RES(1, 1);
	D_REAL_RES.setZero();
	
	if (batchIsDue) // hold back the update to the disc weights - need fake loss
		pars.batch_update = 1;

	ptr_D->preFeedSideChannel(x_matrix);
	ptr_D->train_GAN_D(D_REAL_Y, y_matrix, D_REAL_RES, true, pars);

	*D_REAL_ERR = Sig(D_REAL_RES(0, 0));

	//(2) Draw Z and produce generator sample ===============================================
	MAT Z(ptr_G->getSideChannelSize(), 1);
	Z.setRandom().unaryExpr(&abs<fREAL>);
	ptr_G->preFeedSideChannel(Z);
	MAT G_Sample(y_matrix);
	ptr_G->forProp(G_Sample, x_matrix, false, pars); // G_Sample contains the generator sample

	// (3) Train D fake =====================================================================
	MAT D_FAKE_Y(y_matrix); // Y copy - expensive
	MAT D_FAKE_RES(1, 1);
	D_FAKE_RES.setZero();

	if (batchIsDue) // undo the change to batch_update 
		pars.batch_update = 0;

	ptr_D->preFeedSideChannel(G_Sample);
	ptr_D->train_GAN_D(D_FAKE_Y, y_matrix, D_FAKE_RES, false, pars);

	*D_FAKE_ERR = Sig(D_FAKE_RES(0, 0));

	// (4) Prepare and Train the Generator ==================================================
	if (prepGen) {
		Z.setRandom().unaryExpr(&abs<fREAL>);
		ptr_G->preFeedSideChannel(Z);
		G_Sample = y_matrix;
		ptr_G->forProp(G_Sample, x_matrix, true, pars); // New Gen sample, SAVE = true!
		ptr_D->preFeedSideChannel(G_Sample); // feed the new generator output to the discriminator

		MAT D_G_Y(y_matrix);
		MAT D_G_RES(1, 1);
		D_G_RES.setZero();
		ptr_D->train_GAN_G_D(D_G_Y, D_G_RES, pars); // forprop through new weights of D
													// backprop as well so that we can collect deltas
		MAT deltas(ptr_G->getNOUT(), 1);
		ptr_D->copyNthDelta(handOverLayer, deltas.data(), ptr_G->getNOUT()); // collect deltas
		
		// Change a few settings for the generator
		pars.spectral_normalization = false;
		pars.weight_normalization = weight_norm;
		pars.eta = eta_G;

		ptr_G->backProp_GAN_G(y_matrix, deltas, pars); // backprop those deltas through G
	}
	// (5) Copy the G_sample into the X-Array ================================================

	// Resize the output matrix & copy into outgoing array.
	G_Sample.resize(xFormat[0], xFormat[1]);
	G_Sample.transposeInPlace(); // Go back to Row-major format
	G_Sample.resize(ptr_G->getNOUT(), 1);
	copyToOut(G_Sample.data(), X, ptr_G->getNOUT());
}

__declspec(dllexport) void __stdcall feedSideChannel(CNet* ptr, fREAL* const sideChannelArray, int32_t* const format) {

	MAT sideChannelMatrix = MATMAP_ROWMAJOR(sideChannelArray, format[0], format[1]);
	sideChannelMatrix.resize(format[0] * format[1], 1);
	ptr->preFeedSideChannel(sideChannelMatrix);

}

__declspec(dllexport) void __stdcall addMixtureDensity(CNet* ptr, size_t NOUT, size_t features, size_t BlockXY) {
	ptr->addMixtureDensity( NOUT,  features,  BlockXY);
}
__declspec(dllexport) void __stdcall debugMsg(CNet* ptr, fREAL* msg) {
	ptr->debugMsg(msg);
}
__declspec(dllexport) uint32_t __stdcall initializeNetwork(CNet* ptr) {
	if (!sameCNet(ptr)) {
		ptr->linkChain();
	}
	return ptr->getLayerNumber();
}
__declspec(dllexport) void __stdcall saveCNet(CNet* ptr, char* filePath) {
	if (!sameCNet(ptr)) {
		ptr->linkChain();
	}

	ptr->saveToFile(string(filePath));
}
__declspec(dllexport) void __stdcall loadCNet(CNet* ptr, char* filePath) {

	ptr->loadFromFile(string(filePath));

	sameCNet(ptr); // store ptr 
	ptr->linkChain(); // relink chain regardless
	
}
__declspec(dllexport) void __stdcall loadCNet_layer(CNet* ptr, uint32_t layer, char* filePath) {
	
	ptr->loadFromFile_layer(string(filePath), layer);

	sameCNet(ptr); // store ptr
	ptr->linkChain();// relink chain regardless

}

__declspec(dllexport) void __stdcall destroyCNet(CNet* ptr) {
	ptr->~CNet();
}
// share-Layer functionality which enables dynamical switching
__declspec(dllexport) void __stdcall shareLayer(CNet* ptr, CNet* ptrOther, uint32_t firstLayer, uint32_t lastLayer) {
	ptr->shareLayers(ptrOther, firstLayer, lastLayer);
}
__declspec(dllexport) void __stdcall writeLayer(CNet* ptr, uint32_t layer, fREAL* const toCopyTo, int32_t* toCopyToFormat) {
	ptr->copyNthLayer(layer, toCopyTo);
}
__declspec(dllexport) void __stdcall getActivation(CNet* ptr, uint32_t layer, fREAL* const toCopyTo, int32_t* toCopyToFormat) {
	ptr->copyNthActivation(layer, toCopyTo);
}
__declspec(dllexport) void __stdcall getDelta(CNet* ptr, uint32_t layer, fREAL* const toCopyTo, int32_t* const toCopyToFormat) {
	//MAT test(1, 1);
	//test.setZero();
	//copyToOut(test.data(), toCopyTo, 1);
	ptr->copyNthDelta(layer, toCopyTo, (toCopyToFormat[0]* toCopyToFormat[1]));
}
__declspec(dllexport) void __stdcall getWeight(CNet* ptr, uint32_t layer, fREAL* const toCopyTo, int32_t* const toCopyToFormat) {
	//MAT test(1, 1);
	//test.setZero();
	//copyToOut(test.data(), toCopyTo, 1);
	ptr->copyNthLayer(layer, toCopyTo);
}
__declspec(dllexport) void __stdcall getLayerDimension(CNet* ptr, uint32_t layer, uint32_t* rows, uint32_t* cols) {
	size_t rows_ = 0;
	size_t cols_ = 0;
	ptr->inquireDimensions(layer, rows_, cols_);
	*rows = rows_;
	*cols = cols_;
}
__declspec(dllexport) void __stdcall setLayer(CNet* ptr, uint32_t layer, fREAL* const copyFrom, int32_t* const format) {
	MAT newLayer = MATMAP_ROWMAJOR(copyFrom, format[0], format[1]); // is newLayer now a row major matrix? or is the input just mapped in a row-major way?
	ptr->setNthLayer(layer, newLayer);
}
__declspec(dllexport) uint32_t __stdcall test() {
	return 0;
}
/* DEPRECATED HOLONET Stuff **************************************************************************************************************


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
*/
/* CLearn Stuff ***************************************************************************************************************


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
__declspec(dllexport) fREAL __stdcall forward(CLearn* ptr, fREAL* const SLM, fREAL* const image, fREAL* const eta, fREAL* const clip, fREAL* const gamma, fREAL* const lambda, uint32_t* const nesterov, int* const validate) {
	// remember not to use .size() on a matrix. It's not numpy ;)
	learnPars pars = { *eta, *clip, *gamma, *lambda, *nesterov };

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
*/