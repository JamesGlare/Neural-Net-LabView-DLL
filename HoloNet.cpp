#include "stdafx.h"
#include "HoloNet.h"

CHoloNet::CHoloNet(uint32_t _NINXY, uint32_t _NOUTXY, uint32_t _NNODES) : NINXY(_NINXY), NOUTXY(_NOUTXY),NNODES(_NNODES){

	// (1) construct inLayer

	inFourier = MAT(NINXY*NINXY, 1);//MAT(NINXY*(NINXY+1)/2,1);
	inFourier.setConstant(0);
	kernelSize = 5;
	stride = 0;
	padding = 0;

	// (2) Setup hidden layer
	hiddenLayer1 = MAT(NNODES, inFourier.rows() + 1);
	hiddenLayer1.setRandom();
	hiddenDelta1 = MAT(hiddenLayer1.rows(), 1);
	hiddenAct1 = MAT(hiddenLayer1.rows(), 1);
	hiddenDelta1.setConstant(0);
	hiddenLayer2 = MAT(NOUTXY*NOUTXY, NNODES + 1); // convolution with the output of this matrix should yield (NOUTX * NOUTY)
	hiddenLayer2.setRandom();
	hiddenDelta2 = MAT(hiddenLayer2.rows(), 1);
	hiddenAct2 = MAT(hiddenLayer2.rows(), 1);
	// (3) Setup kernel matrix
	kernel = MAT(kernelSize, kernelSize);
	kernel.setRandom();
	kernel *= 1 / kernel.sum();
}

CHoloNet::~CHoloNet()
{
}
fREAL CHoloNet::l2Error(const MAT& diff) {
	return 0.5*cumSum(matNorm(diff));
}

// Getter functions
uint32_t CHoloNet::get_NIN() {
	return NINXY;
}
uint32_t CHoloNet::get_NOUT() {
	return NOUTXY;
}
uint32_t CHoloNet::get_NNODES() {
	return NNODES;
}


MAT CHoloNet::ACT(const MAT& in) {
	// be careful with overloads here - it needs to be clear which function to use
	// only call other static functions.
	return in.unaryExpr(&ReLu);
}
MAT CHoloNet::DACT(const MAT& in) {
	return in.unaryExpr(&DReLu);
}
MAT CHoloNet::forProp(const MAT& in, bool saveActivations) {
	MAT temp = in;// fourier(in);
	temp.resize(NINXY*NINXY, 1);
	if (saveActivations)
		inFourier = temp;

	appendOne(temp);
	temp = hiddenLayer1*temp; // (NNODES,1)
	if (saveActivations)
		hiddenAct1 = temp;

	temp = ACT(temp);
	appendOne(temp);
	temp = hiddenLayer2*temp; // (NOUTXY*NOUTXY,1)
	if (saveActivations)
		hiddenAct2 = temp;

	temp = ACT(temp); // (NOUTXY*NOUTXY,1)
	temp.resize(NOUTXY, NOUTXY); // RESIZE - this should be removed, unncessary cost
	return conv(temp, kernel,1, padSizeForEqualConv(temp.rows(), kernelSize, 1)); // (NOUTXY, NOUTXY)
}

fREAL CHoloNet::backProp(const MAT& in, MAT& dOut, learnPars pars) {
	
	// (1) save activations
	MAT out = forProp(in, true);
	// (2) Calculate deltas
	MAT outDelta = out - dOut; // (NOUTXY, NOUTXY)
	fREAL error = l2Error(outDelta); // save error
	
	// (2.1) Kernel layer - Calculate deltas for NOUTXYxNOUTXY matrix, then later, sum down to kernel matrix with identity matrix
	/*
	MAT grad = ptr->conv(delta, in, 1, (kernelSize - 1) / 2).reverse(); // .reverse()
	ptr->getKernel() = ptr->getKernel().reverse() - eta*grad;
	*/
	MAT kernelDelta = conv( outDelta, kernel.reverse(), 1, padSizeForEqualConv(NOUTXY, kernelSize, 1)); // ()
	kernelDelta.resize(NOUTXY*NOUTXY, 1); // RESIZE
											   
	// (2.2) Hidden layer 2 - (NOUTXY^2,1)
	// Activation has (NOUTXY^2,1) shape
	// (NOUTXY^2,1) = DACT(NOUTXY^2,1) * (NOUTXY^2,1)
	hiddenDelta2 = DACT(hiddenAct2).cwiseProduct(kernelDelta);
	// (2.3) Hidden layer 1 -  
	// (NNODES,1) = DACT(NNODES,1) * (NOUTXY^2, NNODES).T x (NOUTXY^2,1) = (NNODES,1)
	hiddenDelta1 = DACT(hiddenAct1).cwiseProduct(hiddenLayer2.leftCols(hiddenLayer2.cols() - 1).transpose()*hiddenDelta2);

	// (3) Apply gradient
	//(3.1) kernelLayer
	// (kS, kS)
	MAT temp = ACT(hiddenAct2);
	temp.resize(NOUTXY, NOUTXY);
	// update the kernel
	// convolve 
	kernel = kernel.reverse() - pars.eta*conv(outDelta, temp, 1, (kernelSize - 1) / 2).reverse();
	// (3.2) hidden layer 2
	temp = appendOneInline(ACT(hiddenAct1));
	// (NOUTXY^2, NNODES+1 ) = (NOUTXY^2, 1) x ( 1, NNODES+1)
	hiddenLayer2 = hiddenLayer2 - pars.eta*hiddenDelta2*temp.transpose();
	// (3.3) hidden layer 1
	temp = appendOneInline(inFourier);
	// (NNODES, NIN*(NIN+1)/2+1) = (NNODES,1) x (1, NIN*(NIN+1)/2 + 1)
	hiddenLayer1 = hiddenLayer1 - pars.eta *hiddenDelta1*temp.transpose();

	dOut = out;
	return error;
}

MAT CHoloNet::fourier(const MAT& in) {
	int32_t L = NINXY; // == in.cols() 
	MAT Z = MAT(L*(L + 1) / 2,1); // number of unique elements - should match inLayer's dimensionality 
	Z.setConstant(0);
	uint32_t k = 0;
	fREAL arg = 0;
	
	for (int32_t j = 0; j < L; j++) {
		for (int32_t i = 0; i < L; i++) {
			k = 0;
			for (int32_t n = 0; n < L; n++) {
				for (int32_t m = 0; m <=n; m++) {
					arg = 2 * M_PI / L * (m *i + n*j); //
					Z(k,0) +=  in(i, j) *cos(arg); // in column-major order  
					//Z(k, 1)+=in(i, j)  *sin(arg); // ; //
					k++;
				}
			}
		}
	}
	//Z = Z.rowwise().sum();
	//Z=Z.unaryExpr(&norm); // row-> (Re^2,  Im^2)
	return  Z;// row -> Re
}
MAT CHoloNet::antiConv(const MAT& delta, const MAT& _kernel) {
	int32_t nRows = delta.rows();
	int32_t nCols = delta.cols();
	MAT _kernelDelta = MAT(_kernel.rows(), _kernel.cols()); // make private member so don't have to allocate each time
	_kernelDelta.setConstant(0);
	MAT norm = MAT(_kernel.rows(), _kernel.cols());

	for (int32_t j = 0; j < nCols; j++) {
		for (int32_t i = 0; i < nRows; i++) {
			for (int32_t n = j - kernelSize / 2 >= 0 ? -kernelSize / 2 : -j; n < (j + kernelSize / 2 + 1 <= nCols ? kernelSize / 2 + 1 : nCols - j); n++) {
				for (int32_t m = i - kernelSize / 2 >= 0 ? -kernelSize / 2 : -i; m < (i + kernelSize / 2 + 1 <= nRows ? kernelSize / 2 + 1 : nRows - i); m++) {
					_kernelDelta(m + kernelSize / 2, n + kernelSize / 2) += _kernel(m + kernelSize / 2, n + kernelSize / 2)*delta(i + m, j + n); //kernel(m+kernelSize/2, n + kernelSize / 2)*in(i + m, j + n);
					norm(m + kernelSize / 2, n + kernelSize / 2) += 1.0f;
				}
			}
		}
	}
	
	return norm.cwiseQuotient(_kernelDelta);
}
MAT CHoloNet::conv(const MAT& in, const MAT& _kernel, uint32_t stride, uint32_t padding) {
	
	int32_t inRows = in.rows(); // we only accept square matrices 
	if (inRows != in.cols()) {
		// error
		return in;
	}
	int32_t inCols = inRows; // to make it easier to read
	int32_t _kernelSize = _kernel.rows();
	if (_kernelSize != _kernel.cols()) {
		return in;
	}
	size_t outSize = convoSize(inRows, _kernelSize, padding, stride); 

	MAT out(outSize, outSize); // square matrix
	MAT paddedIn(inRows + 2*padding, inCols + 2*padding);
	paddedIn.setConstant(0);
	// fill paddedIn matrix
	paddedIn.block(padding, padding, inRows, inCols) = in;
	
	out.setConstant(0);
	// convolution
	for (int32_t i = 0; i < outSize; i++) {
		for (int32_t j = 0; j < outSize; j++) {
			for (int32_t n = 0; n < kernelSize; n++) {
				for (int32_t m = 0; m < kernelSize; m++) {
					out(j, i) += _kernel(m, n)*paddedIn(j*stride + m, i*stride + n);
				}
			}
		}
	}
	return out;
}

MAT& CHoloNet::getKernel() {
	return kernel;
}
