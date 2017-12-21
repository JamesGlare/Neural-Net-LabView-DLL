#include "stdafx.h"
#include "HoloNet.h"

CHoloNet::CHoloNet(uint32_t _NINXY, uint32_t _NOUTXY, uint32_t _NNODES) : NINXY(_NINXY), NOUTXY(_NOUTXY),NNODES(_NNODES){

	// (1) construct inLayer

	inFourier = MAT(NINXY*(NINXY+1)/2,1);
	inFourier.setConstant(0);
	kernelSize = 3;
	stride = 0;
	padding = 0;

	// (2) Setup hidden layer
	hiddenLayer1 = MAT(NNODES, inFourier.size() + 1);
	hiddenLayer1.setRandom();
	hiddenDelta1 = MAT(NNODES, 1);
	hiddenAct1 = MAT(NNODES, 1);
	hiddenDelta1.setConstant(0);
	hiddenLayer2 = MAT(NOUTXY*NOUTXY, NNODES + 1); // convolution with the output of this matrix should yield (NOUTX * NOUTY)
	hiddenLayer2.setRandom();
	hiddenDelta2 = MAT(NOUTXY*NOUTXY, 1);
	hiddenAct2 = MAT(NOUTXY*NOUTXY, 1);
	// (3) Setup kernel matrix
	kernel = MAT(kernelSize, kernelSize);
	gauss(kernel);
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
	return in.unaryExpr(&Tanh);
}
MAT CHoloNet::DACT(const MAT& in) {
	return in.unaryExpr(&DTanh);
}
MAT CHoloNet::forProp(const MAT& in, bool saveActivations) {
	MAT temp = fourier(in);
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
	return conv(temp, kernel); // (NOUTXY, NOUTXY)
}

fREAL CHoloNet::backProp(const MAT& in, MAT& dOut, learnPars pars) {
	
	// (1) save activations
	MAT out = forProp(in, true);
	// (2) Calculate deltas
	MAT outDelta = out - dOut; // (NOUTXY, NOUTXY)
	fREAL error = l2Error(outDelta); // save error
	
	// (2.1) Kernel layer - Calculate deltas for NOUTXYxNOUTXY matrix, then later, sum down to kernel matrix with identity matrix
	MAT kernelDelta = conv(outDelta, kernel.reverse());// CHECK FLIPPED KERNEL(NOUTXYxNOUTXY);
	MAT temp = kernelDelta;
	temp.resize(NOUTXY*NOUTXY, 1); // RESIZE
											   
	// (2.2) Hidden layer 2 - (NOUTXY^2,1)
	// Activation has (NOUTXY^2,1) shape
	// (NOUTXY^2,1) = DACT(NOUTXY^2,1) * (NOUTXY^2,1)
	hiddenDelta2 = DACT(hiddenAct2).cwiseProduct(temp);
	// (2.3) Hidden layer 1 -  
	// (NNODES,1) = DACT(NNODES,1) * (NOUTXY^2, NNODES).T x (NOUTXY^2,1) = (NNODES,1)
	hiddenDelta1 = DACT(hiddenAct1).cwiseProduct(hiddenLayer2.leftCols(hiddenLayer2.cols() - 1).transpose()*hiddenDelta2);

	// (3) Apply gradient
	//(3.1) kernelLayer
	// (kS, kS)
	temp = ACT(hiddenAct2);
	temp.resize(NOUTXY, NOUTXY);
	MAT oneMatrix = MAT(kernelSize, kernelSize);
	oneMatrix.setConstant(1);
	//kernel = kernel - pars.eta * antiConv(temp.cwiseProduct(kernelDelta),oneMatrix); // .cwiseProduct(kernelDelta)
	
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
MAT CHoloNet::conv(const MAT& in, const MAT& _kernel) {
	
	int32_t nRows = in.rows();
	int32_t nCols = in.cols();
	MAT out(nRows, nCols);
	out.setConstant(0);
	for (int32_t j = 0; j < nCols; j++) {
		for (int32_t i = 0; i < nRows; i++) {
			for (int32_t n = j-kernelSize/2 >=0 ? -kernelSize / 2: -j; n < ( j+ kernelSize/2+1 <= nCols ? kernelSize / 2 + 1:nCols-j) ; n++) {
				for (int32_t m = i - kernelSize / 2 >= 0 ? -kernelSize / 2 : -i; m < (i + kernelSize / 2 + 1 <= nRows ? kernelSize / 2 + 1 : nRows - i); m++) {
					out(i, j) += _kernel(m + kernelSize / 2, n + kernelSize / 2)*in(i + m, j + n); //kernel(m+kernelSize/2, n + kernelSize / 2)*in(i + m, j + n);
 				}
			} 
		}
	}
	return out;
}

MAT CHoloNet::getKernel() const {
	return kernel;
}
/*
 * DLL Functions
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

__declspec(dllexport) void __stdcall testConvolution(CHoloNet* ptr, fREAL* const img) {
	MAT in = MATMAP(img, ptr->get_NIN(), ptr->get_NIN()); // (NIN, 1) Matrix
	MAT out(in.rows(), in.cols());
	out = ptr->conv( in, ptr->getKernel());
	copyToOut(out.data(), img, out.size());
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
void gauss(MAT& in) {
	//EIGEN stores matrices in column-major order! 
	// iterate columns (second index)
	int32_t nRows = in.rows();
	int32_t nCols = in.cols();
	// outer perimeter of window is at 3 sigma boundary
	fREAL std = (nRows + nCols) / 2;
	fREAL norm = 1.0f / ( std*std* 2 * M_PI);

	for (int32_t j = 0; j < nCols; j++) {
		for (int32_t i = 0; i < nRows; i++) {
			in(i, j) = norm* exp(-(j - nCols / 2)*(j - nCols / 2) / (2*std*std) - (i - nRows / 2)*(i - nRows / 2) / (2* std*std));
		}
	}
}