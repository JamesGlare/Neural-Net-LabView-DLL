#include "stdafx.h"
#include "HoloNet.h"


CHoloNet::CHoloNet(uint32_t _NINXY, uint32_t _NOUTXY, uint32_t _NNODES) : NINXY(_NINXY), NOUTXY(_NOUTXY),NNODES(_NNODES){

	// (1) construct inLayer
	inLayer = MAT(NINXY*(NINXY+1)/2,1);
	inLayer.setRandom();
	inAct = MAT(inLayer.size(), 1);
	inAct.setConstant(0);

	kernelSize = 3;
	stride = 0;
	padding = 0;

	// (2) Setup hidden layer
	hiddenLayer1 = MAT(NNODES, inLayer.size() + 1);
	hiddenLayer2 = MAT(NOUTXY*NOUTXY, NNODES + 1); // convolution with the output of this matrix should yield (NOUTX * NOUTY)
	// (3) Setup kernel matrix
	kernel = MAT(kernelSize, kernelSize);
	kernel.setRandom();

}

CHoloNet::~CHoloNet()
{
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
	MAT temp = inLayer.cwiseProduct(fourier(in));
	temp.resize(inLayer.size(), 1);
	if (saveActivations)
		inAct = temp;

	temp = ACT(temp);
	appendOne(temp);
	temp = hiddenLayer1*temp;
	if (saveActivations)
		hiddenAct1 = temp;

	temp = ACT(temp);
	appendOne(temp);
	temp = hiddenLayer2*temp;
	if (saveActivations)
		hiddenAct2 = temp;

	temp = ACT(temp);

	return conv(temp);
}

fREAL CHoloNet::backProp(const MAT& in, MAT& out, learnPars pars) {
	
	// (1) save activations
	MAT out = forProp(in, true);

}

MAT CHoloNet::fourier(const MAT& in) {
	size_t L = in.rows(); // == in.cols()
	MAT out = MAT(L*(L+1)/2,2); // number of unique elements
	out.setConstant(0);
	uint32_t k = 0;
	for (size_t j = 0; j < L; j++) {
		for (size_t i = 0; i < L; i++) {
			k = 0;
			for (size_t n = 0; n < L; n++) {
				for (size_t m = 0; m <= n; m++) {
					out(k,0) += in(i, j)*cos(2*m*i*M_PI / L+2*n*j*M_PI / L); // in column-major order
					out(k,1) += in(i, j)*sin(2 * m*i*M_PI / L + 2 * n*j*M_PI / L);
					k++;
				}
			}
		}
	}
	out.unaryExpr(&norm); // row-> (Re^2,  Im^2)
	return out.rowwise.sum(); // row -> Re^2+Im^2
}

MAT CHoloNet::conv(const MAT& in) {
	
	size_t nRows = in.rows();
	size_t nCols = in.cols();
	MAT out = MAT(nRows, nCols); // make private member so don't have to allocate each time
	out.setConstant(0);

	for (size_t j = 0; j < nCols; j++) {
		for (size_t i = 0; i < nRows; i++) {
			for (int32_t n = max(0, j-kernelSize/2); n < min(nCols, j+kernelSize/2+1); n++) {
				for (int32_t m = max(0, i-kernelSize/2); m < min(nRows, i+kernelSize/2+1); m++) {
					out(i, j) += kernel(m, n)*in(i + m, j + n);
 				}
			} 
		}
	}
	return out;
}