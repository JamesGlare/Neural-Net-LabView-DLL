#include "stdafx.h"
#include "defininitions.h"

/* Define a few more complex functions
*/


// MAT functions
void appendOne(MAT& in) {
	in.conservativeResize(in.rows() + 1, in.cols()); // (NIN+1,1)
	in.bottomRows(1).setConstant(1); // bottomRows etc can be used as lvalue 
}
void shrinkOne(MAT& in) {
	in.conservativeResize(in.rows() - 1, in.cols());
}
MAT& appendOneInline(MAT& toAppend) {
	//MAT temp = MAT(toAppend.rows() + 1, toAppend.cols());
	//temp.setOnes();
	//temp.topRows(toAppend.rows()) = toAppend;
	toAppend.conservativeResize(toAppend.rows() + 1, toAppend.cols());
	toAppend.bottomRows(1).setConstant(1);
	return toAppend;
}
MAT conv(const MAT& in, const MAT& kernel, uint32_t kernelStrideY, uint32_t kernelStrideX, uint32_t paddingY, uint32_t paddingX, uint32_t features ) {

	size_t inY = in.rows(); // we only accept square matrices 
	size_t inX = in.cols();

	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols()/features;
	
	size_t outSizeY = convoSize(inY, kernelY, paddingY, kernelStrideY);
	size_t outSizeX = convoSize(inX, kernelX, paddingX, kernelStrideX);

	MAT out(outSizeY, features*outSizeX); // 
	MAT paddedIn(inY + 2 * paddingY, inX+ 2 * paddingX);
	// Set padding and out matrix zero, then fill inner part.
	paddedIn.setZero();
	out.setZero();
	paddedIn.block(paddingY, paddingX, inY, inX) = in;
	uint32_t f = 0;
	uint32_t iDim = 0;
	// convolution
	for (size_t i = 0; i < features*outSizeX; i++) { // max(i) = outSizeX-1, //max(f*outSizeX+i) = (features-1)*outSizeX + outSizeX-1 = features*outSizeX-1
		f = i / outSizeX;
		iDim = i%outSizeX;
		for (size_t n = 0; n < kernelX; n++) {
			for (size_t j = 0; j < outSizeY; j++) {
				for (size_t m = 0; m < kernelY; m++) {

					out(j, f*outSizeX + iDim) += kernel(m, f*kernelX + n)*paddedIn(j*kernelStrideY + m, iDim*kernelStrideX + n);
				}
			}
		}
	}
	return out;
}
MAT convGrad(const MAT& delta, const MAT& input, uint32_t strideY, uint32_t strideX, uint32_t kernelY, uint32_t kernelX, uint32_t paddingY, uint32_t paddingX, uint32_t features) {

	size_t NOUTY = delta.rows(); // we only accept square matrices 
	size_t NOUTX = delta.cols()/features;

	size_t NINY = input.rows();
	size_t NINX = input.cols();

	MAT out(kernelY, features*kernelX); // 
	MAT paddedIn(NINY + 2 * paddingY, NINX+ 2 * paddingX);
	paddedIn.setConstant(0);
								 // fill paddedIn matrix
	paddedIn.block(paddingY, paddingX, NINY, NINX) = input;

	out.setConstant(0);
	uint32_t f = 0;
	uint32_t iDim = 0;

	// convolution
	// NOTE: Eigen matrices are stored column-major.
	// Y array should be continguous.
	for (size_t i = 0; i < kernelX*features; i++) { // 0, ...,7
		f = i / kernelX;
		iDim = i%kernelX;
		for (size_t n = 0; n < NOUTX; n++) { // 0, 1
			for (size_t j = 0; j < kernelY; j++) { // 0, ...,7
				for (size_t m = 0; m < NOUTY; m++) { // 0,1
					out(j, f*kernelX+ iDim) += paddedIn(j + m*strideY, iDim + n*strideX)*delta(m, f*NOUTX+n); // max[j+m*stride] == K-1+(NO-1)*stride => [K+NO*stride - stride] -1
				}
			}
		}
	}
	return out;
}
MAT antiConvGrad(const MAT& delta, const MAT& input, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t features) {

	size_t NOUTY = delta.rows(); 
	size_t NOUTX= delta.cols();

	size_t NINY = input.rows();
	size_t NINX = input.cols()/features; // features are along the x-dimension

	size_t outSizeY = inStrideConvoSize(NOUTY, NINY, strideY, paddingY); // 8
	size_t outSizeX = inStrideConvoSize(NOUTX, NINX, strideX, paddingX);

	MAT out(outSizeY, features*outSizeX); // 
	out.setZero();

	MAT paddedDelta(NOUTY + 2 * paddingY, NOUTX + 2 * paddingX);
	paddedDelta.setZero();
	// fill paddedIn matrix
	paddedDelta.block(paddingY, paddingX, NOUTY, NOUTX) = delta;
	uint32_t f = 0;
	uint32_t iDim = 0;

	// convolution
	for (size_t i = 0; i < features*outSizeX; i++) { // 0, ...,7
		f = i / outSizeX;
		iDim = i%outSizeX;
		for (size_t n = 0; n < NINX; n++) { // 0, 1
			for (size_t j = 0; j < outSizeY; j++) { // 0, ...,7
				for (size_t m = 0; m < NINY; m++) { // 0,1
					out(j, f*outSizeX+iDim) += input(m, f*NINX+n)*paddedDelta(j + m*strideY, iDim + n*strideX); // max -> 7+2 = 9
				}
			}
		}
	}
	return out;
}

/*
MAT deltaActConv(const MAT& deltaAbove, const MAT& actBelow, uint32_t kernelSizeY, uint32_t kernelSizeX, uint32_t strideUsed, uint32_t paddingYUsed, uint32_t paddingXUsed) {
	// outSize == kernelSize

	size_t deltaSizeY = deltaAbove.rows();
	size_t deltaSizeX = deltaAbove.cols();
	MAT actPadded(actBelow.rows()+2*paddingYUsed, actBelow.cols()+2*paddingXUsed);
	actPadded.setConstant(0);
	actPadded.block(paddingYUsed, paddingXUsed, actBelow.rows() , actBelow.cols()) = actBelow;
	MAT out(kernelSizeY, kernelSizeX);
	out.setConstant(0);
	for (size_t i = 0; i < kernelSizeX; i++) {
		for (size_t j = 0; j < kernelSizeY; j++) {
			for (size_t m = 0; m < deltaSizeY; m++) {
				for (size_t n = 0; n < deltaSizeX; n++) {
					out(j, i) += deltaAbove(m, n)*actPadded(j+strideUsed*m,i+strideUsed*n);
				}
			}
		}
	}
	return out;
}*/
MAT antiConv(const MAT& in, const MAT& kernel, uint32_t strideY, uint32_t strideX, uint32_t antiPaddingY, uint32_t antiPaddingX, uint32_t features) {

	size_t inY = in.rows(); 
	size_t inX = in.cols() / features; // features are in X direction 

	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols()/features; // features are in X direction 

	size_t outSizeY = antiConvoSize(inY,kernelY, antiPaddingY, strideY); // inSize*stride - stride + kernelSize-2*padding
	size_t outSizeX = antiConvoSize(inX, kernelX, antiPaddingX, strideX) ;
	MAT out(outSizeY + 2 * antiPaddingY, outSizeX + 2 * antiPaddingX); // make it bigger, we need to cut out later
	out.setConstant(0);

	uint32_t f = 0;
	uint32_t iDim = 0;

	// convolution
	for (size_t i = 0; i < features*inX; i++) {
		f = i / inX;
		iDim = i%inX;
		for (size_t n = 0; n < kernelX; n++) {
			for (size_t j = 0; j < inY; j++) {
				for (size_t m = 0; m < kernelY; m++) {
					out(j*strideY + m, iDim*strideX + n) += kernel._FEAT(f)(m, n)*in(j, f*inX+iDim); // max[j*stride+m] = (inY-1)*stride + kernelY-1 = inY*stride-stride + kernelY-1 == outSizeY-1 -> correct
				}
			}
		}
	}
	return out.block(antiPaddingY, antiPaddingX, outSizeY, outSizeX);
}
MAT fourier(const MAT& in) {
	size_t L = in.rows(); // == in.cols() 
	MAT Z = MAT(L*(L + 1) / 2, 2); // number of unique elements - should match inLayer's dimensionality 
	Z.setConstant(0);
	uint32_t k = 0;
	fREAL arg = 0;

	for (size_t j = 0; j < L; j++) {
		for (size_t i = 0; i < L; i++) {
			k = 0;
			for (size_t n = 0; n < L; n++) {
				for (size_t m = 0; m <= n; m++) {
					arg = 2.0f * M_PI / L * (m *i + n*j); //
					Z(k, 0) += in(i, j) *cos(arg); // in column-major order  
					Z(k, 1)+=in(i, j)  *sin(arg); // ; //
					k++;
				}
			}
		}
	}
	//Z = Z.rowwise().sum();
	//Z=Z.unaryExpr(&norm); // row-> (Re^2,  Im^2)
	return  Z;// row -> Re
}

void gauss(MAT& in) {
	//EIGEN stores matrices in column-major order! 
	// iterate columns (second index)
	int32_t nRows = in.rows();
	int32_t nCols = in.cols();
	// outer perimeter of window is at 3 sigma boundary
	fREAL stdY =nRows;
	fREAL stdX = nCols;
	fREAL norm = 1.0f / ((stdX*stdX+stdY+stdY ) * M_PI);

	for (int32_t j = 0; j < nCols; j++) {
		for (int32_t i = 0; i < nRows; i++) {
			in(i, j) = norm* exp(-(j - nCols / 2)*(j - nCols / 2) / (2 * stdX*stdX) - (i - nRows / 2)*(i - nRows / 2) / (2 * stdY*stdY));
		}
	}
}