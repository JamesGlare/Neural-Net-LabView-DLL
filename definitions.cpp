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
void extract(MAT& out, const MAT& full, const MATINDEX& ind) {
	size_t num_indices = ind.rows();

	for (size_t i = 0; i < num_indices; ++i) {
		out(ind(i, 0), 0) = full(ind(i, 0), 0);
	}
}
/* ind specifies a set of indices where matrix in is set to zero.
*  Caller has responsibility to check whether indices are in range.
*/
void setZeroAtIndex(MAT& in, const MATINDEX& ind, size_t nrFromTop) {
	for (size_t i = 0; i < nrFromTop; ++i) {
		in(ind(i, 0), 0) = 0.0f;
	}
}
/* Parallelized convolution routine.
*/
MAT conv(const MAT& in, const MAT& kernel, uint32_t kernelStrideY, uint32_t kernelStrideX, uint32_t paddingY, uint32_t paddingX, uint32_t features) {

	size_t inY = in.rows(); // we only accept square matrices 
	size_t inX = in.cols();

	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols() / features;

	size_t outSizeY = convoSize(inY, kernelY, paddingY, kernelStrideY);
	size_t outSizeX = convoSize(inX, kernelX, paddingX, kernelStrideX);

	MAT out(outSizeY, features*outSizeX); // 
	MAT paddedIn(inY + 2 * paddingY, inX + 2 * paddingX);
	// Set padding and out matrix zero, then fill inner part.
	paddedIn.setZero();
	out.setZero();
	paddedIn.block(paddingY, paddingX, inY, inX) = in;
	uint32_t f = 0;
	uint32_t iDim = 0;
	// convolution
	int32_t i = 0;
#pragma omp parallel for private(i) shared(out, kernel, paddedIn)
	for (i = 0; i < features*outSizeX; ++i) { // max(i) = outSizeX-1, //max(f*outSizeX+i) = (features-1)*outSizeX + outSizeX-1 = features*outSizeX-1
		f = i / outSizeX;
		iDim = i%outSizeX;
		for (size_t n = 0; n < kernelX; ++n) {
			for (size_t j = 0; j < outSizeY; ++j) {
				for (size_t m = 0; m < kernelY; ++m) {
					#pragma omp critical
					out(j, f*outSizeX + iDim) += kernel(m, f*kernelX + n)*paddedIn(j*kernelStrideY + m, iDim*kernelStrideX + n);
				}
			}
		}
	}
	return out;
}
/* Parallelized convolution routine with in/out features.
*/
MAT conv_(const MAT& in, const MAT& kernel, uint32_t NOUTY, uint32_t NOUTX, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t outFeatures, uint32_t inFeatures) {
	
	// (1) Geometry of the situation
	size_t NINY = in.rows();
	size_t NINX = in.cols() / inFeatures;
	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols()/outFeatures;
	//size_t NOUTY = convoSize(NINY,kernelY,paddingY, strideY);
	//size_t NOUTX = convoSize(NINX, kernelX, paddingX, strideX);
	
	/*MAT paddedIn(NINY + 2 * paddingY, NINX + 2 * paddingX);
	// Set padding and out matrix zero, then fill inner part.
	paddedIn.setZero();
	paddedIn.block(paddingY, paddingX, NINY, NINX) = in;
	*/
	// (2) Allocate matrices 
	MAT out(NOUTY, NOUTX*inFeatures*outFeatures); // stack features along x in accord with convention
	out.setZero();

	// (3) Begin loop
	int32_t i = 0;
	int32_t f = 0;
	int32_t xInd = 0;
	int32_t yInd = 0;
	fREAL temp = 0.0f;

//#pragma omp parallel for private(i, xInd,yInd,f, temp) shared(out, kernel, in) // Choose (probably) smallest rowwise loop size for parallelization.
	for (i = 0; i < NOUTX; ++i) {
		for (size_t n = 0; n < kernelX; ++n) {
			for (size_t inF = 0; inF < inFeatures; ++inF) {
				for (size_t outF = 0; outF < outFeatures; ++outF) {
					for (size_t m = 0; m < kernelY; ++m) {
						for (size_t j = 0; j < NOUTY; ++j) { // Eigen matrices are stored in column-major order.
							f = inF + inFeatures*outF;
							yInd = j*strideY + m - paddingY;
							xInd = i*strideX + n - paddingX;
							 if(yInd < NINY &&
								yInd >= 0 &&
								xInd < NINX &&
								xInd >= 0) { // Check we're not in the padding.
								 temp = kernel(m, outF*kernelX + n) * in(yInd, xInd + inF*NINX);
								//#pragma omp critical
								out(j, i + f*NOUTX) += temp;
							} 
						}
					}
				}
			}
		}
	}
	return out;
}
/* Routine specifically for backpropagating deltas through a convolutional layer.
*/
MAT backPropConv_(const MAT& deltaIn, const MAT& kernel, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t features, uint32_t inFeatures) {
	// (1) Geometry of the situation
	size_t deltaNINY = deltaIn.rows();
	size_t deltaNINX = deltaIn.cols() / (features*inFeatures);
	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols() / features;
	size_t deltaNOUTY = antiConvoSize(deltaNINY, kernelY, paddingY, strideY);
	size_t deltaNOUTX = antiConvoSize(deltaNINX, kernelX, paddingX, strideX);

	// (2) Allocate matrices 
	MAT deltaOut(deltaNOUTY, deltaNOUTX*features); // stack features along x in accord with convention
	deltaOut.setZero();

	// (3) Begin loop
	int32_t i = 0;
	int32_t f = 0;
	int32_t xInd = 0;
	int32_t yInd = 0;
	fREAL temp = 0.0f;

//#pragma omp parallel for private(i,xInd,yInd,f, temp) shared(deltaOut, kernel, deltaIn) // Choose (probably) smallest rowwise loop size for parallelization.
	for ( i = 0; i < deltaNINX; ++i) {
		for (size_t inF = 0; inF < inFeatures; ++inF) {
			for (size_t outF = 0; outF < features; ++outF) {
				for (size_t n = 0; n < kernelX; ++n) {
					for (size_t m = 0; m < kernelY; ++m) {
						for (size_t j = 0; j < deltaNINX; ++j) { // Eigen matrices are stored in column-major order.
							f = inF + inFeatures*outF;
							yInd = j*strideY + m - paddingY ;
							xInd = i*strideX + n -paddingX ; // max[xInd] = (NOUTX-1)*strideX+kernelX - 1 = (NINX-kernelX+2*paddingX)+kernelX - 1 = NINX + 2*paddingX - 1 
							if (yInd < deltaNINX  &&
								yInd >= 0 &&
								xInd < deltaNINX &&
								xInd >= paddingX ) { // Check we're not in the padding.
								temp = kernel(m, outF*kernelX + n) * deltaIn(j, i + f*deltaNINX);
								//#pragma omp critical
								deltaOut(yInd, xInd + outF*deltaNOUTX) += temp;
							}
						}
					}
				}
			}
		}
	}
	return deltaOut;
}
MAT convGrad_(const MAT& in, const MAT& delta, uint32_t strideY, uint32_t strideX, uint32_t kernelY, uint32_t kernelX, uint32_t paddingY, uint32_t paddingX, uint32_t features, uint32_t inFeatures) {

	// (1) Geometry of the situation
	size_t NINY = in.rows();
	size_t NINX = in.cols() / inFeatures;
	size_t NINY_padded = convoSize(NINY, kernelY, paddingY, strideY);
	size_t NINX_padded = convoSize(NINX, kernelX, paddingX, strideX);
	size_t deltaY = delta.rows(); // deltaX = (NINX-kernelX+2*paddingX)/strideX +1 
	size_t deltaX = delta.cols() /(inFeatures* features);

	// (2) Allocate matrices 
	MAT kernelGrad(kernelY, kernelX*features); // stack features along x in accord with convention
	kernelGrad.setZero();

	// (3) Begin loop
	int32_t n = 0;
	size_t f = 0;
	int32_t xInd = 0;
	int32_t yInd = 0;
	fREAL temp = 0.0f;

//#pragma omp parallel for private(n,xInd, yInd,f, temp) shared(kernelGrad, delta, in)// Choose (probably) smallest rowwise loop size for parallelization.
	for (n = 0; n < deltaX; ++n) {
		for (size_t inF = 0; inF < inFeatures; ++inF) {
			for (size_t outF = 0; outF < features; ++outF) {
				for (size_t i = 0; i < kernelX; ++i) {
					for (size_t m = 0; m < deltaY; ++m) {
						for (size_t j = 0; j < kernelY; ++j) { // Eigen matrices are stored in column-major order.
							f = inF + inFeatures*outF;
							yInd = j + m*strideY - paddingY;
							xInd = i + n*strideX - paddingX; // max [xInd] = (kernelX-1)+ (deltaX-1)*strideX = (kernelX-1)+ (NINX-kernelX+2*paddingX) = NINX+2*paddingX-1 -> correct
							
							if (yInd < NINY &&
								yInd >= 0 &&
								xInd < NINX &&
								xInd >= 0) { // Check we're not in the padding.

								temp = delta(m, f*deltaX + n) * in(yInd, xInd + inF*NINX);
								//#pragma omp critical
								kernelGrad(j, i + outF*kernelX) += temp;
							}
						}
					}
				}
			}
		}
	}
	return kernelGrad;
}
/* Parallelized deconvolution operation (in this library referred to as Anticonvolution).
*/
MAT antiConv(const MAT& in, const MAT& kernel, uint32_t strideY, uint32_t strideX, uint32_t antiPaddingY, uint32_t antiPaddingX, uint32_t features) {

	size_t inY = in.rows();
	size_t inX = in.cols() / features; // features are in X direction 

	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols() / features; // features are in X direction 

	size_t outSizeY = antiConvoSize(inY, kernelY, antiPaddingY, strideY); // inSize*stride - stride + kernelSize-2*padding
	size_t outSizeX = antiConvoSize(inX, kernelX, antiPaddingX, strideX);
	MAT out(outSizeY + 2 * antiPaddingY, outSizeX + 2 * antiPaddingX); // make it bigger, we need to cut out later
	out.setConstant(0);

	uint32_t f = 0;
	uint32_t iDim = 0;
	fREAL temp = 0;
	// convolution
	int32_t i = 0;
	//#pragma omp parallel for private(i, temp) shared(out, kernel, in)
	for (i = 0; i < features*inX; ++i) {
		f = i / inX;
		iDim = i%inX;
		for (size_t n = 0; n < kernelX; ++n) {
			for (size_t j = 0; j < inY; ++j) {
				for (size_t m = 0; m < kernelY; ++m) {
					temp = kernel._FEAT(f)(m, n)*in(j, f*inX + iDim);
					//#pragma omp critical
					out(j*strideY + m, iDim*strideX + n) += temp; // max[j*stride+m] = (inY-1)*stride + kernelY-1 = inY*stride-stride + kernelY-1 == outSizeY-1 -> correct
				}
			}
		}
	}
	return out.block(antiPaddingY, antiPaddingX, outSizeY, outSizeX);
}
MAT antiConv_(const MAT& in, const MAT& kernel, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t outFeatures, uint32_t inFeatures) {
	// (1) Geometry of the situation
	size_t NINY = in.rows();
	size_t NINX = in.cols() / inFeatures;
	size_t kernelY = kernel.rows();
	size_t kernelX = kernel.cols() / outFeatures;
	size_t NOUTY = antiConvoSize(NINY, kernelY, paddingY, strideY);
	size_t NOUTX = antiConvoSize(NINX, kernelX, paddingX, strideX);

	// (2) Allocate matrices 
	MAT out(NOUTY, NOUTX*inFeatures*outFeatures); // stack features along x in accord with convention
	out.setZero();

	// (3) Begin loop
	int32_t n = 0;
	size_t f = 0;
	size_t xInd = 0;
	size_t yInd = 0;
	fREAL temp = 0;

//#pragma omp parallel for private(n, temp) shared(out, kernel, in) // Choose (probably) smallest rowwise loop size for parallelization.
	for (n = 0; n < kernelX; ++n) {
		for (size_t inF = 0; inF < inFeatures; ++inF) {
			for (size_t outF = 0; outF < outFeatures; ++outF) {
				for (size_t i = 0; i < NOUTX; ++i) {
					for (size_t m = 0; m < kernelY; ++m) {
						for (size_t j = 0; j < NOUTY; ++j) { // Eigen matrices are stored in column-major order.
							f = inF + inFeatures*outF;
							yInd = j*strideY + m;
							xInd = i*strideX + n;
							if (yInd < NOUTY && xInd < NOUTX) { // Check we're not in the padding.
								temp = kernel(m, outF*kernelX + n) * in(j, i + inF*NINX);
								//#pragma omp critical
								out(yInd, xInd + f*NOUTX) += temp;
							}
						}
					}
				}
			}
		}
	}
	return out;
}

/* Parallelized routine to calculate convolution gradient.
*/
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
	int32_t i = 0;
	#pragma omp parallel for private(i) shared(out, delta, paddedIn)
	for (i = 0; i < kernelX*features; ++i) { // 0, ...,7
		f = i / kernelX;
		iDim = i%kernelX;
		for (size_t n = 0; n < NOUTX; ++n) { // 0, 1
			for (size_t j = 0; j < kernelY; ++j) { // 0, ...,7
				for (size_t m = 0; m < NOUTY; ++m) { // 0,1
					#pragma omp critical
					out(j, f*kernelX+ iDim) += delta(m, f*NOUTX + n)*paddedIn(j + m*strideY, iDim + n*strideX); // max[j+m*stride] == K-1+(NO-1)*stride => [K+NO*stride - stride] -1
				}
			}
		}
	}
	return out;
}
/* Parallelized routine to calculate gradient for anticonvolutional layers.
*/
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
	int32_t i = 0;
	//#pragma omp parallel for private(i) shared(out, input, paddedDelta)
	for (i = 0; i < features*outSizeX; ++i) { // 0, ...,7
		f = i / outSizeX;
		iDim = i%outSizeX;
		for (size_t n = 0; n < NINX; ++n) { // 0, 1
			for (size_t j = 0; j < outSizeY; ++j) { // 0, ...,7
				for (size_t m = 0; m < NINY; ++m) { // 0,1
					//#pragma omp critical
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
	size_t nRows = in.rows();
	size_t nCols = in.cols();
	// outer perimeter of window is at 3 sigma boundary
	fREAL stdY =nRows;
	fREAL stdX = nCols;
	fREAL norm = 1.0f / ((stdX*stdX+stdY+stdY ) * M_PI);

	for (size_t j = 0; j < nCols; j++) {
		for (size_t i = 0; i < nRows; i++) {
			in(i, j) = norm* exp(-(j - nCols / 2.0f)*(j - nCols / 2.0f) / (2.0f * stdX*stdX) - (i - nRows / 2.0f)*(i - nRows / 2.0f) / (2.0f * stdY*stdY));
		}
	}
}
fREAL normalDistribution(const MAT& t, const MAT& mu, fREAL var) {
	static const fREAL epsilon = 1E-8;
	//assert(t.size() == mu.size());
	static const fREAL TwoPi= 2.0f*M_PI;
	fREAL norm = pow(TwoPi*var, t.size() / 2.0f);
	fREAL diffSq = (t - mu).squaredNorm();
	return exp(-0.5f*diffSq / (var)) / norm;
}
/* Calculate probability to find t in given distribution.
*/
fREAL multiNormalDistribution(const MAT& t, const MAT& mu, const MAT& corvar) {
	// (1) by contract: x.rows == t.rows

	const fREAL logSqrt2Pi = 0.5f*std::log(2.0f * M_PI);

	int32_t i = 0;
	CHOL cholesky(corvar); // Compute cholesky decomposition
	if (cholesky.info() != Eigen::Success) {
		// problem !!
		return NAN;
	}
	const CHOL::Traits::MatrixL& L = cholesky.matrixL();
	fREAL quadform = (L.solve(mu - t)).squaredNorm();
	return exp(-t.rows()*logSqrt2Pi - 0.5f*quadform) / L.determinant();
}