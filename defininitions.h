#pragma once
#ifndef M_PI
#define M_PI 3.14159265359f
#endif // !1

#include <vector>
#include <memory>
#include "Eigen/Core"

using namespace Eigen;
using namespace std;

// Define a few types.
typedef float fREAL;
typedef Matrix<fREAL, Dynamic, Dynamic> MAT;
typedef Matrix<fREAL, Dynamic, Dynamic, Dynamic> MAT3;
typedef Matrix<uint8_t, Dynamic, Dynamic> MATU8;
typedef Map<MAT> MATMAP;
typedef vector<MAT> MATVEC;
typedef Map<MATU8> MATU8MAP;

template<typename T>
void inline copyToOut(T* const in, T* const out, uint32_t N) {
	for (uint32_t i = 0; i < N; i++) {
		out[i] = in[i];
	}
}

struct learnPars {
	fREAL eta; // learning rate
	fREAL metaEta; // learning rate decay rate
	fREAL gamma; // inertia term
	fREAL lambda; // regularizer
	uint32_t nesterov;
};

fREAL cumSum(const MAT& in) {
	return in.sum();
}
// static functions
inline fREAL Tanh(fREAL f) {
	return std::tanh(f);
}
inline fREAL Sig(fREAL f) {
	return 1.0f / (1.0f + std::exp(-1.0f*f));
}
inline fREAL DSig(fREAL f) {
	return Sig(f)*(1.0f - Sig(f));
}
inline fREAL ReLu(fREAL f) {
	return std::log(1.0f + std::exp(f));
}
inline fREAL DReLu(fREAL f) {
	return Sig(f);
}
inline fREAL norm(fREAL f) {
	return f*f;
}
inline MAT matNorm(const MAT& in) {
	return in.unaryExpr(&norm);
}
// MAT functions
void appendOne(MAT& in) {
	in.conservativeResize(in.rows() + 1, in.cols()); // (NIN+1,1)
	in.bottomRows(1) = MAT(1, 1).setConstant(1); // bottomRows etc can be used as lvalue 
}
void shrinkOne(MAT& in) {
	in.conservativeResize(in.rows() - 1, in.cols());
}
inline MAT appendOneInline(const MAT& toAppend) {
	MAT temp = MAT(toAppend.rows() + 1, toAppend.cols()).setConstant(1);
	temp.topRows(toAppend.rows()) = toAppend;
	return temp;
}

fREAL gauss(MAT& in, uint32_t x, uint32_t y) {
//EIGEN stores matrices in column-major order! 
// iterate columns (second index)
	size_t nRows = in.rows();
	size_t nCols = in.cols();
	// outer perimeter of window is at 3 sigma boundary
	uint32_t std = (nRows + nCols) / 2;
	fREAL norm = 1.0f /(3*std* sqrt(2 * M_PI));

	for (size_t j = 0; j < nCols; j++) {
		for (size_t i = 0; i < nRows; i++) {
			in(i, j) = norm* exp(-(j-nCols/2)*(j - nCols / 2) /(18*std)-(i - nRows / 2)*(i - nRows/ 2) / (18*std));
		}
	}
}