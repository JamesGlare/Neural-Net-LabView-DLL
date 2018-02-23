#pragma once


#ifndef DEFINITIONS_H_INCLUDE
#define DEFINITIONS_H_INCLUDE

#ifndef M_PI
#define M_PI 3.14159265359f
#endif // !1
#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Core"
using namespace Eigen;
using namespace std;

// Define a few types.
typedef float fREAL;
typedef Matrix<fREAL, Dynamic,1> VEC;
typedef Matrix<fREAL, Dynamic, Dynamic> MAT;
typedef Matrix<fREAL, Dynamic, Dynamic, Dynamic> MAT3;
typedef Matrix<uint8_t, Dynamic, Dynamic> MATU8;
typedef Matrix<size_t, Dynamic, Dynamic> MATINDEX;

typedef Map<MAT> MATMAP;
typedef vector<MAT> MATVEC;
typedef Map<MATU8> MATU8MAP;
typedef fREAL(*ACTFUNC)(fREAL);

enum actfunc_t {RELU =1, TANH=2, SIG=3, NONE=4};
enum layer_t { fullyConnected = 0, convolutional = 1, antiConvolutional=2, maxPooling = 3, avgPooling=4, cnet = 5}; // enumerators: 1, 2, 4 range: 0..7
enum pooling_t {max =1, average = 2};
enum hierarchy_t { input = 1, hidden = 2, output = 3};

struct learnPars {
	fREAL eta; // learning rate
	fREAL metaEta; // learning rate decay rate
	fREAL gamma; // inertia term
	fREAL lambda; // regularizer
	uint32_t conjugate;
};
// library functions

MAT conv(const MAT& in, const MAT& _kernel, uint32_t instride, uint32_t kernelStride, uint32_t paddingY, uint32_t paddingX);
MAT antiConv(const MAT& in, const MAT& kernel, uint32_t stride, uint32_t paddingY, uint32_t paddingX);
MAT deltaActConv(const MAT& deltaAbove, const MAT& actBelow, uint32_t kernelSizeY, uint32_t kernelSizeX, uint32_t strideUsed, uint32_t paddingYUsed, uint32_t paddingXUsed);

MAT fourier(const MAT& in);

// found online - check for NANs and infinities
template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
	return ((x - x).array() == (x - x).array()).all();
}

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
	return ((x.array() == x.array())).all();
}

template<typename T>
inline void copyToOut(T* const in, T* const out, uint32_t N) {
	for (uint32_t i = 0; i < N; i++) {
		out[i] = in[i];
	}
}

inline fREAL iden(fREAL f) {
	return f;
}
inline fREAL DIden(fREAL f) {
	return 0;
}
inline fREAL cumSum(const MAT& in) {
	return in.sum();
}
// Activation functions & derivatives
inline fREAL Tanh(fREAL f) {
	return std::tanh(f);
}
inline fREAL DTanh(fREAL f) {
	return 1.0f - Tanh(f)*Tanh(f);
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
inline uint32_t convoSize(uint32_t inSize, uint32_t kernelSize, uint32_t padding, uint32_t stride) {
	return (inSize - kernelSize + 2 * padding) / stride + 1;
}
inline uint32_t antiConvoSize(uint32_t inSize, uint32_t kernelSize,uint32_t antiPadding, uint32_t stride) {
	return inSize*stride + kernelSize - stride - 2*antiPadding;
}
inline uint32_t padSizeForEqualConv(uint32_t inSize, uint32_t kernelSize, uint32_t stride) {
	return ((stride - 1)*inSize + kernelSize - stride) / 2; // ONLY WORKS FOR ODD KERNELSIZES if STRIDE==1
}
inline uint32_t padSize(uint32_t outSize, uint32_t inSize, uint32_t kernelSize, uint32_t stride) {
	return (stride*outSize - stride - inSize + kernelSize) / 2; 
}
// MAT functions
void appendOne(MAT&);
void shrinkOne(MAT&);
MAT appendOneInline(const MAT&);

void gauss(MAT& in);
#endif // !DEFINITIONS_H_INCLUDE
