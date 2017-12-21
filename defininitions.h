#pragma once
#ifndef M_PI
#define M_PI 3.14159265359f
#endif // !1

#include <vector>
#include <memory>
#include "Eigen/Core"

using namespace Eigen;
using namespace std;
#ifndef DEFINITIONS_H_INCLUDE
#define DEFINITIONS_H_INCLUDE


// Define a few types.
typedef float fREAL;
typedef Matrix<fREAL, Dynamic, Dynamic> MAT;
typedef Matrix<fREAL, Dynamic, Dynamic, Dynamic> MAT3;
typedef Matrix<uint8_t, Dynamic, Dynamic> MATU8;
typedef Map<MAT> MATMAP;
typedef vector<MAT> MATVEC;
typedef Map<MATU8> MATU8MAP;



struct learnPars {
	fREAL eta; // learning rate
	fREAL metaEta; // learning rate decay rate
	fREAL gamma; // inertia term
	fREAL lambda; // regularizer
	uint32_t nesterov;
};
// library functions

template<typename T>
void inline copyToOut(T* const, T* const, uint32_t);

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
// MAT functions
void appendOne(MAT&);
void shrinkOne(MAT&);
MAT appendOneInline(const MAT& );

void gauss(MAT& in);
#endif // !DEFINITIONS_H_INCLUDE
