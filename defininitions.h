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

template<typename T>
void inline copyToOut(T* const, T* const, uint32_t);

inline fREAL cumSum(const MAT&);
// static functions
inline fREAL Tanh(fREAL);
inline fREAL DTanh(fREAL);
inline fREAL Sig(fREAL);
inline fREAL DSig(fREAL);
inline fREAL ReLu(fREAL);
inline fREAL DReLu(fREAL);
inline fREAL norm(fREAL);
inline MAT matNorm(const MAT&);
// MAT functions
void appendOne(MAT&);
void shrinkOne(MAT&);
inline MAT appendOneInline(const MAT& );

void gauss(MAT& in);
#endif // !DEFINITIONS_H_INCLUDE
