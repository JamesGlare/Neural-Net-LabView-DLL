#pragma once


#ifndef CNET_DEFINITIONS_H_INCLUDE
#define CNET_DEFINITIONS_H_INCLUDE

#ifndef M_PI
#define M_PI 3.14159265359f
#endif // !1
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include "Eigen/Core"
#include "Eigen/Cholesky"
using namespace Eigen;
using namespace std;

// Define a few types.
typedef float fREAL;
typedef Matrix<fREAL, Dynamic,1> VEC;
typedef Matrix<fREAL, Dynamic, Dynamic> MAT;
typedef Matrix<fREAL, Dynamic, Dynamic, RowMajor> MAT_ROWMAJOR;
typedef Matrix<fREAL, Dynamic, Dynamic, Dynamic> MAT3;
typedef Matrix<uint8_t, Dynamic, Dynamic> MATU8;
typedef Matrix<size_t, Dynamic, Dynamic> MATINDEX; // needed in MaxPool, Dropout
typedef const Eigen::Block<const MAT> MATBLOCK;
#define _FEAT(i) block(0, kernelX*i, kernelY, kernelX) // break of style *** TODO REPLACE WITH FUNCTION ***
#define _UFEAT(i) block(kernelY*i, 0, kernelY, 1)
#define _VFEAT(i) block(kernelX*i, 0, kernelX, 1)

typedef tuple<MAT, MAT> MATPAIR;

struct MATIND {
	size_t rows;
	size_t cols;
};
typedef Map<MAT> MATMAP;
typedef Map<MAT_ROWMAJOR> MATMAP_ROWMAJOR;
typedef vector<MAT> MATVEC;
typedef Map<MATU8> MATU8MAP;
typedef fREAL(*ACTFUNC)(fREAL);
typedef LLT<MAT> CHOL;

enum actfunc_t {RELU =1, TANH=2, SIG=3, NONE=4, SOFTPLUS=5, LEAKYRELU=6};
enum layer_t { fullyConnected = 0, convolutional = 1, antiConvolutional=2, maxPooling = 3, avgPooling=4, cnet = 5, passOn = 6, dropout=7, mixtureDensity=8, reshape=9, sideChannel = 10}; // enumerators: 1, 2, 4 range: 0..7
enum pooling_t {max =1, average = 2};
enum hierarchy_t { input = 1, hidden = 2, output = 3};

struct learnPars {
	learnPars() {
		eta = 0.0f; GAN_c = 0.0f; gamma = 0.0f; lambda = 0.0f; rmsprop = 0; adam = 0;
		batch_update = 0; weight_normalization = 0; spectral_normalization = 0; firstTrain = 0; lastTrain = 0; accept = true;
	};
	learnPars(fREAL _eta, fREAL _GAN_c, fREAL _gamma, fREAL _lambda, uint32_t _rmsprop, uint32_t _adam, uint32_t _batch_update,
		uint32_t _weight_normalization, uint32_t _spectral_normalization, uint32_t _firstTrain, uint32_t _lastTrain, bool _accept)
		: eta(_eta), GAN_c(_GAN_c), gamma(_gamma), lambda(_lambda), rmsprop(_rmsprop), adam(_adam), batch_update(_batch_update), weight_normalization(_weight_normalization),
		spectral_normalization(_spectral_normalization), firstTrain(_firstTrain), lastTrain(_lastTrain), accept(_accept){};

	fREAL eta; // learning rate
	fREAL GAN_c; // e.g. for clipping weights or gradient penalty (wasserstein GAN)
	fREAL gamma; // inertia term
	fREAL lambda; // regularizer
	uint32_t rmsprop; // 0 or 1
	uint32_t adam;
	uint32_t batch_update;
	uint32_t weight_normalization;
	uint32_t spectral_normalization;
	uint32_t firstTrain;
	uint32_t lastTrain;
	bool accept;
};
// library functions

MAT conv_(const MAT& in, const MAT& kernel, uint32_t NOUTY, uint32_t NOUTX, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t outChannels, uint32_t inChannels);
MAT antiConv_(const MAT& in, const MAT& kernel, size_t NOUTY, size_t NOUTX, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t outChannels, uint32_t inChannels);
MAT convGrad_(const MAT& input, const MAT& delta, uint32_t strideY, uint32_t strideX, uint32_t kernelY, uint32_t kernelX, uint32_t paddingY, uint32_t paddingX, uint32_t outChannels, uint32_t inChannels);
MAT antiConvGrad_(const MAT& delta, const MAT& input, size_t kernelY, size_t kernelX, uint32_t strideY, uint32_t strideX, uint32_t paddingY, uint32_t paddingX, uint32_t outChannels, uint32_t inChannels);
MAT fourier(const MAT& in);
void clipParameters(MAT& layers, fREAL clip);
fREAL spectralNorm(const MAT& W, const MAT& u1, const MAT& v1);
void updateSingularVectors(const MAT& W, MAT& u, MAT& v, uint32_t n);
// found online - check for NANs and infinities
template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
	return ((x - x).array() == (x - x).array()).all();
}
/* Extract a submatrix out of a full matrix
* given a number of indices.
*/
void extract(MAT& out, const MAT& full, const MATINDEX& ind);
void setZeroAtIndex(MAT& in, const MATINDEX& ind, size_t nrFromTop);

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
	return ((x.array() == x.array())).all();
}
template <typename T> 
inline int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}
template<typename T>
inline void copyToOut(T* const in, T* const out, uint32_t N) {
	for (uint32_t i = 0; i < N; i++) {
		out[i] = in[i];
	}
}
inline bool inRange(uint32_t x, uint32_t a, uint32_t b) {
	return x >= a && x <= b;
}
inline fREAL iden(fREAL f) {
	return f;
}
inline fREAL DIden(fREAL f) {
	return 1.0f;
}

inline fREAL cumSum(const MAT& in) {
	return in.sum();
}
// Activation functions & derivatives
inline fREAL Tanh(fREAL f) {
	return tanh(f);
}
inline fREAL DTanh(fREAL f) {
	return 1.0f - Tanh(f)*Tanh(f);
}
inline fREAL Sig(fREAL f) {
	return 1.0f / (1.0f + exp(-1.0f*f));
}
inline fREAL LogExp(fREAL x) {
	return log(1.0f + exp(-1.0f*x));
}
inline fREAL LogAbsExp(fREAL x) {
	return log(1.0f + exp(-1.0f*abs(x)));
}
inline fREAL DLogAbsExp(fREAL x) {
	fREAL absX = abs(x);
	return -x / (absX*(exp(absX) + 1.0f));
}
inline fREAL DSig(fREAL f) {
	return Sig(f)*(1.0f - Sig(f));
}
inline fREAL SoftPlus(fREAL f) {
	return std::log(1.0f + std::exp(f)); // smooth relu
}
inline fREAL DSoftPlus(fREAL f) {
	return Sig(f); // smooth relu
}
inline fREAL ReLu(fREAL f) {
	return f > 0.0f ? f : 0.0f;
	//
}
inline fREAL DReLu(fREAL f) {
	return f > 0.0f ? 1.0f : 0.0f;
	//
}
inline fREAL LeakyReLu(fREAL f) {
	return f > 0.0f ? f : 0.2f*f;
}

inline fREAL DLeakyReLu(fREAL f) {
	return f > 0.0f ? 1.0f : -0.2f;
}

inline fREAL norm(fREAL f) {
	return f*f;
}
inline fREAL cube(fREAL f) {
	return f*f*f;
}
inline fREAL sqroot(fREAL f) {
	return sqrt(f); // to avoid overloading problems
}
inline fREAL invSqrt(fREAL f) {
	return 1.0f/sqrt(f);
}
inline fREAL exp_fREAL(fREAL f) {
	return exp(f);
}
inline fREAL logP1_fREAL(fREAL f) {
	return log(f+1.0f);
}
inline MAT matNorm(const MAT& in) {
	return in.unaryExpr(&norm);
}
inline fREAL normSum(const MAT& in) {
	return sqrt(matNorm(in).sum());
}
inline uint32_t convoSize(uint32_t inSize, uint32_t kernelSize, uint32_t padding, uint32_t stride) {
	return (inSize - kernelSize + 2 * padding) / stride + 1;
}
inline uint32_t inStrideConvoSize(uint32_t NOUTXY, uint32_t NINXY, uint32_t stride, uint32_t padding) {
	return NOUTXY - stride*(NINXY - 1) + 2 * padding; 
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
inline uint32_t antiConvPad(uint32_t inSize, uint32_t stride, uint32_t kernelSize, uint32_t outSize) {
	return (stride*(inSize - 1) + kernelSize - outSize) / 2;
}
void flipUD(MAT& toFlip);
void flipLR(MAT& toFlip);

// MAT functions
void appendOne(MAT&);
void shrinkOne(MAT&);
//MAT& appendOneInline(MAT&);
MAT& appendOneInline(MAT&);
void gauss(MAT& in);
fREAL multiNormalDistribution(const MAT& t, const MAT& mu, const MAT& corvar);
fREAL normalDistribution(const MAT& t, const MAT& mu, fREAL var);
fREAL normal_dist(fREAL mu, fREAL stddev); // for initialization purposes

#endif // !DEFINITIONS_H_INCLUDE
