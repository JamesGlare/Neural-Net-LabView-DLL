#pragma once
#include "defininitions.h"

#ifndef CNET_BATCHNORMALIZER
#define CNET_BATCHNORMALIZER

class BatchNormalizer {
public:
	BatchNormalizer(size_t NIN );
	MAT& normalize(MAT& input) const;
	MAT& deNormalize(MAT& input) const;
	void updateBuffer(MAT& input);
	void updateModel(); // infer variance, mean from minibatch
	MAT& passOnNormalized(); // NOTE: make this MAT&& ? return value optimization of the compiler probably gets rid of this anyway
	inline uint32_t stillToCome() const { return stillToGo; }; // can't name function as variable...
	void clearBuffer();
	void setToAverage();
	void trainParameters(const MAT& y, const learnPars& pars);

	//friend class PhysicalLayer; // Physical layer orchestrates the normalization process
private:
	size_t stillToGo; 
	MATVEC batchBuffer; // Store input matrices over minibatch
	fREAL eps = 1e-10;
	MAT mues; // mean
	MAT sigmas; // standard deviations
	MAT gammas; // scaling parameters 1
	MAT betas; // scaling parameters 2
	size_t NIN;  // input dimensions

};

#endif