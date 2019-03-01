#pragma once
#include "defininitions.h"
#include <stack>
#ifndef CNET_BATCHBUFFER
#define CNET_BATCHBUFFER

class BatchBuffer {
public:
	BatchBuffer(MATIND _layerInd, size_t NOUT, size_t NIN);
	BatchBuffer(size_t NOUT, size_t NIN);
	// Standard gradient minibatch
	void swallowGradient(const MAT& grad);
	MAT avgGradient();
	MAT rmsGradient();
	
	void clearGradients();
	// Buffer for actual inputs - if we are interested in statistics etc
	void updateBuffer(MAT& input);
	void updateModel();
	MAT batchRMS();
	MAT batchMax();
	MAT batchMean();
	inline size_t stillToCome() const { return gradientBuffer.size(); }; // can't name function as variable...
	inline size_t batchSize() const { return batchBuffer.size(); };
	void clearBuffer();
	void normalize(MAT& input) const;

private:
	size_t stillToGo;
	MATVEC batchBuffer; // Store input matrices over minibatch
	MATVEC gradientBuffer;
	MAT nullGradient;
	const size_t MAX_SIZE = 100;
	fREAL eps = 1e-10;
	MAT mues; // mean
	MAT sigmas; // standard deviations
	size_t NIN;  // input dimensions
	size_t NOUT; //output dimensions
};

#endif