#pragma once
#include "defininitions.h"

#ifndef CNET_BATCHBUFFER
#define CNET_BATCHBUFFER

class BatchBuffer {
public:
	BatchBuffer(MATIND _layerInd, size_t NOUT, size_t NIN);
	
	// Standard gradient minibatch
	void swallowGradient(const MAT& gradient);
	MAT& avgGradient();
	void clearGradient();
	void notifyFormChange(MATIND _newForm);
	// Buffer for actual inputs - if we are interested in statistics etc
	void updateBuffer(MAT& input);
	void updateModel();
	MAT& passOnNormalized(); // NOTE: make this MAT&& ? return value optimization of the compiler probably gets rid of this anyway
	inline uint32_t stillToCome() const { return stillToGo; }; // can't name function as variable...
	void clearBuffer();

	//friend class PhysicalLayer; // Physical layer orchestrates the normalization process
private:
	MAT& normalize(MAT& input) const;
	
	size_t stillToGo; 
	MATVEC batchBuffer; // Store input matrices over minibatch
	MAT gradientBuffer;
	fREAL eps = 1e-10;
	MAT mues; // mean
	MAT sigmas; // standard deviations
	size_t NIN;  // input dimensions
	size_t NOUT; //output dimensions
};

#endif