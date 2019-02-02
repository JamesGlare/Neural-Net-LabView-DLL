#include "stdafx.h"
#include "BatchBuffer.h"


BatchBuffer::BatchBuffer(MATIND _WInd, size_t _NOUT, size_t _NIN) :  NIN(_NIN), NOUT(_NOUT){
	mues = MAT(_NIN, 1); // each input has an independent offset...
	sigmas= MAT(_NIN, 1);  // ... and offset.
	// A correctly-shaped zero-gradient.
	nullGradient = MAT(_WInd.rows, _WInd.cols);
	nullGradient.setZero(); 

	gradientBuffer = MATVEC();
	mues.setZero();
	sigmas.setOnes();

}

void BatchBuffer::swallowGradient(const MAT& grad) {
	if(gradientBuffer.size() < MAX_SIZE)
		gradientBuffer.push_back(grad);
}
// Average over mini batch
MAT BatchBuffer::avgGradient() {
	size_t bufferSize = gradientBuffer.size();
	MAT grad;

	if (bufferSize > 0) {
		grad = gradientBuffer[0];
	} else {
		return nullGradient;
	}
	for (size_t i = 1; i < bufferSize; ++i) {
		grad += gradientBuffer[i];
	}
	return grad / bufferSize;
}
// RMS needed for rms prop...
MAT BatchBuffer::rmsGradient() {
	size_t bufferSize = gradientBuffer.size();
	MAT grad;

	if (bufferSize > 0) {
		grad = gradientBuffer[0].unaryExpr(&norm);
	} else {
		return nullGradient;
	}
	for (size_t i = 1; i < bufferSize; ++i) {
		grad += gradientBuffer[i].unaryExpr(&norm);
	}
	return grad / bufferSize;
}

void BatchBuffer::clearGradients() {
	gradientBuffer.clear();
}
void BatchBuffer::updateBuffer(MAT& input) {
		this->batchBuffer.push_back(input);
}
void BatchBuffer::updateModel() {
	size_t batchSize = batchBuffer.size();
	// (1) Calculate mean of all inputs
	if (batchSize > 0) {
		mues = batchBuffer[1];
	}
	for (size_t i = 1; i < batchSize; i++) {
		mues += batchBuffer[i];
	}
	mues /= batchSize; // normalize by batch size

	// (2) Now, calculate the variance of all activations in the batch
	if (batchSize > 0) {
		sigmas = (batchBuffer[1] - mues).unaryExpr(&norm);
	}
	for (size_t i = 1; i < batchSize; i++) {
		sigmas += (batchBuffer[i] - mues).unaryExpr(&norm);
	}
	sigmas /= batchSize; // normalize by batch size
	sigmas += MAT::Constant(NIN, 1, eps); // numerical stability
	sigmas.unaryExpr(&sqroot); // to make it truly a
	// (3) Save the number of batches you have in this counter variable
	stillToGo = batchSize;
}



MAT& BatchBuffer::normalize(MAT& input) const {
	input -= mues; // subtract mean
	input = input.cwiseQuotient( (sigmas));
	return input;
}


MAT& BatchBuffer::passOnNormalized() {
	size_t batchSize= batchBuffer.size();
	if (stillToGo >0){ // if stillToGo ==0 => problem.
		if (batchSize > 0) { // to avoid crashed, if nrPassedOn
			stillToGo--;
			return normalize(batchBuffer[stillToGo - 1]);
		} else { // error -> we shouldn't be here
			MAT lvalRef = MAT::Constant(NIN, 1, -1);
			return lvalRef;
		}
	} else {
		clearBuffer();
	}
}

void BatchBuffer::clearBuffer() {
	batchBuffer.clear();
}