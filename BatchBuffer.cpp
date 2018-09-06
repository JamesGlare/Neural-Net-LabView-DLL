#include "stdafx.h"
#include "BatchBuffer.h"


BatchBuffer::BatchBuffer(MATIND _layerInd, size_t _NOUT, size_t _NIN) :  NIN(_NIN), NOUT(_NOUT), stillToGo(0){
	mues = MAT(_NIN, 1); // each input has an independent offset...
	sigmas= MAT(_NIN, 1);  // ... and offset.
	gradientBuffer = MAT(_layerInd.rows, _layerInd.cols);
	// init matrices
	gradientBuffer.setZero();
	mues.setZero();
	sigmas.setOnes();

}
void BatchBuffer::notifyFormChange(MATIND _newForm) {
	gradientBuffer.resize(_newForm.rows, _newForm.cols);
}
void BatchBuffer::swallowGradient(const MAT& gradient) {
	gradientBuffer.noalias() += gradient;
	stillToGo++;
}

MAT& BatchBuffer::avgGradient() {
	if(stillToGo > 0)
		gradientBuffer /= stillToGo;
		return gradientBuffer;
}
void BatchBuffer::clearGradient() {
	gradientBuffer.setZero();
	stillToGo = 0;
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