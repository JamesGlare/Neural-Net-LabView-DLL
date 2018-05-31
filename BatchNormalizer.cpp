#include "stdafx.h"
#include "BatchNormalizer.h"

BatchNormalizer::BatchNormalizer(size_t _NIN) :  NIN(_NIN), stillToGo(0){
	mues = MAT(NIN, 1); // each input has an independent offset...
	sigmas= MAT(NIN, 1);  // ... and offset.
	gammas = MAT(NIN, 1);
	betas = MAT(NIN, 1);
	mues.setConstant(0);
	sigmas.setConstant(1);
	gammas.setConstant(1);
	betas.setConstant(0);
}

void BatchNormalizer::updateBuffer(MAT& input) {
		this->batchBuffer.push_back(input);
}

void BatchNormalizer::updateModel() {
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

void BatchNormalizer::trainParameters(const MAT& dY, const learnPars& pars) {
	size_t batchSize = batchBuffer.size();
	for (size_t i = 0; i < batchSize; i++) {
		//gammas -= pars.eta*dY.cwiseProduct();
	}
}

void BatchNormalizer::setToAverage() {
	gammas.setConstant(1);
	gammas = gammas.cwiseQuotient(sigmas);
	betas = -mues;
}

MAT& BatchNormalizer::normalize(MAT& input) const {
	input -= mues; // subtract mean
	input = input.cwiseQuotient( (sigmas));
	return input;
}

MAT& BatchNormalizer::deNormalize(MAT& input) const {
	input.cwiseProduct(gammas); // in the article: gamma == sigma
	input += betas; // beta == mues
	return input;
}

MAT& BatchNormalizer::passOnNormalized() {
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

void BatchNormalizer::clearBuffer() {
	batchBuffer.clear();
}