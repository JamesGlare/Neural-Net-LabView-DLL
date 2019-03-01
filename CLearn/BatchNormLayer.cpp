#include "stdafx.h"
#include "BatchNormLayer.h"

BatchNormLayer::BatchNormLayer(size_t NIN) : gamma(NIN, 1), beta(NIN, 1), currMean(NIN,1), currSigma(NIN,1),
gamma_stepper(MATIND{ NIN,1 }), beta_stepper(MATIND{ NIN,1 }), buffer(NIN, NIN), CNetLayer(NIN, NIN, actfunc_t::NONE) {

	init();

}

BatchNormLayer::BatchNormLayer(CNetLayer & lower) : gamma(lower.getNOUT(), 1), beta(lower.getNOUT(), 1), currMean(lower.getNOUT(), 1), currSigma(lower.getNOUT(), 1),
gamma_stepper(MATIND{ lower.getNOUT(),1 }), beta_stepper(MATIND{ lower.getNOUT(),1 }), buffer(lower.getNOUT(), lower.getNOUT()), 
CNetLayer(lower.getNOUT(), actfunc_t::NONE, lower) {

	init();
}

BatchNormLayer::~BatchNormLayer() {
}

layer_t BatchNormLayer::whoAmI() const {
	return layer_t::batchNorm;
}

void BatchNormLayer::forProp(MAT & in, bool training, bool recursive) {
	if (training) {
		buffer.updateBuffer(in); // swallow this

		// store the x-hats
		actSave = in; // x^
		// Now transform using the internal matrices
		in = gamma.cwiseProduct(in);
		in += beta;
		if (recursive && getHierachy() != hierarchy_t::output)
			above->forProp(in, true, true);

	} else {
		// This is slightly weird
		// they want us to compute 
		in = gamma.cwiseQuotient(currSigma).cwiseProduct(in);
		in += beta - gamma.cwiseProduct(currMean).cwiseQuotient(currSigma);
		if (recursive && getHierachy() != hierarchy_t::output)
			above->forProp(in, false, true);
	}
}

void BatchNormLayer::backPropDelta(MAT & delta, bool recursive) {
	deltaSave = delta;
	delta.cwiseProduct(gamma);
	if (recursive && getHierachy() != hierarchy_t::input)
		below->backPropDelta(delta, true);
}

void BatchNormLayer::applyUpdate(const learnPars & pars, MAT & input, bool recursive) {
}

void BatchNormLayer::normalize(MAT & in){
	in -= currMean;
	in.cwiseQuotient(currSigma);
}

void BatchNormLayer::init(){
	gamma.setOnes();
	beta.setZero();
	currMean.setZero();
	currSigma.setOnes();
}
