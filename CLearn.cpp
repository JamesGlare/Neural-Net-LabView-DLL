// CLearn.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "CLearn.h"

/*
 * CLearn functions 
 */

// Now, define the constructor etc
CLearn::CLearn(uint32_t _NIN, uint32_t _NOUT, uint32_t _NHIDDEN, uint32_t* const _NNODES) : NIN(_NIN), NOUT(_NOUT), NHIDDEN(_NHIDDEN) {
	// dont incorporate the bias node into NIN, NOUT etc
	// the network propagates from right to left, like standard matrix multiplication.
	if (NHIDDEN < 1) {
		return;
	}

	NNODES = new uint32_t[NHIDDEN];
	memcpy(NNODES, _NNODES, NHIDDEN*sizeof(uint32_t));

	// Matrix ( NNEXT, NPREV+1)
	inLayer = MAT(NNODES[0], NIN + 1); // add bias node 
	inVel = MAT(NNODES[0], NIN + 1);
	// initialize the inlayer
	inLayer.setRandom();
	inVel.setConstant(0);
	inLayer *= 1.0f/NIN;
	//inLayer.rightCols(1).setConstant(1); // bias nodes

	if (NHIDDEN > 1){
		hiddenLayers = MATVEC(NHIDDEN - 1);// there are one fewer weight matrices than node layers !!!
		hiddenVel = MATVEC(NHIDDEN - 1);
		for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
			hiddenVel[i] = MAT(NNODES[i + 1], NNODES[i] + 1);
			hiddenVel[i].setConstant(0);
			hiddenLayers[i] = MAT(NNODES[i + 1], NNODES[i] + 1); // add bias node  MAT(NNODE, NNODE+1)
			hiddenLayers[i].setRandom();
			hiddenLayers[i] *= 1.0f / NNODES[i];
			//hiddenLayers[i].rightCols(1).setConstant(1);
		}
	}
	else {
		hiddenVel = MATVEC();
		hiddenLayers = MATVEC(); // nonexistent
	}
	outLayer = MAT(NOUT, NNODES[NHIDDEN - 1] + 1); // add bias node
	outVel = MAT(NOUT, NNODES[NHIDDEN - 1] + 1);
	
	outLayer.setRandom();
	outLayer *= 1.0 / NNODES[NHIDDEN - 1];
	outVel.setConstant(0);

	// Activation storage matrices
	// The activation matrices have always dimensions
	// of outgoing weights
	inAct = MAT(NNODES[0], 1);
	inAct.setConstant(0);
	if (NHIDDEN > 1) {
		hiddenActs = MATVEC(NHIDDEN - 1); // this is the 
		for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
			hiddenActs[i] = MAT(NNODES[i + 1], 1);
			hiddenActs[i].setConstant(0);
		}
	}
	else {
		hiddenActs = MATVEC();
	}
	outAct = MAT(NOUT, 1);
	outAct.setConstant(0);

	// Delta storage matrices
	// the deltas need to have the same dimensionality as the activations
	outDelta = MAT(NOUT, 1);
	outDelta.setConstant(0);
	if (NHIDDEN > 1) {
		hiddenDeltas = MATVEC(NHIDDEN - 1);
		for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
			hiddenDeltas[i] = MAT(NNODES[i + 1], 1);
			hiddenDeltas[i].setConstant(0);
		}
	}
	else {
		hiddenDeltas = MATVEC(0);
	}
	inDelta = MAT(NNODES[0], 1);
	inDelta.setConstant(0);
}
// destructor
CLearn::~CLearn() {
	// delete nnode array
	delete NNODES;
	// resize saved matrices
	for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
		hiddenDeltas[i].resize(0, 0);
		hiddenVel[i].resize(0, 0);
		hiddenLayers[i].resize(0, 0);
		hiddenActs[i].resize(0, 0);
	}
}

// Careful, this function assumes a column vector.
void CLearn::appendOne(MAT& in) {
	in.conservativeResize(in.rows() + 1, in.cols()); // (NIN+1,1)
	in.bottomRows(1) = MAT(1, 1).setConstant(1); // bottomRows etc can be used as lvalue 
}
void CLearn::shrinkOne(MAT& in) {
	in.conservativeResize(in.rows() - 1, in.cols());
}
MAT CLearn::forProp(const MAT& in, bool saveActivations) {
	// size(in) = (NIN,1)
	MAT temp = appendOneInline(in);
	temp = inLayer*temp; // (NNODE_0, NIN+1) x (NIN+1, 1) = (NNODE_0, 1)
	if (saveActivations)
		inAct = temp;
	temp = ACT(temp); // (NNODE_0,1)
	// this should be automatically save if NHIDDEN ==1 
	for (uint32_t i = 0; i < NHIDDEN-1; i++) { // number of multiplications WITHIN the hidden layers -> NHIDDEN-1
		appendOne(temp); //(NODE[i]+1, 1)
		temp = hiddenLayers[i]*temp;  // (NNODE_(i+1), NNODE_i +1) x (NNODE_i +1,1) = (NNODE_(i+1),1)
		if (saveActivations)
			hiddenActs[i] = temp;
		temp = ACT(temp); //(NODE_(i+1), 1) 
	}
	appendOne(temp); //(NODE_(NHIDDEN-1) +1 , 1) if NHIDDEN==1 (NODE_0 +1, 1)
	temp = outLayer*temp; // (NOUT, NNODE_(NHIDDEN-1)+1) x (NNODE_(NHIDDEN-1)+1,1) = (NOUT,1)
	
	if (saveActivations) // save activation of output layer
		outAct = temp;

	return ACT(temp);
}
fREAL CLearn::l2Error(const MAT& deltaOut) {
	return 0.5f*cumSum(matNorm(deltaOut));
}
inline MAT CLearn::appendOneInline(const MAT& toAppend) {
	MAT temp = MAT(toAppend.rows() + 1, toAppend.cols()).setConstant(1);
	temp.topRows(toAppend.rows()) = toAppend;
	return temp;
}
fREAL CLearn::backProp(const MAT& in, MAT& dOut, learnPars pars) {
	// (1) Forward propagate through network, but save activations
	if (pars.nesterov == 1) {
		inLayer = inLayer - pars.gamma * inVel;
		for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
			hiddenLayers[i] = hiddenLayers[i] - pars.gamma*hiddenVel[i];
		}
		outLayer = outLayer - pars.gamma*outVel;
	}
	MAT out= forProp(in, true); 

	// (2) Compute deltas backwards
	// DSig(NOUT,1) * (NOUT,1) - (NOUT, 1) = (NOUT,1)
	outDelta = out - dOut; // deltaOut == outAct
	fREAL error = l2Error(outDelta);

	if (NHIDDEN > 1) {
		// delta_j = DSig(a_j)*sum_k w_kj*delta_k j in { 1, M }
		// DSig( NNODE_(NHIDDEN-1),1 ) * (NOUT, NNODE_(NHIDDEN-1)).T x(NOUT, 1)  = DSig( NNODE_(NHIDDEN-1),1 ) * (NNODE_(NHIDDEN-1) ,NOUT) x (NOUT,1) = (NNODE_(NHIDDEN-1),1)
		hiddenDeltas[NHIDDEN - 2] = DACT(hiddenActs[NHIDDEN - 2]).cwiseProduct(outLayer.leftCols(outLayer.cols() - 1).transpose()*outDelta);

		for (int i = NHIDDEN-3; i >=0; i--) {
			// DSig( NNODE_(i+1),1 ) * (NNODE_(i+2), NNODE_(i+1)).T x(NNODE_(i+2), 1)  = DSig( NNODE_(i+1),1 ) * (NNODE_(i+1),NNODE_(i+2)) x (NNODE_(i+2),1) = (NNODE_(i+1),1)
			hiddenDeltas[i] = DACT(hiddenActs[i]).cwiseProduct(hiddenLayers[i + 1].leftCols(hiddenLayers[i + 1].cols() - 1).transpose()*hiddenDeltas[i + 1]);
		}

		// DSig(NNODE_0,1) * (NNODE_1, NNODE_0).T x (NNode_1,1) = DSig(NNODE_0,1) * (NNODE_0, NNODE_1) x (NNode_1,1) = (NNODE_0,1)
		inDelta = DACT(inAct).cwiseProduct(hiddenLayers[0].leftCols(hiddenLayers[0].cols() - 1).transpose()*hiddenDeltas[0]);
	}
	else {
		// DSig( NNODE_0,1 ) * (NOUT, NNODE_0).T x(NOUT, 1)  = DSig( NNODE_0,1 ) * (NNODE_0,NOUT) x (NOUT,1) = (NNODE_0,1)
		inDelta = DACT(inAct).cwiseProduct(outLayer.leftCols(outLayer.cols() - 1).transpose()*outDelta);
	}
	//(3) Compute gradient
	fREAL discount = 1.0f - pars.lambda;
	if (NHIDDEN > 1) {
		MAT temp = appendOneInline(ACT(hiddenActs[NHIDDEN - 2])); //(NNODE_(NHIDDEN-1)+1, 1)
		// (NOUT, NNode_(NHIDDEN-1)+1) = (NOUT,1)x(1,NNODE_(NHIDDEN-1)+1) 
		outVel = pars.gamma*outVel + pars.eta*outDelta*temp.transpose();
		outLayer = discount*outLayer - outVel;

		for (int i = NHIDDEN - 2; i > 0; i--) { // only if NHIDDEN> 2
			temp = appendOneInline(ACT(hiddenActs[i-1])); // 
			// (NNODE_(i+1), NNODE_i +1) = (NNODE_(i+1),1 ) x (1,NNODE_i +1 ) = (NNODE_(i+1), NNODE_i +1 )
			hiddenVel[i] = pars.gamma*hiddenVel[i] + pars.eta*hiddenDeltas[i] * temp.transpose();
			hiddenLayers[i] = discount*hiddenLayers[i] - hiddenVel[i];
		}
		
		temp = appendOneInline(ACT(inAct)); //(NNODE_0 +1, 1)
		// (NODE_1, NNODE_0 +1) = (NNODE_1, 1) x (1,NNODE_0 +1)
		hiddenVel[0] = pars.gamma * hiddenVel[0] + pars.eta*hiddenDeltas[0] * temp.transpose();
		hiddenLayers[0] = discount*hiddenLayers[0] - hiddenVel[0];
		
	}
	else {
		MAT temp = appendOneInline(ACT(inAct)); //(NNODE_0+1,1)
		outVel = pars.gamma*outVel + pars.eta*outDelta*temp.transpose();
		outLayer = discount*outLayer - outVel;
	}
	MAT temp = appendOneInline(in); // redeclare
	// (NNODE_0, NIN+1) = (NNODE_0,1)x(1,NIN+1)
	inVel = pars.gamma * inVel + pars.eta*inDelta*temp.transpose();
	inLayer = discount*inLayer - inVel;

	return error;
}
fREAL CLearn::cumSum(const MAT& in) {
	return in.sum();
}
// Activation functions
inline fREAL CLearn::Tanh(fREAL f) {
	return std::tanh(f);
}
inline fREAL CLearn::Sig(fREAL f) {
	return 1.0f / (1.0f + std::exp(-1.0f*f));
}
inline fREAL CLearn::DSig(fREAL f) {
	return Sig(f)*(1.0f-Sig(f));
}
inline fREAL CLearn::ReLu(fREAL f) {
	return std::log(1.0f + std::exp(f));
}
inline fREAL CLearn::DReLu(fREAL f) {
	return Sig(f);
}
fREAL inline CLearn::norm(fREAL f) {
	return f*f;
}
MAT CLearn::matNorm(const MAT& in) {
	return in.unaryExpr(&CLearn::norm);
}
MAT CLearn::ACT(const MAT& in) {
	// be careful with overloads here - it needs to be clear which function to use
	// only call other static functions.
	return in.unaryExpr(&CLearn::ReLu);
}
MAT CLearn::DACT(const MAT& in) {
	return in.unaryExpr(&CLearn::DReLu);
}
uint32_t CLearn::get_NIN() {
	return NIN;
}
uint32_t CLearn::get_NOUT() {
	return NOUT;
}
uint32_t CLearn::get_NNODE(uint32_t i) {
	return NNODES[i];
}
uint32_t CLearn::get_NHIDDEN() {
	return NHIDDEN;
}

void CLearn::copy_inWeights(fREAL* const weights) {
	memcpy(weights, this->inLayer.data(), this->inLayer.size()*sizeof(fREAL));
}
void CLearn::copy_hiddenWeights(fREAL* const weights, uint32_t layer) {
	memcpy(weights, this->hiddenLayers[layer].data(), this->hiddenLayers[layer].size() * sizeof(fREAL));
}
void CLearn::copy_outWeights(fREAL* const weights) {
	memcpy(weights, this->outLayer.data(), this->outLayer.size() * sizeof(fREAL));
}


/*
 * DLL Functions
 */

__declspec(dllexport) int __stdcall initClass(CLearn** ptr, uint32_t NIN, uint32_t NOUT, uint32_t NHIDDEN, uint32_t* const NNODES) {
	*ptr = new CLearn(NIN, NOUT, NHIDDEN, NNODES);
	return 1;
}

__declspec(dllexport) int __stdcall callClass(CLearn* ptr, uint8_t* const SLM, uint8_t* const out, int32_t kx, int32_t ky, double val) {
	int sx = 100;
	int sy = 100;
	for (int i = 0; i < sx; i++) {
		for (int j = 0; j < sy; j++) {
			SLM[i*sy+j] = 127*val*(sin((double)(ky*i)/sy)*sin((double)(kx*j)/sx)+1);
		}
	}
	return 1;
}
__declspec(dllexport) fREAL __stdcall forward(CLearn* ptr, fREAL* const SLM, fREAL* const image, fREAL* const eta, fREAL* const metaEta, fREAL* const gamma, fREAL* const lambda, uint32_t* const nesterov, int* const validate) {
	// remember not to use .size() on a matrix. It's not numpy ;)
	learnPars pars = {*eta, *metaEta, *gamma, *lambda, *nesterov};

	MAT in = MATMAP(SLM, ptr->get_NIN(),1); // (NIN, 1) Matrix
	MAT dOut = MATMAP(image, ptr->get_NOUT(), 1);
	fREAL error = 0.0;
	if (*validate == 0) {
		error = ptr->backProp(in, dOut, pars); // overwrites dOut with prediction
	}
	else {
		dOut = ptr->forProp(in, false);
	}
	copyToOut(dOut.data(), image, ptr->get_NOUT()); // copy data into pointer
	//MAT SLMSub = MATMAP(SLM, ptr->get_NIN(), 1);
	//SLMSub.setConstant(128);
	//copyToOut(SLMSub.data(), SLM, ptr->get_NIN()); // copy data into pointer

	return error;
}

__declspec(dllexport) void __stdcall getINWeights(CLearn* ptr, fREAL* const inWeights) {
	ptr->copy_inWeights(inWeights);
}
__declspec(dllexport) void __stdcall getHiddenWeights(CLearn* ptr, fREAL* const hiddenWeights, uint32_t layer) {
	ptr->copy_hiddenWeights(hiddenWeights, layer);
}
__declspec(dllexport) void __stdcall getOutWeights(CLearn* ptr, fREAL* const outWeights) {
	ptr->copy_outWeights(outWeights);
}
__declspec(dllexport) int __stdcall terminateClass(CLearn* ptr) {
	delete (ptr);
	return 1;
}
