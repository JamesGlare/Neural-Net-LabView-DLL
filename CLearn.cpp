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
	filePath = "C:\\Jannes\\learnSamples\\";

	NNODES = new uint32_t[NHIDDEN];
	memcpy(NNODES, _NNODES, NHIDDEN*sizeof(uint32_t));
	this->init();
}
void CLearn::init() {

	// Matrix ( NNEXT, NPREV+1)
	inLayer = MAT(NNODES[0], NIN + 1); // add bias node 
	inVel = MAT(NNODES[0], NIN + 1);
	// initialize the inlayer
	inLayer.setRandom();
	inVel.setConstant(0);
	inLayer *= 1.0f / NIN;
	inLayer.rightCols(1).setConstant(0); // bias nodes

	if (NHIDDEN > 1) {
		hiddenLayers = MATVEC(NHIDDEN - 1);// there are one fewer weight matrices than node layers !!!
		hiddenVel = MATVEC(NHIDDEN - 1);
		for (uint32_t i = 0; i < NHIDDEN - 1; i++) {
			hiddenVel[i] = MAT(NNODES[i + 1], NNODES[i] + 1);
			hiddenVel[i].setConstant(0);
			hiddenLayers[i] = MAT(NNODES[i + 1], NNODES[i] + 1); // add bias node  MAT(NNODE, NNODE+1)
			hiddenLayers[i].setRandom();
			hiddenLayers[i] *= 1.0f / NNODES[i];
			hiddenLayers[i].rightCols(1).setConstant(0);
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
	outLayer.rightCols(1).setConstant(0); // initialize bias nodes
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
// initialize from file
int CLearn::initializeFromFiles() {
	//(1) Read structure file
	ifstream strucFile(filePath + "structure.dat");
	if (strucFile.is_open()) {
		strucFile >> NIN;
		strucFile >> NOUT;
		strucFile >> NHIDDEN;
		delete NNODES;

		NNODES = new uint32_t[NHIDDEN];
		for (int32_t i = 0; i < NHIDDEN; i++) {
			strucFile >> NNODES[i];
		}
	} else {
		return -1;
	}
	strucFile.close();
	// Now initialize this file
	//this->init(); 
 
	// (2) Read in matrix
	ifstream inFile(filePath + "in.dat");
	if (inFile.is_open()) {
		// column major order - inner loop must be columns
		for (size_t i = 0; i < inLayer.rows(); i++) {
			for (size_t j = 0; j < inLayer.cols(); j++) {
				inFile >> inLayer(i, j);
			}
		}
	}
	else {
		return -1;
	}
	inFile.close();
	// (3) read out matrix
	ifstream outFile(filePath + "out.dat");
	if (outFile.is_open()) {
		// column major order - inner loop must be columns
		for (size_t i = 0; i < outLayer.rows(); i++) {
			for (size_t j = 0; j < outLayer.cols(); j++) {
				outFile >> outLayer(i, j);
			}
		}
	}
	else {
		return -1;
	}
	outFile.close();

	// (4) read hidden matrices
	for (int32_t k = 0; k < NHIDDEN-1; k++) {
		ifstream hiddenFile(filePath + "hidden_" + to_string(k) + ".dat");
		if (hiddenFile.is_open()) {
			for (size_t i = 0; i < hiddenLayers[k].rows(); i++) {
				for (size_t j = 0; j < hiddenLayers[k].cols(); j++) {
					hiddenFile >> hiddenLayers[k](i, j);
				}
			}
		}
		else {
			return -1;
		}
		hiddenFile.close();
	}
	return 1;
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
/* Save the network to file.
*/
int CLearn::saveToFile() {
	// (1) open filestream - except hidden layers
	ofstream strucFile(filePath+"structure.dat");
	ofstream inMat(filePath + "in.dat");
	ofstream outMat(filePath + "out.dat");

	// (2) write structure of the network
	if (strucFile.is_open()) {
		strucFile << NIN << endl;
		strucFile << NOUT << endl;
		strucFile << NHIDDEN << endl;
		for (int32_t i = 0; i < NHIDDEN; i++) {
			strucFile << NNODES[i] << endl;
		}
	}
	else {
		return -1;
	}
	strucFile.close();

	//(3) write in matrix
	if (inMat.is_open()) {
		inMat << inLayer;
		inMat.close();
	}
	else {
		return -1;
	}
	
	// (4) write out matrix
	if (outMat.is_open()) {
		outMat << outLayer;
		outMat.close();
	}
	else {
		return -1;
	}
	
	//(5) open & write hidden layers
	for (int32_t i = 0; i < NHIDDEN-1; i++) {
		ofstream hiddenMat(filePath + "hidden_" + to_string(i) + ".dat");
		if (hiddenMat.is_open()) {
			hiddenMat << hiddenLayers[i];
			hiddenMat.close();
		}
		else {
			return -1;
		}
	}
	return 1;
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

fREAL CLearn::backProp(const MAT& in, MAT& dOut, learnPars pars) {
	// (1) Forward propagate through network, but save activations
	if (pars.conjugate == 1) {
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
	dOut = out; // copy the output

	return error;
}

MAT CLearn::ACT(const MAT& in) {
	// be careful with overloads here - it needs to be clear which function to use
	// only call other static functions.
	return in.unaryExpr(&ReLu);
}
MAT CLearn::DACT(const MAT& in) {
	return in.unaryExpr(&DReLu);
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