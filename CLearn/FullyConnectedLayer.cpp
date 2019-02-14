#include "stdafx.h"
#include "FullyConnectedLayer.h"

// constructor - for input layers
FullyConnectedLayer::FullyConnectedLayer(size_t _NOUT, size_t _NIN, actfunc_t type) :
	PhysicalLayer(_NOUT, _NIN, type, MATIND{ _NOUT, _NIN }, MATIND{ _NOUT, _NIN }, MATIND{ _NOUT, 1 }, MATIND{ _NOUT, _NIN }) {
	// declare matrix
	init();
}
// constructor - for hidden layers and output layers
FullyConnectedLayer::FullyConnectedLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower) :
	PhysicalLayer(_NOUT, type, MATIND{ _NOUT, lower.getNOUT() }, MATIND{ _NOUT, lower.getNOUT() }, MATIND{ _NOUT, 1 }, MATIND{_NOUT, lower.getNOUT()}, lower) {
	// layer and velocity matrices
	init();
}
// Destructor
FullyConnectedLayer::~FullyConnectedLayer() {}

layer_t FullyConnectedLayer::whoAmI() const {
	return layer_t::fullyConnected;
}
// init
void FullyConnectedLayer::init() {

	//W /= getNIN(); //sqrt(sqrt(getNOUT()*getNIN())); // ... but small.
	//b.setZero(); // set bias terms zero

	assert(actSave.rows() == getNOUT());
	assert(deltaSave.rows() == getNOUT());
	assert(W.rows() == getNOUT());
	assert(W.cols() == getNIN());
}

uint32_t FullyConnectedLayer::getFeatures() const {
	return 1;
}

// weight normalization reparametrize
void FullyConnectedLayer::wnorm_setW() {
	/*MAT inversV = inversVNorm();
	for (size_t i = 0; i < NOUT; i++) {
	layer.leftCols(NIN).row(i) = G(i,0)*inversV(i,0)*V.leftCols(NIN).row(i);
	}
	layer.rightCols(1) = V.rightCols(1); // keep bias weights
	*/
	W = V.cwiseProduct(VInversNorm);
	W = W.cwiseProduct(G.replicate(1, getNIN()));
}

void FullyConnectedLayer::wnorm_initV() {
	V = W;
	//V.setRandom();
	//V /= sqrt(getNIN());
	//normalizeV();
}

void FullyConnectedLayer::wnorm_normalizeV() {
	for (size_t i = 0; i < getNOUT(); ++i) {
		V.row(i) /= normSum(V.row(i));
	}
}
/* take cwiseProduct with this matrix to obtain X/||V|| or V/||V||^2
*/
void FullyConnectedLayer::wnorm_inversVNorm() {
	//MAT out(NOUT, NIN+1);
	VInversNorm.setOnes();
	//MAT oneRow = MAT::Constant(1, NIN,1.0f);
	for (size_t i = 0; i < getNOUT(); ++i) {
		VInversNorm.row(i) /= normSum(V.row(i)); //
	}
}

void FullyConnectedLayer::wnorm_initG() {
	for (size_t i = 0; i < getNOUT(); ++i) {
		G(i, 0) = normSum(W.leftCols(getNIN()).row(i));
	}
	//G.setOnes();
}

MAT FullyConnectedLayer::wnorm_gGrad(const MAT& grad) {
	// = MAT(NOUT, 1);  //(NOUT, NIN)
	MAT out = grad.cwiseProduct(VInversNorm.cwiseProduct(V)); //(NOUT, NIN)

	return out.rowwise().sum(); //(NOUT,1)
}

MAT FullyConnectedLayer::wnorm_vGrad(const MAT& grad, MAT& ggrad) {
	//MAT out(NOUT, NIN+1);
	MAT gRep = G.replicate(1, getNIN());
	MAT out = grad.cwiseProduct(VInversNorm).cwiseProduct(gRep);
	out -= gRep.cwiseProduct(VInversNorm.unaryExpr(&norm)).cwiseProduct(ggrad.replicate(1, getNIN())).cwiseProduct(V); // (NOUT, NIN)
	/*MAT inversV = inversVNorm();
	   for (size_t i = 0; i < NOUT; i++) {
	 // (1) multiply rows of grad with G's
	out.leftCols(NIN).row(i) *= G(i,0)*inversV(i,0);
	// (2) subtract
	out.leftCols(NIN).row(i) -= G(i, 0)*ggrad(i, 0)*V.leftCols(NIN).row(i)* (inversV(i, 0)*inversV(i, 0));
	}*/
	// we leave the rightCols(1) part of out untouched - these are the gradients for the bias, which get optimized in the normal way
	return out;
}

MAT FullyConnectedLayer::b_grad() {
	return deltaSave;
}

void FullyConnectedLayer::forProp(MAT& inBelow, bool training, bool recursive) {

	if (training) {
		/* normal training forward pass
		*/
		// Eigen assumes aliasing by default for matrix products A*B type situations
		actSave.noalias() = W*inBelow + b; // save the activations before non-linearity
		inBelow = move(getACT());
		if (recursive&& getHierachy() != hierarchy_t::output)
			above->forProp(inBelow, true, true);
		
	} else {
		/* Non-training forward pass
		*/
		MAT temp;
		// Eigen assumes aliasing by default for matrix products A*B type situations
		temp.noalias() = (W*inBelow + b).unaryExpr(act);
		inBelow = move(temp);
		if (recursive && getHierachy() != hierarchy_t::output)
			above->forProp(inBelow, false, true);
		
	}
}

void FullyConnectedLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	//DACT(inAct).cwiseProduct(hiddenLayers[0].leftCols(hiddenLayers[0].cols() - 1).transpose()*hiddenDeltas[0]);
	deltaSave = deltaAbove;

	if (getHierachy() != hierarchy_t::input) {
		if (getHierachy() == hierarchy_t::output)
			deltaAbove = deltaAbove.cwiseProduct(this->getDACT());

		deltaAbove.noalias() = (below->getDACT()).cwiseProduct(W.transpose() * deltaSave); // (NIN,1) cw* (NOUT, NIN).T x (NOUT, 1) = (NIN,1) cw* (NIN, 1) = (NIN,1) 
		if (recursive)
			below->backPropDelta(deltaAbove, true);
	}
}
/* Same dimensionality as layer.
*/
MAT FullyConnectedLayer::w_grad(MAT& input) {
	if (getHierachy() == hierarchy_t::input) {
		return deltaSave*(input.transpose()); //(NOUT, 1) x (NIN+1,1).T = (NOUT, NIN+1)
	}
	else {
		//if (kappa > 0.0f) {
		//	MAT temp = appendOneInline(below->getACT()).transpose();
		//	return (deltaSave + kappa*getDACT())*temp;
		//} else { // if no l2-reg applied, don't even store the temp matrix
		return deltaSave*((below->getACT()).transpose());
		//}

	}
}
/* Implementation of spectral norm functions
*/
void FullyConnectedLayer::snorm_setW() {
	W = W_temp/spectralNorm(W_temp, u1, v1);
}

void FullyConnectedLayer::snorm_updateUVs() {
	updateSingularVectors(W_temp, u1, v1, 1);
}

MAT FullyConnectedLayer::snorm_dWt(MAT& grad) {
	// This is actual work
	if(lambdaCount>0)
		grad -= lambdaBatch/lambdaCount*(u1*v1.transpose());
	
	grad /= spectralNorm(W_temp, u1, v1);
	lambdaBatch = 0; // reset this
	lambdaCount = 0;
	return grad;
}

void FullyConnectedLayer::saveToFile(ostream& os) const {
	// say which and if you're using a normalization
	os << spectralNormMode << " " << weightNormMode << endl;

	MAT temp = W;
	temp.resize(W.size(), 1);
	os << temp<<endl;
	os << b << endl;
	if (spectralNormMode) {
		temp = W_temp;
		temp.resize(W_temp.size(), 1);
		os << temp << endl;
		os << u1 << endl;
		os << v1 << endl;
	}
	if (weightNormMode) {
		temp = V;
		temp.resize(V.size(), 1);
		os << V<< endl;
		temp = G;
		temp.resize(G.size(), 1);
		os << temp << endl;
	}
	os << endl;
}

void FullyConnectedLayer::loadFromFile(ifstream& in) {

	W = MAT(getNOUT()*getNIN(), 1);    // initialize as a column vector
	W_temp = MAT(getNOUT(),getNIN());
	u1 = MAT(getNOUT(), 1);
	v1 = MAT(getNIN(), 1);
	V = MAT(getNOUT(), getNIN());
	G = MAT(getNOUT(), 1);
	b = MAT(getNOUT(), 1);
	W_temp.setZero();
	u1.setZero();
	v1.setZero();
	b.setZero();
	V.setZero();
	G.setZero();
	w_stepper.reset();
	b_stepper.reset();

	/*for (size_t j = 0; j< getNIN()+1; ++j) {
	for (size_t i = 0; i < getNOUT(); ++i) {
	in >> layer(i,j);
	}
	}*/
	in >> spectralNormMode;
	in >> weightNormMode; 
	for (size_t i = 0; i < W.size(); ++i) {
		in >> W(i, 0);
	}
	W.resize(getNOUT(), getNIN());
	for (size_t i = 0; i < b.size(); ++i) {
		in >> b(i, 0);
	}
	if (spectralNormMode) {
		W_temp.resize(W_temp.size(), 1);
		for (size_t i = 0; i < W_temp.size(); ++i) {
			in >> W_temp(i,0);
		}
		W_temp.resize(getNOUT(), getNIN());
		for (size_t i = 0; i < u1.size(); ++i) {
			in >> u1(i, 0);
		}
		for (size_t i = 0; i <v1.size(); ++i) {
			in >> v1(i, 0);
		}
	} 
	if (weightNormMode) {
		V.resize(V.size(), 1);
		for (size_t i = 0; i < V.size(); ++i) {
			in >> V(i,0);
		}
		V.resize(getNOUT(), getNIN());
		for (size_t i = 0; i < G.size(); ++i) {
			in >> G(i, 0);
		}
	}
}