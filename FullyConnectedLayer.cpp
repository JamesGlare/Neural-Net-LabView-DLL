#include "stdafx.h"
#include "FullyConnectedLayer.h"

// constructor - for input layers
FullyConnectedLayer::FullyConnectedLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : 
	PhysicalLayer(_NOUT, _NIN, type, MATIND{ _NOUT, _NIN + 1 }, MATIND{ _NOUT, _NIN +1 }, MATIND{ _NOUT, 1 }) {
	 // declare matrix
	init();
}
// constructor - for hidden layers and output layers
FullyConnectedLayer::FullyConnectedLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower) : 
	PhysicalLayer(_NOUT, type, MATIND{ _NOUT, lower.getNOUT() + 1 }, MATIND{ _NOUT, lower.getNOUT() +1 }, MATIND{ _NOUT, 1 },lower) {
	// layer and velocity matrices
	init();
}
// Destructor
FullyConnectedLayer::~FullyConnectedLayer() {
	layer.resize(0, 0);
}

layer_t FullyConnectedLayer::whoAmI() const {
	return layer_t::fullyConnected;
}
// init
void FullyConnectedLayer::init() {


	fREAL max = 1.0f / NIN;
	fREAL min = -1.0f / NIN;
	layer *= (max-min)/2.0f ; // (-max, max)

	//layer = (layer.array() + (max+min)/2.0f).matrix(); // (min,  2+min)
	assert(actSave.rows() == NOUT);
	assert(deltaSave.rows() == NOUT);
	assert(layer.rows() == NOUT);
	assert(layer.cols() == NIN + 1);
}
// weight normalization reparametrize
void FullyConnectedLayer::updateW() {
	/*MAT inversV = inversVNorm();
	for (size_t i = 0; i < NOUT; i++) {
		layer.leftCols(NIN).row(i) = G(i,0)*inversV(i,0)*V.leftCols(NIN).row(i);
	}
	layer.rightCols(1) = V.rightCols(1); // keep bias weights
	*/
	layer = V.cwiseProduct(inversVNorm());
	layer.leftCols(NIN) = layer.leftCols(NIN).cwiseProduct(G.replicate(1, NIN ));
}

void FullyConnectedLayer::initV() {
	V = layer;
	V.setRandom();
	//normalizeV();
}
void FullyConnectedLayer::normalizeV() {
	for (size_t i = 0; i < NOUT; i++) {
		V.leftCols(NIN).row(i) /= normSum(V.leftCols(NIN).row(i));
	}
}
/* take cwiseProduct with this matrix to obtain X/||V|| or V/||V||^2
*/
MAT FullyConnectedLayer::inversVNorm() {
	MAT out(NOUT, NIN+1);
	out.setOnes();
	//MAT oneRow = MAT::Constant(1, NIN,1.0f);
	for (size_t i = 0; i < NOUT; i++) {
		out.leftCols(NIN).row(i) /=  normSum(V.leftCols(NIN).row(i)); //
	}
	return out;
}
void FullyConnectedLayer::initG() {
//	for (size_t i = 0; i < NOUT; i++) {
//		G(i,0) = normSum(layer.leftCols(NIN).row(i));
//	}
	G.setOnes();
}
MAT FullyConnectedLayer::gGrad(MAT& grad) {
	// = MAT(NOUT, 1);  //(NOUT, NIN)
	MAT out=grad.leftCols(NIN).cwiseProduct( (inversVNorm().cwiseProduct(V)).leftCols(NIN) ); //(NOUT, NIN)
	
	return {std::move(out.rowwise().sum())}; //(NOUT,1)
}
MAT FullyConnectedLayer::vGrad(MAT& grad, MAT& ggrad) {
	MAT temp = inversVNorm(); //(NOUT, NIN)
	MAT out(NOUT, NIN+1);
	MAT gRep = G.replicate(1, NIN + 1);
	out = grad.cwiseProduct(temp).cwiseProduct(gRep);
	out.noalias() -= gRep.cwiseProduct(temp.unaryExpr(&norm)).cwiseProduct(ggrad.replicate(1, NIN + 1)).cwiseProduct(V); // (NOUT, NIN)
	out.rightCols(1) = grad.rightCols(1);
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

void FullyConnectedLayer::forProp(MAT& inBelow, bool training, bool recursive) {

	if (training) {
		/* normal training forward pass
		*/ 
	
		//appendOne(inBelow);
		// Eigen assumes aliasing by default for matrix products A*B type situations
		actSave.noalias() = layer*appendOneInline(inBelow); // save the activations before non-linearity
		if (hierarchy != hierarchy_t::output) {
			inBelow = actSave.unaryExpr(act); 
			if(recursive)
				above->forProp(inBelow, true, true);
		} else {
			inBelow = actSave;
		}
	} else {
		/* Non-training forward pass
		*/
		MAT temp;
		if (hierarchy != hierarchy_t::output) {
			// Eigen assumes aliasing by default for matrix products A*B type situations
			temp.noalias() = (layer*appendOneInline(inBelow)).unaryExpr(act);
			inBelow = temp;
			if(recursive)
				above->forProp(inBelow, false, true);
		} else {
			temp.noalias() = layer*appendOneInline(inBelow);
			inBelow = temp;
		}
	}
}

void FullyConnectedLayer::backPropDelta(MAT& deltaAbove, bool recursive) {
	//DACT(inAct).cwiseProduct(hiddenLayers[0].leftCols(hiddenLayers[0].cols() - 1).transpose()*hiddenDeltas[0]);
	deltaSave = deltaAbove;

	if (hierarchy != hierarchy_t::input) {
		deltaAbove.noalias() = (below->getDACT()).cwiseProduct((layer.leftCols(NIN)).transpose() * deltaSave); // (NIN,1) cw* (NOUT, NIN).T x (NOUT, 1) = (NIN,1) cw* (NIN, 1) = (NIN,1) 
		if(recursive)
			below->backPropDelta(deltaAbove, true);
	}
}
/* Same dimensionality as layer.
*/
MAT FullyConnectedLayer::grad(MAT& input) {
	if (hierarchy == hierarchy_t::input) {
			// VC does not perform RVO for some reason :/
		return {std::move(deltaSave*appendOneInline(input).transpose())}; //(NOUT, 1) x (NIN+1,1).T = (NOUT, NIN+1)
	}
	else {
		return {std::move(deltaSave * appendOneInline(below->getACT()).transpose()) };
	}
}

void FullyConnectedLayer::saveToFile(ostream& os) const {
	os<< NOUT << "\t" << NIN <<endl; // header line 2
	os << layer;
}
void FullyConnectedLayer::loadFromFile(ifstream& in) {
	in >> NOUT;
	in >> NIN;
	
	layer = MAT(NOUT, NIN + 1);  
	V = MAT(NOUT, NIN + 1);
	G = MAT(NOUT, 1);
	V.setZero();
	G.setZero();

	for (size_t i = 0; i < NOUT; i++) {
		for (size_t j = 0; j < NIN+1; j++) {
			in >> layer(i, j);
		}
	}
}