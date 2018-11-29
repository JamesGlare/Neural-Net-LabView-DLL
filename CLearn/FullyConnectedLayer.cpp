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


	//fREAL max = 1.0f / getNIN();
	//fREAL min = -1.0f / getNIN();
	//layer += MAT::Constant(layer.rows(), layer.cols(), 1.0f); // make everything positive
	layer /= sqrt(getNIN()); //sqrt(sqrt(getNOUT()*getNIN())); // ... but small.

	assert(actSave.rows() == getNOUT());
	assert(deltaSave.rows() == getNOUT());
	assert(layer.rows() == getNOUT());
	assert(layer.cols() == getNIN() + 1);
}
// weight normalization reparametrize
void FullyConnectedLayer::updateW() {
	/*MAT inversV = inversVNorm();
	for (size_t i = 0; i < NOUT; i++) {
		layer.leftCols(NIN).row(i) = G(i,0)*inversV(i,0)*V.leftCols(NIN).row(i);
	}
	layer.rightCols(1) = V.rightCols(1); // keep bias weights
	*/
	layer = V.cwiseProduct(VInversNorm);
	layer.leftCols(getNIN()) = layer.leftCols(getNIN()).cwiseProduct(G.replicate(1, getNIN() ));
}

void FullyConnectedLayer::initV() {
	//V = layer;
	V.setRandom();
	V /= sqrt(getNIN());
	//normalizeV();
}
void FullyConnectedLayer::normalizeV() {
	for (size_t i = 0; i < getNOUT(); ++i) {
		V.leftCols(getNIN()).row(i) /= normSum(V.leftCols(getNIN()).row(i));
	}
}
/* take cwiseProduct with this matrix to obtain X/||V|| or V/||V||^2
*/
void FullyConnectedLayer::inversVNorm() {
	//MAT out(NOUT, NIN+1);
	VInversNorm.setOnes();
	//MAT oneRow = MAT::Constant(1, NIN,1.0f);
	for (size_t i = 0; i < getNOUT(); ++i) {
		VInversNorm.leftCols(getNIN()).row(i) /=  normSum(V.leftCols(getNIN()).row(i)); //
	}
}
void FullyConnectedLayer::initG() {
	for (size_t i = 0; i < getNOUT(); ++i) {
		G(i,0) = normSum(layer.leftCols(getNIN()).row(i));
	}
	//G.setOnes();
}
MAT FullyConnectedLayer::gGrad(const MAT& grad) {
	// = MAT(NOUT, 1);  //(NOUT, NIN)
	MAT out=grad.leftCols(getNIN()).cwiseProduct( (VInversNorm.cwiseProduct(V)).leftCols(getNIN()) ); //(NOUT, NIN)
	
	return out.rowwise().sum(); //(NOUT,1)
}
MAT FullyConnectedLayer::vGrad(const MAT& grad, MAT& ggrad) {
	//MAT out(NOUT, NIN+1);
	MAT gRep = G.replicate(1, getNIN() + 1);
	MAT out = grad.cwiseProduct(VInversNorm).cwiseProduct(gRep);
	out -= gRep.cwiseProduct(VInversNorm.unaryExpr(&norm)).cwiseProduct(ggrad.replicate(1, getNIN() + 1)).cwiseProduct(V); // (NOUT, NIN)
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
		if (getHierachy() != hierarchy_t::output) {
			inBelow = getACT(); 
			if(recursive)
				above->forProp(inBelow, true, true);
		} else {
				inBelow = actSave;
		}
	} else {
		/* Non-training forward pass
		*/
		MAT temp;
		if (getHierachy() != hierarchy_t::output) {
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

	if (getHierachy() != hierarchy_t::input) {
		deltaAbove.noalias() = (below->getDACT()).cwiseProduct((layer.leftCols(getNIN())).transpose() * deltaSave); // (NIN,1) cw* (NOUT, NIN).T x (NOUT, 1) = (NIN,1) cw* (NIN, 1) = (NIN,1) 
		if(recursive)
			below->backPropDelta(deltaAbove, true);
	}
}
/* Same dimensionality as layer.
*/
MAT FullyConnectedLayer::grad(MAT& input) {
	if (getHierachy() == hierarchy_t::input) {
			// VC does not perform RVO for some reason :/
		return deltaSave*appendOneInline(input).transpose(); //(NOUT, 1) x (NIN+1,1).T = (NOUT, NIN+1)
	} else {
		return deltaSave * appendOneInline(below->getACT()).transpose();
	}
}

void FullyConnectedLayer::saveToFile(ostream& os) const {
	/*for (size_t j = 0; j < getNIN() + 1; ++j) {
		for (size_t i = 0; i < getNOUT(); ++i) {
			os<< layer(i, j)<<"\t";
		}
	} */
	MAT temp = layer;
	temp.resize(layer.size(), 1);
	os << temp;
	os << endl;
}
void FullyConnectedLayer::loadFromFile(ifstream& in) {
	
	layer = MAT(getNOUT()*(getNIN() + 1),1);    // initialize as a column vector
	V = MAT(getNOUT(), getNIN() + 1);
	G = MAT(getNOUT(), 1);
	V.setZero();
	G.setZero();
	stepper.reset();

	/*for (size_t j = 0; j< getNIN()+1; ++j) {
		for (size_t i = 0; i < getNOUT(); ++i) {
			in >> layer(i,j);
		}
	}*/
	for (size_t i = 0; i < layer.size(); ++i) {
		in >> layer(i);
	}
	layer.resize(getNOUT(), getNIN() + 1);
	//layer.rowwise().reverseInPlace();
	//layer.transposeInPlace();
	//layer.colwise().reverseInPlace();

}