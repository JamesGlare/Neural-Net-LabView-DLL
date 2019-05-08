#include "stdafx.h"
#include "GaussianReparametrizationLayer.h"

GaussianReparametrizationLayer::GaussianReparametrizationLayer(size_t NOUT) : ones(NOUT, 1), eps(NOUT, 1), DiscarnateLayer(NOUT, 2*NOUT)
{
	init();
}

GaussianReparametrizationLayer::GaussianReparametrizationLayer(CNetLayer & lower) : ones(lower.getNOUT()/2, 1), eps(lower.getNOUT()/2, 1), DiscarnateLayer(lower.getNOUT()/2, actfunc_t::NONE, lower)
{
	init();
}

GaussianReparametrizationLayer::~GaussianReparametrizationLayer()
{
}

layer_t GaussianReparametrizationLayer::whoAmI() const
{
	return layer_t::gaussreparam;
}

void GaussianReparametrizationLayer::forProp(MAT & in, bool saveActivation, bool recursive)
{

	//PRECON in has to be of shape (NOUT, 2) where the zeroth column decodes mu, while the first contains log(sigma)
	// (0) redraw eps - think about the sequence of this
	// (0.5) resize in
	in.resize(getNOUT(), 2); // should leave the values intact
	// (1) Build out
	if (saveActivation) {
		actSave = in;
		in = move(actSave.leftCols(1) + actSave.rightCols(2).unaryExpr(&exp_fREAL).cwiseProduct(eps)); // (NOUT,1) vector
	} 
	else 
	{
		eps.unaryExpr(&std_normal);
		in = move(in.leftCols(1) + in.rightCols(1).unaryExpr(&exp_fREAL).cwiseProduct(eps)); // (NOUT,1) vector
	}
	
	if (getHierachy() != hierarchy_t::output && recursive)
		above->forProp(in, saveActivation, recursive);
}

void GaussianReparametrizationLayer::backPropDelta(MAT & delta, bool recursive)
{
	// (0) Delta is an (NOUT, 1)-shaped matrix by contract
	// We need to derive the deltas for {mu, log sigma} and then pass that moster on
	deltaSave = delta;
	static fREAL beta = 0.1;
	if (getHierachy() != hierarchy_t::input) {
		MAT newDelta(getNOUT(), 2);

		// And, like, you know, about this KL-term, 
		// we simply add the gradient of the Kullback-Leibler divergence, right? 

		newDelta.leftCols(1) = move(beta*delta - actSave.leftCols(1)); // dz/dmu = 1 -> delta = deltahere*deltaAbove = 1*deltaAbove
		newDelta.rightCols(1) = move(beta*delta.cwiseProduct(actSave.rightCols(1).cwiseProduct(eps))- 0.5f*(actSave.rightCols(1).unaryExpr(&exp_fREAL) - ones)); // dz/dlogsigma = sigma*eps
		newDelta.resize(2 * getNOUT(), 1); // should leave coeffs untouched
		delta = newDelta; // (2*NOUT, 1)
		if (recursive) {
			below->backPropDelta(delta, true); // cascade...
		}
	}
}

void GaussianReparametrizationLayer::init()
{
	assert(getNIN() == 2 * getNOUT());
	actSave.resize(getNOUT(), 2);
	actSave.setZero();
	ones.setOnes();
	eps.unaryExpr(&std_normal);
}

void GaussianReparametrizationLayer::saveToFile(ostream & os) const
{
}

void GaussianReparametrizationLayer::loadFromFile(ifstream & in)
{
}
