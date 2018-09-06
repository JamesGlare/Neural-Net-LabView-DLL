#include "stdafx.h"
#include "DiscarnateLayer.h"

DiscarnateLayer::DiscarnateLayer(size_t _NOUT, size_t _NIN) : CNetLayer(_NOUT, _NIN) {};
DiscarnateLayer::DiscarnateLayer(size_t _NOUT, size_t _NIN, actfunc_t type) : CNetLayer(_NOUT, _NIN, type) {};
DiscarnateLayer::DiscarnateLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower) : CNetLayer(_NOUT, type, lower) {};

void DiscarnateLayer::applyUpdate(const learnPars& pars, MAT& input, bool recursive) {
	if (hierarchy != hierarchy_t::output && recursive) {
		above->applyUpdate(pars, input, true);
	}
}
