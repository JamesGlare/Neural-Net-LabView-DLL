
#include "stdafx.h"
#include "MaxPoolLayer.h"

// Constructors
MaxPoolLayer::MaxPoolLayer(size_t NINXY, size_t _maxOverXY)
	: NINX(NINXY), NINY(NINXY), maxOverX(_maxOverXY), maxOverY(_maxOverXY), NOUTX((NINXY / _maxOverXY)), NOUTY((NINXY / _maxOverXY)), 
	DiscarnateLayer(((int)(NINXY / _maxOverXY))*((int)(NINXY / _maxOverXY)),NINXY*NINXY, actfunc_t::NONE){
	init();
	assertGeometry();
}

MaxPoolLayer::MaxPoolLayer(size_t _maxOverXY, CNetLayer& lower) 
	: NINX((int)sqrt(lower.getNOUT())), NINY((int)sqrt(lower.getNOUT())), maxOverX(_maxOverXY), maxOverY(_maxOverXY), NOUTX(((int)sqrt(lower.getNOUT())) / _maxOverXY), NOUTY(((int)sqrt(lower.getNOUT())) / _maxOverXY), 
	DiscarnateLayer(((int)sqrt(lower.getNOUT()) / _maxOverXY)*((int)sqrt(lower.getNOUT()) / _maxOverXY), actfunc_t::NONE, lower) {
	init();
	assertGeometry();
}
MaxPoolLayer::MaxPoolLayer(CNetLayer& lower) : MaxPoolLayer::MaxPoolLayer(2, lower) {}

MaxPoolLayer::~MaxPoolLayer(){}

void MaxPoolLayer::assertGeometry() {
	assert(NOUTY*NOUTX == getNOUT());
	assert(NINX*NINY == getNIN());
}

void MaxPoolLayer::init() {
	indexX = MATINDEX(NOUTY, NOUTX);
	indexX.setConstant(0);
	indexY = MATINDEX(NOUTY, NOUTX);
	indexY.setConstant(0);
	actSave = MAT(getNOUT(), 1); 
	actSave.setConstant(0);
	deltaSave = MAT(1, 1);
	deltaSave.setConstant(0);
} 
layer_t MaxPoolLayer::whoAmI() const {
	return layer_t::maxPooling;
}


void MaxPoolLayer::saveToFile(ostream& os) const {
	os << maxOverX << "\t" << maxOverY << "\t" << NINX << "\t" << NINY << "\t" << NOUTX << "\t"<< NOUTY << endl;
}
void MaxPoolLayer::loadFromFile(ifstream& in) {
	in >> maxOverX;
	in >> maxOverY;
	in >> NINX;
	in >> NINY; 
	in >> NOUTX;
	in >> NOUTY;
}

void MaxPoolLayer::forProp(MAT& inBelow,  bool training, bool recursive) {
	inBelow.resize(NINY, NINX);
	inBelow = maxPool(inBelow, training);
	inBelow.resize(getNOUT(), 1);
	if (training)
		actSave = inBelow;
	if (getHierachy() != hierarchy_t::output) {
		if(recursive)
			above->forProp(inBelow,  training, true);
	}
}

void MaxPoolLayer::backPropDelta(MAT& deltaAbove, bool recursive) {

	if (getHierachy() != hierarchy_t::input || !recursive) { // ... this is not an input layer.
		deltaAbove.resize(NOUTY, NOUTX);

		MAT newDelta(NINY, NINX);
		newDelta.setConstant(0);
		for (size_t m = 0; m < NOUTY; m++) {
			for (size_t n = 0; n < NOUTX; n++) {
				if(indexY(m, n) >=0 && indexY(m, n) < NINY 
					&& indexX(m, n) >= 0 && indexX(m, n) < NINX)
					newDelta(indexY(m, n), indexX(m, n)) = deltaAbove(m, n);
			}
		}
		newDelta.resize(getNIN(), 1);
		deltaAbove = newDelta; // convention - the "deltaAbove" instance should carry the deltas.
		if(recursive)
			below->backPropDelta(deltaAbove, true);
	}
}

MAT MaxPoolLayer::maxPool(const MAT& in, bool saveIndices){
	MAT out( NOUTY, NOUTX );
	fREAL curMax = 0;
	size_t ii = -1;
	size_t jj = -1;

	for (size_t j = 0; j < NOUTY; j++) {
		for (size_t i = 0; i < NOUTX; i++) {
			for (size_t k = 0; k < maxOverY; k++) {
				for (size_t l = 0; l < maxOverX; l++) {
					if (maxOverY*j + k < NINY && maxOverX*i + l < NINX && in(maxOverY*j + k, maxOverX*i + l) >= curMax) {
							curMax = in(maxOverY*j + k, maxOverX*i + l); // max(maxOverY*j+k) = maxOverY*((inY/maxOverY)-1) + maxOverY-1 ~ inY-maxOverY + maxOverY-1 = inY - 1 -> Correct, same for x 
							if (saveIndices) {
								ii = maxOverX*i + l;
								jj = maxOverY*j + k;
							}
						}
				}
			}
			out(j, i) = curMax;
			if (saveIndices) {
				indexX(j, i) = ii;
				indexY(j, i) = jj;
				ii = -1;
				jj = -1;
			}
			curMax = 0;
		}
	}
	return out;
}