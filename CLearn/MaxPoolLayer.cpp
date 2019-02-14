
#include "stdafx.h"
#include "MaxPoolLayer.h"

// Constructors
MaxPoolLayer::MaxPoolLayer(size_t NINXY, size_t _maxOverXY)
	: inFeatures(1), NINX(NINXY), NINY(NINXY), maxOverX(_maxOverXY), maxOverY(_maxOverXY), NOUTX((NINXY / _maxOverXY)), NOUTY((NINXY / _maxOverXY)), 
	DiscarnateLayer(((int)(NINXY / _maxOverXY))*((int)(NINXY / _maxOverXY)),NINXY*NINXY, actfunc_t::NONE){
	init();
	assertGeometry();
}
// This constructor assumes a square input NINX x NINY.
// However, we 
MaxPoolLayer::MaxPoolLayer(size_t _maxOverXY, CNetLayer& lower) 
	: inFeatures(lower.getFeatures()), NINX((int)sqrt(lower.getNOUT()/lower.getFeatures())), NINY((int)sqrt(lower.getNOUT() / lower.getFeatures())),
	maxOverX(_maxOverXY), maxOverY(_maxOverXY), NOUTX(((int)sqrt(lower.getNOUT() / lower.getFeatures())) / _maxOverXY), NOUTY(((int)sqrt(lower.getNOUT() / lower.getFeatures())) / _maxOverXY),
	DiscarnateLayer( lower.getFeatures()*((int)sqrt(lower.getNOUT() / lower.getFeatures()) / _maxOverXY)*((int)sqrt(lower.getNOUT() / lower.getFeatures()) / _maxOverXY), actfunc_t::NONE, lower) {
	init();
	assertGeometry();
}
MaxPoolLayer::MaxPoolLayer(CNetLayer& lower) : MaxPoolLayer::MaxPoolLayer(2, lower) {}

MaxPoolLayer::~MaxPoolLayer(){}

void MaxPoolLayer::assertGeometry() {
	assert(NOUTY*NOUTX*inFeatures == getNOUT());
	assert(inFeatures*NINX*NINY == getNIN());
}

void MaxPoolLayer::init() {
	indexX = MATINDEX(NOUTY, inFeatures*NOUTX);
	indexX.setConstant(0);
	indexY = MATINDEX(NOUTY, inFeatures*NOUTX);
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
	os << maxOverX << " " << maxOverY << " " << NINX << " " << NINY << " " << NOUTX << " "<< NOUTY << " "<< inFeatures<<endl;
}
void MaxPoolLayer::loadFromFile(ifstream& in) {
	in >> maxOverX;
	in >> maxOverY;
	in >> NINX;
	in >> NINY; 
	in >> NOUTX;
	in >> NOUTY;
	in >> inFeatures;
}

void MaxPoolLayer::forProp(MAT& inBelow,  bool training, bool recursive) {
	inBelow.resize(NINY, inFeatures* NINX);
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
	//deltaSave = deltaAbove;
	if (getHierachy() != hierarchy_t::input || !recursive) { // ... this is not an input layer.
		deltaAbove.resize(NOUTY, inFeatures*NOUTX);

		MAT newDelta(NINY, inFeatures*NINX);
		newDelta.setConstant(0);
		for (size_t f = 0; f < inFeatures; ++f) {
			for (size_t m = 0; m < NOUTY; ++m) {
				for (size_t n = 0; n < NOUTX; ++n) {
					if (indexY(m, n+f*NOUTX) >= 0 && indexY(m, n + f*NOUTX) < NINY
						&& indexX(m, n + f*NOUTX) >= 0 && indexX(m, n + f*NOUTX) < NINX)
						newDelta(indexY(m, n + f*NOUTX), indexX(m, n + f*NOUTX)) = deltaAbove(m, n + f*NOUTX);
				}
			}
		}
		newDelta.resize(getNIN(), 1);
		deltaAbove = std::move(newDelta); 
			
		if (recursive)
			below->backPropDelta(deltaAbove, true);
		
	}
}

MAT MaxPoolLayer::maxPool(const MAT& in, bool saveIndices){
	MAT out( NOUTY, inFeatures*NOUTX );
	static const fREAL initNegative = -1000;
	
	fREAL curMax = initNegative;
	
	size_t ii = -1;
	size_t jj = -1;

	for (size_t f = 0; f < inFeatures; ++f) {
		for (size_t j = 0; j < NOUTY; ++j) {
			for (size_t i = 0; i < NOUTX; ++i) {
				for (size_t k = 0; k < maxOverY; ++k) {
					for (size_t l = 0; l < maxOverX; ++l) {
						if (maxOverY*j + k < NINY && maxOverX*i + l < NINX 
							&& in(maxOverY*j + k, maxOverX*i + l + f*NINX) >= curMax) {
							
							curMax = in(maxOverY*j + k, maxOverX*i + l + f*NINX); // max(maxOverY*j+k) = maxOverY*((inY/maxOverY)-1) + maxOverY-1 ~ inY-maxOverY + maxOverY-1 = inY - 1 -> Correct, same for x 
							if (saveIndices) {
								ii = maxOverX*i + l; // indices should remain within (NOUTY, NOUTX) - the features are taken care of manually
								jj = maxOverY*j + k;
							}
						}
					}
				}
				out(j, i + f*NOUTX) = curMax;
				if (saveIndices) {
					indexX(j, i + f*NOUTX) = ii;
					indexY(j, i + f*NOUTX) = jj;
					ii = -1;
					jj = -1;
				}
				curMax = initNegative;
			}
		}
	}
	return out;
}