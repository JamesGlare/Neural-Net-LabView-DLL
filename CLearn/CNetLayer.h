#pragma once

#include "defininitions.h"

#ifndef CNET_CNETLAYER
#define CNET_CNETLAYER

/* Abstract base class for a layer of weights.
 * This means that this class sits in between nodes and forwards input or
 * backpropagates deltas.
 */
class CNetLayer {
	public:
		// constructors and initializers
		CNetLayer(size_t _NOUT, size_t _NIN);
		CNetLayer(size_t _NOUT, size_t _NIN, actfunc_t type);
		CNetLayer(size_t _NOUT, actfunc_t type, CNetLayer& lower);
		// Delete Copy- and Move CTORs
		CNetLayer(const CNetLayer& other) = delete;
		CNetLayer(CNetLayer&& other) = delete;
		
		virtual ~CNetLayer() {}; 

		// type
		virtual layer_t whoAmI() const = 0;
		// forProp
		virtual void forProp(MAT& in, bool training, bool recursive) = 0; // recursive
		// backprop
		virtual void backPropDelta(MAT& delta, bool recursive) = 0; // recursive
		virtual void applyUpdate(const learnPars& pars, MAT& input, bool recursive) =0 ; // recursive

		// Getter Functions
		inline size_t getNIN() const { return NIN; };
		inline size_t getNOUT() const { return NOUT; };
		inline size_t getLayerNumber() const { return layerNumber; };
		inline hierarchy_t getHierachy() const { return hierarchy; };

		MAT getDACT() const; // derivative of activation function
		MAT getACT() const; // activation function
		MAT getDelta() const; // get backpropagated delta

		// Connect to layer above and change hierarchy from output to hidden
		void connectAbove(CNetLayer* ptr);
		void connectBelow(CNetLayer* ptr);
		void checkHierarchy(bool recursive);

		// save to file
		friend ostream& operator<<(ostream& os, const CNetLayer& toSave); // almost virtual member
		friend ifstream& operator >> (ifstream& in, CNetLayer& toReconstruct);

	protected:
		MAT actSave; // keep activation before propagation
		MAT deltaSave; // store deltas for backprop
		// saving functions
		void saveMother(ostream& os) const;
		void reconstructMother(ifstream& in) ;
		virtual void saveToFile(ostream& os) const = 0;
		virtual void loadFromFile(ifstream& in) = 0;
		CNetLayer* below; // store pointer to layer below ... 
		CNetLayer* above; // and above. 
		ACTFUNC act; // pointer to the activation function
		ACTFUNC dact; // pointer to the activation function derivative
		void changeActFunc(CNetLayer& changeMyActivation, actfunc_t type); // ask another instance of CNetLayer to change its non-linearity

	private:		
		actfunc_t activationType; // store the type of activation function
		hierarchy_t hierarchy;
		size_t NOUT; // number of outputs. For convolutional layers NOUT = NOUTX*NOUTY*Features
		size_t NIN; // number of inputs. Does not contain the bias node. 
		void assignActFunc(actfunc_t type);

		size_t layerNumber;

};

#endif