#pragma once

#include "defininitions.h"

#ifdef CLEARN_EXPORTS  
#define CLEARN_API__declspec(dllexport)   
#else  
#define CLEARN_API__declspec(dllimport)   
#endif  

class CLearn {
	public:
		CLearn(uint32_t, uint32_t, uint32_t, uint32_t* const);
		~CLearn();
		// Training functions
		fREAL backProp(const MAT&, MAT&, learnPars); // returns cum error and overwrites the second argument with the prediction
		MAT forProp(const MAT&, bool saveActivations);
		fREAL l2Error(const MAT&);
		// Application
		
		// Getter functions
		uint32_t get_NIN();
		uint32_t get_NOUT();
		uint32_t get_NNODE(uint32_t);
		uint32_t get_NHIDDEN();
		// copy the weight matrices
		void copy_inWeights(fREAL* const);
		void copy_hiddenWeights(fREAL* const, uint32_t);
		void copy_outWeights(fREAL* const);

	private:
		// private elements
		MAT inLayer; // weights
		MATVEC hiddenLayers; // weights
		MAT outLayer; // weights

		MAT inVel; //velocities 
		MATVEC hiddenVel; //velocities
		MAT outVel; //velocities

		MAT inAct; // activations
		MATVEC hiddenActs; // activations
		MAT outAct; // activations

		MAT outDelta; // deltas for backprop
		MATVEC hiddenDeltas; // deltas for backprop
		MAT inDelta; // deltas for backprop
	
		// private parameters
		uint32_t NIN;
		uint32_t NOUT;
		uint32_t NHIDDEN; // number of hidden layers (NOT NODES!)
		uint32_t* NNODES;

		// activation functions
		static MAT ACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer
		static MAT DACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer

};
