#pragma once

#ifndef M_PI
#define M_PI 3.14159265359
#endif // !1

#ifdef CLEARN_EXPORTS  
#define CLEARN_API__declspec(dllexport)   
#else  
#define CLEARN_API__declspec(dllimport)   
#endif  
#include <vector>
#include <memory>
#include "Eigen/Core"

using namespace Eigen;
using namespace std;

// Define a few types.
typedef float fREAL;
typedef Matrix<fREAL, Dynamic, Dynamic> MAT;
typedef Matrix<fREAL, Dynamic, Dynamic, Dynamic> MAT3;
typedef Matrix<uint8_t, Dynamic, Dynamic> MATU8;
typedef Map<MAT> MATMAP;
typedef vector<MAT> MATVEC;
typedef Map<MATU8> MATU8MAP;

template<typename T>
void inline copyToOut(T* const in, T* const out, uint32_t N) {
	for (uint32_t i = 0; i < N; i++) {
		out[i] = in[i];
	}
}

struct learnPars {
	fREAL eta; // learning rate
	fREAL metaEta; // learning rate decay rate
	fREAL gamma; // inertia term
	fREAL lambda; // regularizer
	uint32_t nesterov;
};

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

		// private functions
		void appendOne(MAT&);
		void shrinkOne(MAT& in);
		MAT appendOneInline(const MAT&);
		
		static MAT matNorm(const MAT&);
		static fREAL norm(fREAL);
		static fREAL cumSum(const MAT&);

		// activation functions
		static MAT ACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer
		static MAT DACT(const MAT&); // TODO speed up by (const MAT&) pass by pointer
		static fREAL Tanh(fREAL);
		static fREAL Sig(fREAL);
		static fREAL DSig(fREAL);
		static fREAL ReLu(fREAL);
		static fREAL DReLu(fREAL);
};
