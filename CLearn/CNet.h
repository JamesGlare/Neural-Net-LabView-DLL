#pragma once

#include "defininitions.h"
#include <memory>
#include "CNETLayer.h"
#include "MixtureDensityModel.h"
typedef std::unique_ptr<CNetLayer> layerPtr;
 

#ifndef CNET_CNET
#define CNET_CNET
/*	Master Class which directs the learning process and manages the layers.
*	The layers are implemented as a linked list, but each pointer is also stored in a vector in this class.
*	Forward- and back-propagation simply tranverse the list in the respective direction.
*/
class CNet {
	public:
		CNet(size_t NIN);
		~CNet();
		// Physical Layers (i.e. layers with weight parameters)
		void addFullyConnectedLayer(size_t NOUT, actfunc_t type);
		void addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t sideChannels, actfunc_t type);
		void addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t outBoxes, size_t sideChannels, actfunc_t type);

		// Discarnate Layers (i.e. layers without weight parameters)
		void addPoolingLayer(size_t maxOverXY, pooling_t type);
		void addDropoutLayer(fREAL ratio);
		void addPassOnLayer(actfunc_t type);
		void addMixtureDensity(size_t NOUT, size_t features, size_t BlockXY);
		void addReshape();

		// Propagate input matrix through entire network. Results are stored in "in".
		fREAL forProp(MAT& in, const MAT& outDesired, const learnPars& pars);
		// Backpropagate through network. 
		fREAL backProp(MAT& in, MAT& outDesired, const learnPars& pars);
		
		// Save-to-file functionality.
		void saveToFile(string filePath) const;
		void loadFromFile(string filePath);
		void loadFromFile_layer(string filePath, uint32_t layerNr);

		void debugMsg(fREAL* msg);
		// Getter functions
		inline size_t getLayerNumber() const { return layers.size(); };
		inline size_t getNIN() const { return NIN; };
		void copyNthLayer(size_t layer, fREAL* const toCopyTo) const;
		void setNthLayer(size_t layer, const MAT& newLayer);
		size_t getNOUT() const;

		void inquireDimensions (size_t layer, size_t& rows, size_t& cols) const;

	private:

		inline CNetLayer* getLast() const { return layers.back(); };
		inline CNetLayer* getFirst() const { return layers.front(); };

		// error related functions
		MAT errorMatrix(const MAT& outPrediction, const MAT& outDesired);
		fREAL l2_error(const MAT& diff);
		inline bool isPhysical(size_t layer) const {
			return (layers[layer]->whoAmI()	!= layer_t::maxPooling
				&& layers[layer]->whoAmI()	!= layer_t::passOn
				&& layers[layer]->whoAmI()	!= layer_t::dropout
				&& layers[layer]->whoAmI()	!= layer_t::mixtureDensity
				);
		}
		size_t NIN;
		vector<CNetLayer*> layers;
};

#endif 
