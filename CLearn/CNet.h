#pragma once

#include "defininitions.h"
#include <memory>
#include "CNETLayer.h"
#include "MixtureDensityModel.h"
//typedef std::unique_ptr<CNetLayer> layerPtr; // currently not in use.
 

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
		void addConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, actfunc_t type);
		void addAntiConvolutionalLayer(size_t NOUTXY, size_t kernelXY, size_t stride, size_t features, size_t outBoxes, actfunc_t type);

		// Discarnate Layers (i.e. layers without weight parameters)
		void addPoolingLayer(size_t maxOverXY, pooling_t type);
		void addDropoutLayer(fREAL ratio);
		void addPassOnLayer(actfunc_t type);
		void addMixtureDensity(size_t NOUT, size_t features, size_t BlockXY);
		void addReshape();
		void addSideChannel(size_t sideChannelSize);

		// Share Layer Functionality
		void shareLayers(CNet* const otherNet, uint32_t firstLayer, uint32_t lastLayer);

		// Propagate input matrix through entire network. Results are stored in "in".
		fREAL forProp(MAT& in, const MAT& outDesired, bool saveAct, const learnPars& pars);
		// Backpropagate through network. 
		fREAL backProp(MAT& in, MAT& outDesired, const learnPars& pars, bool deltaProvided=false); // set bool to 'true' if you outDesired contains delta's from other network

		// Specialized functions for training GANs
		void train_GAN_D(MAT& Y_copy, MAT &Y, MAT& res, bool real, const learnPars& pars);
		void train_GAN_G_D(MAT& Y_copy, MAT& res, const learnPars& pars);
		void backProp_GAN_G(MAT& Y, MAT& deltaMatrix, learnPars& pars); // Generator

		// Feed a tensor into the sidechannel of the network
		// TODO - Generalize to arbitrarily many sidechannels.
		void preFeedSideChannel(const MAT& sideChannel);
		size_t getSideChannelSize();

		// Save-to-file functionality.
		void saveToFile(string filePath) const;
		void loadFromFile(string filePath);
		void loadFromFile_layer(string filePath, uint32_t layerNr);

		void debugMsg(fREAL* msg);
		// (Re) link the chain must be called directly before forward/backward propagation
		// this enables dynamical switching of layers.
		void linkChain();

		// Getter functions
		inline size_t getLayerNumber() const { return layers.size(); };
		inline size_t getNIN() const { return NIN; };
		void copyNthLayer(size_t layer, fREAL* const toCopyTo) const;
		void setNthLayer(size_t layer, const MAT& newLayer);
		void copyNthActivation(size_t layer, fREAL* const toCopyTo) const;
		void copyNthDelta(size_t layer, fREAL* const toCopyTo, int32_t size) const;
		size_t getNOUT() const;

		void inquireDimensions (size_t layer, size_t& rows, size_t& cols) const;

	private:

		inline CNetLayer* getLast() const { return layers.back(); };
		inline CNetLayer* getFirst() const { return layers.front(); };

		// error related functions
		MAT l2_errorMatrix(const MAT& diff);
		MAT l1_errorMatrix(const MAT& diff);
		fREAL l2_error(const MAT& diff);
		fREAL l1_error(const MAT& diff);
		// GAN Error
		fREAL sigmoid_cross_entropy_with_logits(const MAT& logits, const MAT& labels);
		MAT sigmoid_cross_entropy_errorMatrix(const MAT& logits, const MAT& labels);
		MAT sigmoid_GEN_loss(const MAT& labels, const MAT& logits);
		// Regularzier
		MAT entropyRegularizer(const MAT& out);

		
		inline bool isPhysical(size_t layer) const {
			return (layers[layer]->whoAmI()	== layer_t::fullyConnected
				|| layers[layer]->whoAmI()	== layer_t::convolutional
				|| layers[layer]->whoAmI()	== layer_t::antiConvolutional
				);
		}
		
		
		size_t NIN;
		vector<CNetLayer*> layers;
};

#endif 
