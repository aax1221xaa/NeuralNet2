#include "../header/OptimizerAlgorithm.h"
#include "../header/NN_Cuda.cuh"



/******************************************************

					 NN_Optimizer

*******************************************************/

NN_Optimizer::NN_Optimizer() {

}

NN_Optimizer::~NN_Optimizer() {

}

void NN_Optimizer::Run(GPU_Mem<float> w, float* dw, GPU_Mem<float> wm) {

}



/******************************************************

						NN_SGD

*******************************************************/

NN_SGD::NN_SGD(float _lRate, float _mCoeff) {
	lRate = _lRate;
	mCoeff = _mCoeff;
}

NN_SGD::~NN_SGD() {

}

void NN_SGD::Run(GPU_Mem<float> w, float* dw, GPU_Mem<float> wm) {
	CHECK_CUDA(SGD(w.data, dw, wm.data, lRate, mCoeff, (int)w.elements));
}