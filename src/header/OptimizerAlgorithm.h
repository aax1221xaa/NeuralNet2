#pragma once
#include "NN_Base.h"





/******************************************************

					 NN_Optimizer

*******************************************************/

class NN_Optimizer {
public:
	NN_Optimizer();
	virtual ~NN_Optimizer();

	virtual void Run(GPU_Mem<float> w, float *dw, GPU_Mem<float> wm);
};




/******************************************************

						  NN_SGD

*******************************************************/

class NN_SGD : public NN_Optimizer {
public:
	float lRate;
	float mCoeff;

	NN_SGD(float _lRate, float _mCoeff);
	~NN_SGD();

	void Run(GPU_Mem<float> w, float* dw, GPU_Mem<float> wm);
};