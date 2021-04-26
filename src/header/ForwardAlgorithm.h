#pragma once

#include "NN_Base.h"
#include "BackwardAlgorithm.h"




/******************************************************

						NN_Forward

*******************************************************/

class NN_Forward : public NN_Base {
public:
	const int attr;

	vector<NN_Forward*> prev;
	vector<NN_Forward*> next;

	GPU_Mem<float> y;
	cudnnTensorDescriptor_t yDesc;
	Dim yDim;

	NN_Backward* backModule;

	NN_Forward(const string _name, const string _layerName, const int _attr);
	virtual ~NN_Forward();

	NN_Backward* Create_D_ConcatModule(vector<NN_Backward*>& p);
	virtual void CreateBackwardModule(vector<NN_Backward*>& p);
	void SetPrevNode(NN_Forward* prevLayer);
	void SetNextNode(NN_Forward* nextLayer);
	virtual void SaveWeight(cv::FileStorage& fs);
	virtual void LoadWeight(cv::FileStorage& fs);

	virtual void SetInput(cv::Mat _x);
	cv::Mat GetOutput();
	virtual float CalcLoss(cv::Mat _y_true);
};




/******************************************************

					  NN_ConcatAdd

*******************************************************/

class NN_ConcatAdd : public NN_Forward {
public:
	NN_ConcatAdd(string _name, vector<NN_Forward*> prevLayer);
	~NN_ConcatAdd();

	void Run();
	void CalcSize();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void PrintLayerInfo();
};




/******************************************************

						NN_Input

*******************************************************/

class NN_Input : public NN_Forward {
public:
	cv::Mat x;

	NN_Input(Dim dim, string _name);
	~NN_Input();

	void Run();
	void SetInput(cv::Mat _x);
	void PrintLayerInfo();
};



/******************************************************

						NN_Random

*******************************************************/

class NN_Random {
public:
	virtual ~NN_Random();
	virtual cv::Mat InitWeight(Dim wSize);
};



/******************************************************

					NN_LeCunInit

*******************************************************/

class NN_LeCunInit : public NN_Random {
public:
	cv::Mat InitWeight(Dim wSize);
};



/******************************************************

					NN_XavierInit

*******************************************************/

class NN_XavierInit : public NN_Random {
public:
	cv::Mat InitWeight(Dim wSize);
};



/******************************************************

						NN_HeInit

*******************************************************/

class NN_HeInit : public NN_Random {
public:
	cv::Mat InitWeight(Dim wSize);
};




/******************************************************

					   NN_Dense

*******************************************************/

class NN_Dense : public NN_Forward {
public:
	bool addBias;

	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	GPU_Mem<float> w;
	GPU_Mem<float> b;

	Dim wDim;
	cudnnFilterDescriptor_t wDesc;
	cudnnTensorDescriptor_t bDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t algorithm;

	NN_Dense(int amount, NN_Random* random, bool _addBias, string _name, NN_Forward* prevLayer);
	~NN_Dense();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void SaveWeight(cv::FileStorage& fs);
	void LoadWeight(cv::FileStorage& fs);
	void PrintLayerInfo();
};



/******************************************************

					   NN_Dropout

*******************************************************/

class NN_Dropout : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	int modeState;

	float dropout;
	size_t stateSize;
	size_t reservSize;
	void* reservSpace;
	void* stateSpace;
	cudnnDropoutDescriptor_t dropDesc;

	NN_Dropout(float _dropout, string _name, NN_Forward* prevLayer);
	~NN_Dropout();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};



/******************************************************

						NN_ReLU

*******************************************************/

class NN_ReLU : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	cudnnActivationDescriptor_t actDesc;

	NN_ReLU(string _name, NN_Forward* prevLayer);
	~NN_ReLU();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};



/******************************************************

						NN_Sigmoid

*******************************************************/

class NN_Sigmoid : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	cudnnActivationDescriptor_t actDesc;

	NN_Sigmoid(string _name, NN_Forward* prevLayer);
	~NN_Sigmoid();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};




/******************************************************

					  NN_Softmax

*******************************************************/

class NN_Softmax : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	NN_Softmax(string _name, NN_Forward* prevLayer);
	~NN_Softmax();

	void CreateBackwardModule(vector<NN_Backward*>& p);
	void Run();
	void CalcSize();
	cv::Mat GetOutput();
	void PrintLayerInfo();
};



/******************************************************

					  NN_CrossEntropy

*******************************************************/

class NN_CrossEntropy : public NN_Forward {
public :
	GPU_Mem<float>& x;
	GPU_Mem<float> y_true;
	Dim& xDim;

	cudaError_t(*LossFunc)(float* y_pred, float* y_true, float* loss, int n, int h, int w, int c);

	NN_CrossEntropy(string _name, NN_Forward* prevLayer);
	~NN_CrossEntropy();

	float CalcLoss(cv::Mat _y_true);
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};




/******************************************************

				      NN_Flatten

*******************************************************/

class NN_Flatten : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	NN_Flatten(string _name, NN_Forward* prevLayer);
	~NN_Flatten();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};



/******************************************************

					  NN_Convolution

*******************************************************/

class NN_Convolution : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	bool addBias;
	vector<int> stride;
	vector<int> dilation;
	int pad;

	Dim wDim;
	GPU_Mem<float> w;
	GPU_Mem<float> b;

	cudnnFilterDescriptor_t wDesc;
	cudnnTensorDescriptor_t bDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t algorithm;

	NN_Convolution(
		int amount, 
		vector<int> kSize, 
		vector<int> stride,
		vector<int> dilation, 
		NN_Random* random,
		int pad, 
		bool _addBias,
		string _name,
		NN_Forward* prevLayer
		);
	~NN_Convolution();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void SaveWeight(cv::FileStorage& fs);
	void LoadWeight(cv::FileStorage& fs);
	void PrintLayerInfo();
};




/******************************************************

					  NN_Maxpool

*******************************************************/

class NN_Maxpool : public NN_Forward {
public:
	GPU_Mem<float>& x;
	cudnnTensorDescriptor_t xDesc;
	Dim& xDim;

	const int pad;
	vector<int> kSize;
	vector<int> stride;
	cudnnPoolingDescriptor_t poolDesc;

	NN_Maxpool(vector<int> kSize, vector<int> stride, const int _pad, string _name, NN_Forward* prevLayer);
	~NN_Maxpool();

	void Run();
	void CreateBackwardModule(vector<NN_Backward*>& p);
	void CalcSize();
	void PrintLayerInfo();
};