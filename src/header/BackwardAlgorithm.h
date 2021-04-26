#pragma once
#include "NN_Base.h"
#include "NN_Cuda.cuh"
#include "OptimizerAlgorithm.h"



/******************************************************

					  NN_Backward

*******************************************************/

class NN_Backward : public NN_Base {
public:
	GPU_Mem<float> dx;
	cudnnTensorDescriptor_t dxDesc;

	static NN_Optimizer* opt;

	NN_Backward(const string _name, const string _layerName);
	~NN_Backward();

	static void SetOptimizer(NN_Optimizer* _opt);
	static void ClearOptimizer();
};




/******************************************************

					  NN_D_Disperse

*******************************************************/

class NN_D_Disperse : public NN_Backward {
public:
	GPU_Mem<float>& dy;
	cudnnTensorDescriptor_t dyDesc;

	NN_D_Disperse(NN_Base& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Disperse();

	void Run();
};




/******************************************************

					  NN_D_ConcatAdd

*******************************************************/

class NN_D_ConcatAdd : public NN_Backward {
public:
	vector<GPU_Mem<float>*> dy;
	vector<cudnnTensorDescriptor_t> dyDesc;

	NN_D_ConcatAdd(vector<GPU_Mem<float>*> _dy, vector<cudnnTensorDescriptor_t> _dyDesc);
	~NN_D_ConcatAdd();

	void Run();
	void CalcSize();
};




/******************************************************

				     NN_D_Softmax

*******************************************************/

template <class _T>
class NN_D_Softmax : public NN_Backward {
public:
	GPU_Mem<float>& dy;
	GPU_Mem<float>& y;

	cudnnTensorDescriptor_t dyDesc;
	cudnnTensorDescriptor_t yDesc;
	Dim& xDim;

	NN_D_Softmax(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Softmax();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Softmax<_T>::NN_D_Softmax(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Softmax"),
	dy(_dy),
	y(p.y),
	xDim(p.xDim)
{
	try {
		dyDesc = _dyDesc;
		yDesc = p.yDesc;

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Softmax<_T>::~NN_D_Softmax() {
	try {
		Release(dx);
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Softmax<_T>::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnSoftmaxBackward(
		NN_Base::handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		yDesc,
		y.data,
		dyDesc,
		dy.data,
		&beta,
		dxDesc,
		dx.data
	));
}

template <class _T>
void NN_D_Softmax<_T>::CalcSize() {
	ReAlloc(dx, xDim);
	SetDesc(dxDesc, xDim);
}



/******************************************************

					  NN_D_CrossEntropy

*******************************************************/

template <class _T>
class NN_D_CrossEntropy : public NN_Backward {
public:
	GPU_Mem<float>& x;
	GPU_Mem<float>& y_true;
	Dim& xDim;

	NN_D_CrossEntropy(_T& p);
	~NN_D_CrossEntropy();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_CrossEntropy<_T>::NN_D_CrossEntropy(_T& p) :
	NN_Backward(p.name, "NN_D_CrossEntropy"),
	x(p.x),
	y_true(p.y_true),
	xDim(p.xDim)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_CrossEntropy<_T>::~NN_D_CrossEntropy() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		Release(dx);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_CrossEntropy<_T>::Run() {
	CHECK_CUDA(D_CrossEntropy(x.data, y_true.data, dx.data, (int)x.elements, xDim.n));
}

template <class _T>
void NN_D_CrossEntropy<_T>::CalcSize() {
	ReAlloc(dx, xDim);
	SetDesc(dxDesc, xDim);
}



/******************************************************

					 NN_D_Activation

*******************************************************/

template <class _T>
class NN_D_Activation : public NN_Backward {
public:
	GPU_Mem<float>& x;
	GPU_Mem<float>& y;
	GPU_Mem<float>& dy;
	cudnnTensorDescriptor_t xDesc;
	cudnnTensorDescriptor_t yDesc;
	cudnnTensorDescriptor_t dyDesc;
	Dim& xDim;

	cudnnActivationDescriptor_t actDesc;

	NN_D_Activation(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Activation();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Activation<_T>::NN_D_Activation(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_ReLU"),
	x(p.x),
	y(p.y),
	dy(_dy),
	xDesc(p.xDesc),
	yDesc(p.yDesc),
	dyDesc(_dyDesc),
	xDim(p.xDim),
	actDesc(p.actDesc)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Activation<_T>::~NN_D_Activation() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		Release(dx);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Activation<_T>::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnActivationBackward(
		NN_Base::handle,
		actDesc,
		&alpha,
		yDesc,
		y.data,
		dyDesc,
		dy.data,
		xDesc,
		x.data,
		&beta,
		dxDesc,
		dx.data
	));
}

template <class _T>
void NN_D_Activation<_T>::CalcSize() {
	SetDesc(dxDesc, xDim);
	ReAlloc(dx, xDim);
}



/******************************************************

					   NN_D_Dropout

*******************************************************/

template <class _T>
class NN_D_Dropout : public NN_Backward {
public:
	GPU_Mem<float>& dy;
	cudnnTensorDescriptor_t dyDesc;
	
	_T& _forward;

	NN_D_Dropout(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Dropout();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Dropout<_T>::NN_D_Dropout(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Dropout"),
	dy(_dy),
	dyDesc(_dyDesc),
	_forward(p)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Dropout<_T>::~NN_D_Dropout() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Dropout<_T>::Run() {
	CHECK_CUDNN(cudnnDropoutBackward(
		NN_Base::handle,
		_forward.dropDesc,
		dyDesc,
		dy.data,
		dxDesc,
		dx.data,
		_forward.reservSpace,
		_forward.reservSize
	));
}

template <class _T>
void NN_D_Dropout<_T>::CalcSize() {
	ReAlloc(dx, _forward.xDim);
	SetDesc(dxDesc, _forward.xDim);
}




/******************************************************

					 NN_D_Dense

*******************************************************/

template <class _T>
class NN_D_Dense : public NN_Backward {
public:
	GPU_Mem<float>& x;
	GPU_Mem<float>& dy;
	Dim& xDim;
	cudnnTensorDescriptor_t xDesc;
	cudnnTensorDescriptor_t dyDesc;

	Dim& wDim;
	GPU_Mem<float>& w;
	GPU_Mem<float>& b;

	GPU_Mem<float> wm;
	GPU_Mem<float> bm;
	
	cudnnFilterDescriptor_t wDesc;
	cudnnTensorDescriptor_t bDesc;
	cudnnFilterDescriptor_t dwDesc;
	cudnnTensorDescriptor_t dbDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionBwdDataAlgo_t dataAlgo;
	cudnnConvolutionBwdFilterAlgo_t filterAlgo;

	bool& addBias;

	NN_D_Dense(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Dense();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Dense<_T>::NN_D_Dense(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Dense"),
	x(p.x),
	dy(_dy),
	xDim(p.xDim),
	xDesc(p.xDesc),
	dyDesc(_dyDesc),
	wDim(p.wDim),
	w(p.w),
	b(p.b),
	wDesc(p.wDesc),
	bDesc(p.bDesc),
	convDesc(p.convDesc),
	addBias(p.addBias)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
		CHECK_CUDNN(cudnnCreateFilterDescriptor(&dwDesc));	
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(
			dwDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NHWC,
			wDim.n,
			wDim.c,
			wDim.h,
			wDim.w
		));	
		ReAlloc(wm, wDim);	
		CHECK_CUDA(cudaMemset(wm.data, 0, wm.bytes));
		
		if (addBias) {
			CHECK_CUDNN(cudnnCreateTensorDescriptor(&dbDesc));
			SetDesc(dbDesc, 1, 1, 1, wDim.n);
			ReAlloc(bm, wDim.n);
			CHECK_CUDA(cudaMemset(bm.data, 0, bm.bytes));
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Dense<_T>::~NN_D_Dense() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		CHECK_CUDNN(cudnnDestroyFilterDescriptor(dwDesc));	
		Release(wm);
		Release(dx);
		
		if (addBias) {
			CHECK_CUDNN(cudnnDestroyTensorDescriptor(dbDesc));
			Release(bm);
		}
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Dense<_T>::Run() {
	float* dw = NULL;
	size_t workSize = 0;
	void* workSpace = NULL;

	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		NN_Base::handle,
		wDesc,
		dyDesc,
		convDesc,
		dxDesc,
		dataAlgo,
		&workSize
	));
	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDA(cudaMalloc(&dw, w.bytes));

	CHECK_CUDNN(cudnnConvolutionBackwardData(
		NN_Base::handle,
		&alpha,
		wDesc,
		w.data,
		dyDesc,
		dy.data,
		convDesc,
		dataAlgo,
		workSpace,
		workSize,
		&beta,
		dxDesc,
		dx.data
	));

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		NN_Base::handle,
		xDesc,
		dyDesc,
		convDesc,
		dwDesc,
		filterAlgo,
		&workSize
	));
	CHECK_CUDA(cudaFree(workSpace));
	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDNN(cudnnConvolutionBackwardFilter(
		NN_Base::handle,
		&alpha,
		xDesc,
		x.data,
		dyDesc,
		dy.data,
		convDesc,
		filterAlgo,
		workSpace,
		workSize,
		&beta,
		dwDesc,
		dw
	));
	CHECK_CUDA(cudaFree(workSpace));
	NN_Backward::opt->Run(w, dw, wm);
	CHECK_CUDA(cudaFree(dw));

	if (addBias) {
		float* db = NULL;

		CHECK_CUDA(cudaMalloc(&db, b.bytes));
		CHECK_CUDNN(cudnnConvolutionBackwardBias(
			NN_Base::handle,
			&alpha,
			dyDesc,
			dy.data,
			&beta,
			dbDesc,
			db
		));
		NN_Backward::opt->Run(b, db, bm);
		CHECK_CUDA(cudaFree(db));
	}
}

template <class _T>
void NN_D_Dense<_T>::CalcSize() {
	SetDesc(dxDesc, xDim);
	ReAlloc(dx, xDim);

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
		NN_Base::handle,
		wDesc,
		dyDesc,
		convDesc,
		dxDesc,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
		0,
		&dataAlgo
	));
	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
		NN_Base::handle,
		xDesc,
		dyDesc,
		convDesc,
		dwDesc,
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		0,
		&filterAlgo
	));
}



/******************************************************

					  NN_D_Flatten

*******************************************************/

template <class _T>
class NN_D_Flatten : public NN_Backward {
public:
	GPU_Mem<float>& dy;
	cudnnTensorDescriptor_t dyDesc;
	Dim& xDim;

	NN_D_Flatten(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Flatten();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Flatten<_T>::NN_D_Flatten(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Flatten"),
	dy(_dy),
	dyDesc(_dyDesc),
	xDim(p.xDim)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
		dx = dy;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Flatten<_T>::~NN_D_Flatten() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Flatten<_T>::Run() {
	dx = dy;
}

template <class _T>
void NN_D_Flatten<_T>::CalcSize() {
	SetDesc(dxDesc, xDim);
}



/******************************************************

					  NN_D_Convolution

*******************************************************/

template <class _T>
class NN_D_Convolution : public NN_Backward {
public:
	GPU_Mem<float>& x;
	GPU_Mem<float>& dy;
	Dim& xDim;
	cudnnTensorDescriptor_t xDesc;
	cudnnTensorDescriptor_t dyDesc;

	Dim& wDim;
	GPU_Mem<float>& w;
	GPU_Mem<float>& b;

	GPU_Mem<float> wm;
	GPU_Mem<float> bm;

	cudnnFilterDescriptor_t wDesc;
	cudnnTensorDescriptor_t bDesc;
	cudnnFilterDescriptor_t dwDesc;
	cudnnTensorDescriptor_t dbDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionBwdDataAlgo_t dataAlgo;
	cudnnConvolutionBwdFilterAlgo_t filterAlgo;

	bool& addBias;

	NN_D_Convolution(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Convolution();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Convolution<_T>::NN_D_Convolution(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Convolution"),
	x(p.x),
	dy(_dy),
	xDim(p.xDim),
	wDim(p.wDim),
	w(p.w),
	b(p.b),
	addBias(p.addBias)
{
	try {
		xDesc = p.xDesc;
		dyDesc = _dyDesc;
		wDesc = p.wDesc;
		bDesc = p.bDesc;
		convDesc = p.convDesc;

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
		CHECK_CUDNN(cudnnCreateFilterDescriptor(&dwDesc));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(
			dwDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NHWC,
			wDim.n,
			wDim.c,
			wDim.h,
			wDim.w));
		ReAlloc(wm, w.elements);
		CHECK_CUDA(cudaMemset(wm.data, 0, wm.bytes));

		if (addBias) {
			CHECK_CUDNN(cudnnCreateTensorDescriptor(&dbDesc));
			SetDesc(dbDesc, 1, 1, 1, wDim.n);
			ReAlloc(bm, wDim.n);
			CHECK_CUDA(cudaMemset(bm.data, 0, bm.bytes));
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Convolution<_T>::~NN_D_Convolution() {
	try {
		Release(dx);
		Release(wm);
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		CHECK_CUDNN(cudnnDestroyFilterDescriptor(dwDesc));

		if (addBias) {
			Release(bm);
			CHECK_CUDNN(cudnnDestroyTensorDescriptor(dbDesc));
		}
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Convolution<_T>::Run() {
	float* dw = NULL;
	size_t workSize = 0;
	void* workSpace = NULL;

	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		NN_Base::handle,
		wDesc,
		dyDesc,
		convDesc,
		dxDesc,
		dataAlgo,
		&workSize
	));
	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDA(cudaMalloc(&dw, w.bytes));

	CHECK_CUDNN(cudnnConvolutionBackwardData(
		NN_Base::handle,
		&alpha,
		wDesc,
		w.data,
		dyDesc,
		dy.data,
		convDesc,
		dataAlgo,
		workSpace,
		workSize,
		&beta,
		dxDesc,
		dx.data
	));

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		NN_Base::handle,
		xDesc,
		dyDesc,
		convDesc,
		dwDesc,
		filterAlgo,
		&workSize
	));
	CHECK_CUDA(cudaFree(workSpace));
	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDNN(cudnnConvolutionBackwardFilter(
		NN_Base::handle,
		&alpha,
		xDesc,
		x.data,
		dyDesc,
		dy.data,
		convDesc,
		filterAlgo,
		workSpace,
		workSize,
		&beta,
		dwDesc,
		dw
	));
	CHECK_CUDA(cudaFree(workSpace));
	NN_Backward::opt->Run(w, dw, wm);
	CHECK_CUDA(cudaFree(dw));

	if (addBias) {
		float* db = NULL;

		CHECK_CUDA(cudaMalloc(&db, b.bytes));
		CHECK_CUDNN(cudnnConvolutionBackwardBias(
			NN_Base::handle,
			&alpha,
			dyDesc,
			dy.data,
			&beta,
			dbDesc,
			db
		));
		NN_Backward::opt->Run(b, db, bm);
		CHECK_CUDA(cudaFree(db));
	}
}

template <class _T>
void NN_D_Convolution<_T>::CalcSize() {
	SetDesc(dxDesc, xDim);
	ReAlloc(dx, xDim);

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
		NN_Base::handle,
		wDesc,
		dyDesc,
		convDesc,
		dxDesc,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
		0,
		&dataAlgo
	));
	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
		NN_Base::handle,
		xDesc,
		dyDesc,
		convDesc,
		dwDesc,
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		0,
		&filterAlgo
	));
}




/******************************************************

					  NN_D_Maxpool

*******************************************************/

template <class _T>
class NN_D_Maxpool : public NN_Backward {
public:
	GPU_Mem<float>& dy;
	cudnnTensorDescriptor_t dyDesc;
	Dim& xDim;

	GPU_Mem<float>& x;
	GPU_Mem<float>& y;
	cudnnTensorDescriptor_t xDesc;
	cudnnTensorDescriptor_t yDesc;

	cudnnPoolingDescriptor_t poolDesc;

	NN_D_Maxpool(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc);
	~NN_D_Maxpool();

	void Run();
	void CalcSize();
};

template <class _T>
NN_D_Maxpool<_T>::NN_D_Maxpool(_T& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward(p.name, "NN_D_Maxpool"),
	dy(_dy),
	xDim(p.xDim),
	x(p.x),
	y(p.y)
{
	dyDesc = _dyDesc;
	xDesc = p.xDesc;
	yDesc = p.yDesc;
	poolDesc = p.poolDesc;

	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

template <class _T>
NN_D_Maxpool<_T>::~NN_D_Maxpool() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		Release(dx);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

template <class _T>
void NN_D_Maxpool<_T>::Run() {
	float alpha = 1.f;
	float beta = 0.f;
	
	CHECK_CUDNN(cudnnPoolingBackward(
		NN_Base::handle,
		poolDesc,
		&alpha,
		yDesc,
		y.data,
		dyDesc,
		dy.data,
		xDesc,
		x.data,
		&beta,
		dxDesc,
		dx.data
	));
}

template <class _T>
void NN_D_Maxpool<_T>::CalcSize() {
	SetDesc(dxDesc, xDim);
	ReAlloc(dx, xDim);
}