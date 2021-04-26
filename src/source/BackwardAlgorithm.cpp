#include "../header/BackwardAlgorithm.h"




/******************************************************

					  NN_Backward

*******************************************************/

NN_Optimizer* NN_Backward::opt = NULL;

NN_Backward::NN_Backward(const string _name, const string _layerName) :
	NN_Base("d_" + _name, _layerName)
{

}

NN_Backward::~NN_Backward() {

}

void NN_Backward::SetOptimizer(NN_Optimizer* _opt) {
	opt = _opt;
}

void NN_Backward::ClearOptimizer() {
	delete opt;
	opt = NULL;
}



/******************************************************

					  NN_D_Disperse

*******************************************************/

NN_D_Disperse::NN_D_Disperse(NN_Base& p, GPU_Mem<float>& _dy, cudnnTensorDescriptor_t _dyDesc) :
	NN_Backward("disperse", "NN_D_Disperse"),
	dy(_dy),
	dyDesc(_dyDesc)
{
	dxDesc = _dyDesc;
}

NN_D_Disperse::~NN_D_Disperse() {

}

void NN_D_Disperse::Run() {
	dx = dy;
}




/******************************************************

					  NN_D_ConcatAdd

*******************************************************/

NN_D_ConcatAdd::NN_D_ConcatAdd(vector<GPU_Mem<float>*> _dy, vector<cudnnTensorDescriptor_t> _dyDesc) :
	NN_Backward("concat_add", "NN_D_ConcatAdd"),
	dy(_dy),
	dyDesc(_dyDesc)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_D_ConcatAdd::~NN_D_ConcatAdd() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(dxDesc));
		Release(dx);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_D_ConcatAdd::Run() {
	float alpha = 1.f;
	float beta = 1.f;

	CHECK_CUDA(cudaMemset(dx.data, 0, dx.bytes));
	for (int i = 0; i < dy.size(); ++i) {
		CHECK_CUDNN(cudnnAddTensor(
			NN_Base::handle,
			&alpha,
			dyDesc[i],
			dy[i]->data,
			&beta,
			dxDesc,
			dx.data
		));
	}
}

void NN_D_ConcatAdd::CalcSize() {
	int n, h, w, c;
	int sn, sh, sw, sc;
	cudnnDataType_t dType;

	CHECK_CUDNN(cudnnGetTensor4dDescriptor(
		dyDesc[0],
		&dType,
		&n, &c, &h, &w,
		&sn, &sc, &sh, &sw
	));
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(
		dxDesc,
		CUDNN_TENSOR_NHWC,
		dType,
		n, c, h, w
	));
	ReAlloc(dx, n * h * w * c);
}