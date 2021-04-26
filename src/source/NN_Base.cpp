#include "../header/NN_Base.h"




/******************************************************

				  NN_TensorDescript

*******************************************************/

void SetDesc(cudnnTensorDescriptor_t desc, int n, int h, int w, int c) {
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(
		desc,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		n,
		c,
		h,
		w
	));
}

void SetDesc(cudnnTensorDescriptor_t desc, Dim dim) {
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(
		desc,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		dim.n,
		dim.c,
		dim.h,
		dim.w
	));
}

void GetDescDim(cudnnTensorDescriptor_t desc, Dim &dim) {
	cudnnDataType_t dataType;
	int sn, sh, sw, sc;

	CHECK_CUDNN(cudnnGetTensor4dDescriptor(
		desc,
		&dataType,
		&dim.n,
		&dim.c,
		&dim.h,
		&dim.w,
		&sn,
		&sc,
		&sh,
		&sw
	));
}




/******************************************************

						NN_Base

*******************************************************/

cudnnHandle_t NN_Base::handle;
bool NN_Base::errorFlag = false;
bool NN_Base::changeInput = false;
int NN_Base::mode = NN_Base::NONE_MODE;

void NN_Base::CreateHandle() {
	CHECK_CUDNN(cudnnCreate(&handle));
}

void NN_Base::DestroyHandle() {
	CHECK_CUDNN(cudnnDestroy(handle));
}

NN_Base::NN_Base(const string _name, const string _layerName) :
	name(_name),
	layerName(_layerName)
{

}

void NN_Base::Run() {

}

void NN_Base::CalcSize() {

}

NN_Base::~NN_Base() {

}

void NN_Base::PrintLayerInfo() {

}