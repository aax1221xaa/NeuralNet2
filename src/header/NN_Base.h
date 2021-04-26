#pragma once

#include "Base.h"


using namespace std;




/******************************************************

					    GPU_Mem

*******************************************************/

template <typename _T>
struct GPU_Mem {
	_T *data;
	size_t elements;
	size_t bytes;

	GPU_Mem() {
		data = NULL;
		elements = 0;
		bytes = 0;
	}
};

template <typename _T>
void ReAlloc(GPU_Mem<_T>& mem, size_t _elements) {
	CHECK_CUDA(cudaFree(mem.data));
	mem.elements = _elements;
	mem.bytes = sizeof(_T) * _elements;
	CHECK_CUDA(cudaMalloc(&mem.data, mem.bytes));
}

template <typename _T>
void ReAlloc(GPU_Mem<_T>& mem, Dim dim) {
	CHECK_CUDA(cudaFree(mem.data));
	if (dim.n < 1 || dim.h < 1 || dim.w < 1 || dim.c < 1) {
		throw StringFormat("[GPU_Mem] 입력 크기 [%d %d %d %d] 가 1보다 작습니다.",
			dim.n, dim.h, dim.w, dim.c);
	}
	mem.elements = dim.GetTotalSize();
	mem.bytes = sizeof(_T) * mem.elements;
	CHECK_CUDA(cudaMalloc(&mem.data, mem.bytes));
}

template <typename _T>
void Release(GPU_Mem<_T>& mem) {
	CHECK_CUDA(cudaFree(mem.data));
	mem.bytes = 0;
	mem.elements = 0;
	mem.data = NULL;
}



/******************************************************

					NN_TensorDescript

*******************************************************/

void SetDesc(cudnnTensorDescriptor_t desc, int n, int h, int w, int c);
void SetDesc(cudnnTensorDescriptor_t desc, Dim dim);
void GetDescDim(cudnnTensorDescriptor_t desc, Dim &dim);



/******************************************************

					NN_Base

*******************************************************/

class NN_Base {
public:
	enum {
		VALID,
		SAME,

		INPUT,
		HIDDEN,
		CONCAT,
		OUTPUT,
		LOSS,

		NONE_MODE,
		TRAIN_MODE,
		INFERENCE_MODE
	};

	static cudnnHandle_t handle;

	const string name;
	const string layerName;

	static bool errorFlag;
	static bool changeInput;
	static int mode;

	static void CreateHandle();
	static void DestroyHandle();

	NN_Base(const string _name, const string _layerName);
	virtual void Run();
	virtual void CalcSize();
	virtual ~NN_Base();
	virtual void PrintLayerInfo();
};