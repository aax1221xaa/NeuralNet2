#include "../header/CudaException.h"



void checkCudnn(cudnnStatus_t status, const char* file, int line) {
	if (status != CUDNN_STATUS_SUCCESS) {
		const char* errStr = cudnnGetErrorString(status);

		throw StringFormat("[CUDNN_ERROR] %s : %s : %d", file, errStr, line);
	}
}

void checkCuda(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		const char* errStr = cudaGetErrorString(err);

		throw StringFormat("[CUDA_ERROR] %s : %s : %d", file, errStr, line);
	}
}