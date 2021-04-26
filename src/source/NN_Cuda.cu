#include "../header/NN_Cuda.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_launch_parameters.h>
#include <device_functions.h>


#define TILE		32
#define PAGE		1024
#define EPSILON		1e-7


__device__ float __CalculateLikelieHoodCE(float *y_pred, float *y_true, int c) {
	float loss = 0.f;

	for (int i = 0; i < c; ++i) {
		loss += y_true[i] * __logf(y_pred[i] + EPSILON);
	}

	return loss;
}

__device__ float __CalculateBinaryCE(float* y_pred, float* y_true, int channel) {
	float cost = 0.f;

	for (int i = 0; i < channel; ++i) {
		cost -= y_true[i] * __logf(y_pred[i] + EPSILON) + (1.f - y_true[i]) * __logf(1.f - y_pred[i] + EPSILON);
	}

	return cost;
}

__device__ void __WarpReduceSum(volatile float* src, const int thread) {
	src[thread] += src[thread + 32];
	src[thread] += src[thread + 16];
	src[thread] += src[thread + 8];
	src[thread] += src[thread + 4];
	src[thread] += src[thread + 2];
	src[thread] += src[thread + 1];
}




__global__ void __LikelieHoodCE(float *y_pred, float *y_true, float *loss, int n, int h, int w, int c) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (n * h * w)) {
		float *_y_pred = y_pred + c * idx;
		float *_y_true = y_true + c * idx;

		loss[idx] = __CalculateLikelieHoodCE(_y_pred, _y_true, c);
	}
}

__global__ void __BinaryCE(float* y_pred, float* y_true, float* loss, int n, int h, int w, int c) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (n * h * w)) {
		float* _y_pred = y_pred + c * idx;
		float* _y_true = y_true + c * idx;

		loss[idx] = __CalculateBinaryCE(_y_pred, _y_true, c);
	}
}

__global__ void __ReduceSum(float *src, int n) {
	const int idx = threadIdx.x;

	float sum = 0.f;
	__shared__ float shareSum[PAGE];

	for (int i = idx; i < n; i += PAGE) {
		sum += src[i];
	}
	
	shareSum[idx] = sum;
	__syncthreads();

	if (idx < 512) shareSum[idx] += shareSum[idx + 512]; __syncthreads();
	if (idx < 256) shareSum[idx] += shareSum[idx + 256]; __syncthreads();
	if (idx < 128) shareSum[idx] += shareSum[idx + 128]; __syncthreads();
	if (idx < 64) shareSum[idx] += shareSum[idx + 64]; __syncthreads();
	if (idx < 32) __WarpReduceSum(shareSum, idx);
	if (idx < 1) src[idx] = -shareSum[idx];
}

__global__ void __SGD(float* weight, float* wGrad, float* wMomentum, float lRate, float mCoef, int amount) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float gradient = 0;

	if (idx < amount) {
		gradient = wMomentum[idx] = wGrad[idx]* lRate + wMomentum[idx] * mCoef;
		weight[idx] -= gradient;
	}
}

__global__ void __D_CrossEntropy(float* y_pred, float* y_true, float* dx, int elemSize, int batch) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < elemSize) {
		dx[idx] = (y_pred[idx] - y_true[idx]) / batch;
	}
}




cudaError_t LikelieHoodCE(float* y_pred, float* y_true, float* loss, int n, int h, int w, int c) {
	dim3 threads(PAGE);
	dim3 blocks(((n * h * w) + PAGE - 1) / PAGE);
	cudaError_t err;

	__LikelieHoodCE << <blocks, threads >> > (y_pred, y_true, loss, n, h, w, c);
	err = cudaGetLastError();
	cudaDeviceSynchronize();

	return err;
}

cudaError_t BinaryCE(float* y_pred, float* y_true, float* loss, int n, int h, int w, int c) {
	__BinaryCE << <((n * h * w) + PAGE - 1) / PAGE, PAGE >> > (y_pred, y_true, loss, n, h, w, c);
	cudaDeviceSynchronize();

	return cudaGetLastError();
}

cudaError_t D_CrossEntropy(float* y_pred, float* y_true, float* dx, int elemSize, int batch) {
	dim3 threads(PAGE);
	dim3 blocks((elemSize + PAGE - 1) / PAGE);

	__D_CrossEntropy << <blocks, threads >> > (y_pred, y_true, dx, elemSize, batch);
	cudaDeviceSynchronize();

	return cudaGetLastError();
}

cudaError_t ReduceSum(float *arr, int amount) {
	__ReduceSum<<<1, PAGE>>>(arr, amount);
	cudaDeviceSynchronize();

	return cudaGetLastError();
}

cudaError_t SGD(float* weight, float* wGrad, float* wMomentum, float lRate, float mCoef, int amount) {
	dim3 threads(PAGE);
	dim3 blocks((amount + PAGE - 1) / PAGE);

	__SGD<<<blocks, threads>>>(weight, wGrad, wMomentum, lRate, mCoef, amount);
	cudaError_t err = cudaGetLastError();
	cudaDeviceSynchronize();

	return err;
}