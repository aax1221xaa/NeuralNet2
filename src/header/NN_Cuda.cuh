#ifndef NN_CUDA_CUH
#define NN_CUDA_CUH


#include <cuda.h>
#include <cuda_runtime.h>


cudaError_t LikelieHoodCE(float *y_pred, float *y_true, float* loss, int n, int h, int w, int c);
cudaError_t BinaryCE(float* y_pred, float* y_true, float* loss, int n, int h, int w, int c);
cudaError_t D_CrossEntropy(float* y_pred, float* y_true, float* dx, int elemSize, int batch);
cudaError_t ReduceSum(float* arr, int amount);
cudaError_t SGD(float* weight, float* wGrad, float* wMomentum, float lRate, float mCoef, int amount);

#endif