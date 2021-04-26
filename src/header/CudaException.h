#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <string>
#include <memory>
#include <vector>


using namespace std;



#define CHECK_CUDNN(err) (checkCudnn(err, __FILE__, __LINE__))
#define CHECK_CUDA(err) (checkCuda(err, __FILE__, __LINE__))


void checkCudnn(cudnnStatus_t status, const char* file, int line);
void checkCuda(cudaError_t err, const char* file, int line);


template<typename ... argTypes>
string StringFormat(const char* format, argTypes ... args) {
	size_t size = (size_t)snprintf(nullptr, 0, format, args ...) + 1;
	unique_ptr<char[]> buffer(new char[size]);

	snprintf(buffer.get(), size, format, args...);

	return string(buffer.get(), buffer.get() + size - 1);
}