#include "../header/Base.h"
#include <iostream>
#include <random>





int GetMatChannels(int type) {
	return type / 8;
}

int GetMatType(int type) {
	return type % 8;
}

cv::Mat MeshGrid(cv::Mat x, cv::Mat y) {
	int width = x.size[0];
	int height = y.size[0];

	if (x.type() != y.type()) throw StringFormat("[MeshGrid] x, y 타입이 틀립니다.");

	cv::Mat grid;
	int type = GetMatType(x.type());
	if (type == CV_8U) grid = GetMeshGrid<uchar>(x, y);
	else if (type == CV_8S) grid = GetMeshGrid<char>(x, y);
	else if (type == CV_16U) grid = GetMeshGrid<unsigned short>(x, y);
	else if (type == CV_16S) grid = GetMeshGrid<short>(x, y);
	else if (type == CV_32S) grid = GetMeshGrid<int>(x, y);
	else if (type == CV_32F) grid = GetMeshGrid<float>(x, y);
	else if (type == CV_64F) grid = GetMeshGrid<double>(x, y);
	else throw StringFormat("[MeshGrid] 지원하지 않는 타입입니다.");

	return grid;
}

template <typename _T>
cv::Mat GetMeshGrid(cv::Mat x, cv::Mat y) {
	int width = x.cols;
	int height = y.cols;

	cv::Mat grid = cv::Mat(cv::Size(width, height), CV_MAKE_TYPE(GetMatType(x.type()), 2));
	_T *px = x.ptr<_T>();
	_T *py = y.ptr<_T>();
	for (int y = 0; y < height; ++y) {
		_T *pGrid = grid.ptr<_T>(y);
		for (int x = 0; x < width; ++x) {
			pGrid[x * 2] = px[x];
			pGrid[x * 2 + 1] = py[y];
		}
	}

	return grid;
}

cv::Mat aRange(int start, int end, int step) {
	int length = abs(end - start) / step;
	cv::Mat arr(1, length, CV_32SC1);

	for (int i = 0; i < length; ++i) ((int*)arr.data)[i] = start + (step * i);
	
	return arr;
}

#if 1

template <typename _T>
void _argMax(uchar* src, int high, int middle, int low, uchar* dst) {
	for (int h = 0; h < high; ++h) {
		for (int L = 0; L < low; ++L) {
			int arg = 0;
			_T _max = 0;
			for (int i = 0; i < middle; ++i) {
				_T val = ((_T*)src)[h * low * middle + low * i + L];

				if (i == 0) _max = val;
				else if (val > _max) {
					_max = val;
					arg = i;
				}
			}
			((int*)dst)[h * low + L] = arg;
		}
	}
}

cv::Mat argMax(cv::Mat& src, int rank) {
	vector<int> dims;
	int ch = src.channels();

	for (int i = 0; i < src.dims; ++i) dims.push_back(src.size[i]);
	if (ch > 1) dims.push_back(ch);

	vector<int> idxDim;
	int upIter = 1;
	int downIter = 1;

	for (int i = 0, j = 0; i < dims.size(); ++i) {
		if (rank != i) idxDim.push_back(dims[i]);
		if (i > rank) downIter *= dims[i];
		if (i < rank) upIter *= dims[i];
	}

	cv::Mat idx(idxDim, CV_32SC1);
	if (src.depth() == CV_8U) _argMax<uchar>(src.data, upIter, dims[rank], downIter, idx.data);
	else if(src.depth() == CV_32S) _argMax<int>(src.data, upIter, dims[rank], downIter, idx.data);
	else if(src.depth() == CV_32F) _argMax<float>(src.data, upIter, dims[rank], downIter, idx.data);
	else if(src.depth() == CV_64F) _argMax<double>(src.data, upIter, dims[rank], downIter, idx.data);
	else {
		throw StringFormat("[argMax] 지원 되지 않는 Mat 형식 입니다.");
	}
	
	/*
	   [[[[1, 2, 3], [4, 5, 6]],
		 [[6, 7, 8], [9, 8, 7]],
		 [[6, 5, 4], [3, 2, 1]]],
		[[[1, 2, 3], [4, 5, 6]],
		 [[6, 7, 8], [9, 8, 7]],
		 [[6, 5, 4], [3, 2, 1]]]]	[2, 3, 2, 3] 

		 rank 0 : [[[0, 0, 0], [0, 0, 0], [[0, 0, 0]], [[0, 0, 0], [0, 0, 0], [[0, 0, 0]], [[0, 0, 0], [0, 0, 0], [[0, 0, 0]]]
		 rank 1 : [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]
		 rank 2 : [[[1, 1, 1], [1, 1, 1], [1, 1, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 0]]]
		 rank 3 : [[[2, 2], [2, 0], [0, 0]], [[2, 2], [2, 0], [0, 0]]]
	*/
	
	return idx;
}
#else


cv::Mat argMax(cv::Mat& src, int rank) {
	void* srcMem = NULL;
	size_t memSize = src.total() * src.elemSize();

	CHECK_CUDA(cudaMalloc(&srcMem, memSize));
	CHECK_CUDA(cudaMemcpy(srcMem, src.data, memSize, cudaMemcpyHostToDevice));

	vector<int> idxDims;
	int highiter = 1;
	int lowiter = 1;

	for (int i = 0; i < src.dims; ++i) {
		if (rank != i) idxDims.push_back(src.size[i]);
		if (i > rank) lowiter *= src.size[i];
		if (i < rank) highiter *= src.size[i];
	}

	cv::Mat idx(idxDims, CV_32SC1);
	int* dstMem = NULL;

	CHECK_CUDA(cudaMalloc(&dstMem, idx.total() * sizeof(int)));
	CHECK_CUDA(ArgMax(srcMem, dstMem, src.depth(), highiter, src.size[rank], lowiter));

	CHECK_CUDA(cudaMemcpy(idx.data, dstMem, idx.total() * sizeof(int), cudaMemcpyDeviceToHost));

	CHECK_CUDA(cudaFree(srcMem));
	CHECK_CUDA(cudaFree(dstMem));

	return idx;
}

#endif


cv::Mat nrand(vector<int> dims, int low, int high) {
	cv::Mat src(dims, CV_32SC1);
	random_device rd;
	mt19937 gen(rd());

	uniform_int_distribution<int> dis(low, high);
	for (int i = 0; i < src.total(); ++i) ((int*)src.data)[i] = dis(gen);

	return src;
}

bool StrAllCompare(vector<string> strs) {
	bool isEqual = false;

	for (int i = 0; i < strs.size(); ++i) {
		for (int j = i + 1; j < strs.size(); ++j) {
			if (strs[i].compare(strs[j]) == 0) isEqual = true;
		}
	}

	return isEqual;
}