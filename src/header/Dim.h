#ifndef DIM_H
#define DIM_H

#include <vector>
#include <opencv2\opencv.hpp>

using namespace std;


#define NONE		-1


class Dim {
public:
	int n;
	int h;
	int w;
	int c;

	Dim();
	Dim(const cv::Mat& mat);
	Dim(int _n, int _h = 1, int _w = 1, int _c = 1);

	void SetDim(const cv::Mat& mat);
	void SetDim(int _n, int _h = 1, int _w = 1, int _c = 1);	
	size_t GetTotalSize();

	int& operator[](int idx);
	bool operator!=(const Dim dim);
};


#endif