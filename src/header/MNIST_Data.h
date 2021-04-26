#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include "../header/Base.h"

class MNIST_Data{
protected:
	static cv::Mat LoadImages(const char* path);
	static cv::Mat LoadLabels(const char* path);
	static int ReadHeader(FILE* fp);

public:
	cv::Mat image;
	cv::Mat label;

	const char* imagePath;
	const char* labelPath;

	MNIST_Data(const char *_imagePath, const char *_labelPath);
	~MNIST_Data();

	void Mining();
	void GetSample(cv::Mat& x, cv::Mat& y, int batch);
};

#endif