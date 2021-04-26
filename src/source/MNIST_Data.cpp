#include "../header/MNIST_Data.h"
#include <iostream>


using namespace std;



cv::Mat MNIST_Data::LoadImages(const char* path) {
	FILE* fp = NULL;
	int magicNum = 0;
	int dims[3];

	fopen_s(&fp, path, "rb");
	if (fp == NULL)
		throw StringFormat("[MNIST_Data::LoadImages] %s이미지 경로를 열지 못 했습니다.",
			path);
	
	magicNum = ReadHeader(fp);
	for (int& num : dims) num = ReadHeader(fp);

	cout << "************************************************" << endl;
	cout << "                   load image                   " << endl;
	cout << "************************************************" << endl;
	cout << StringFormat("magic number = %d", magicNum) << endl;
	cout << StringFormat("dims = [%d %d %d]", dims[0], dims[1], dims[2]) << endl;

	cv::Mat images(3, dims, CV_8UC1);

	fread_s(images.data, sizeof(uchar) * images.total(), sizeof(uchar), images.total(), fp);
	fclose(fp);

	cout << "Loading Image is done!" << endl << endl;

	return images;
}

cv::Mat MNIST_Data::LoadLabels(const char* path) {
	FILE* fp = NULL;
	int magicNum = 0;
	int amount = 0;

	fopen_s(&fp, path, "rb");
	if (fp == NULL)
		throw StringFormat("[MNIST_Data::LoadLabels] %s라벨 경로를 열지 못 했습니다.",
			path);

	magicNum = ReadHeader(fp);
	amount = ReadHeader(fp);

	cout << "************************************************" << endl;
	cout << "                   load label                   " << endl;
	cout << "************************************************" << endl;
	cout << StringFormat("magic number = %d", magicNum) << endl;
	cout << StringFormat("dims = [%d 1]", amount) << endl;

	cv::Mat labels(amount, 1, CV_8UC1);

	fread_s(labels.data, sizeof(uchar) * labels.total(), sizeof(uchar), labels.total(), fp);
	fclose(fp);

	cout << "Loading label is done!" << endl << endl;

	return labels;
}

int MNIST_Data::ReadHeader(FILE* fp) {
	union SwapBits {
		uchar bit8[4];
		int bit32;
	}data;

	int i = 4;
	while (i--) {
		fread_s(&data.bit8[i], sizeof(uchar), sizeof(uchar), 1, fp);
	}

	return data.bit32;
}

MNIST_Data::MNIST_Data(const char* _imagePath, const char* _labelPath) :
	imagePath(_imagePath),
	labelPath(_labelPath)
{

}

MNIST_Data::~MNIST_Data(){

}

void MNIST_Data::Mining() {
	image = LoadImages(imagePath);
	label = LoadLabels(labelPath);

	if (image.size[0] != label.size[0]) {
		image = cv::Mat();
		label = cv::Mat();

		throw StringFormat("[MNIST_Data::Mining] 이미지(%d) 와 라벨(%d) 개수가 틀립니다.",
			(int)image.size[0], (int)label.size[0]);
	}
}

void MNIST_Data::GetSample(cv::Mat& x, cv::Mat& y, int batch) {
	int n = image.size[0];
	int h = image.size[1];
	int w = image.size[2];

	int imgDim[3] = { batch, h, w };
	
	x = cv::Mat(3, imgDim, CV_8UC1);
	y = cv::Mat::zeros(batch, 1, CV_8UC1);

	cv::Mat select(1, batch, CV_32SC1);
	cv::RNG rng(cvGetTickCount());
	
	rng.fill(select, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(n));

	for (int i = 0; i < batch; ++i) {
		uchar* spt = image.ptr<uchar>(((int*)select.data)[i]);
		uchar* xpt = x.ptr<uchar>(i);

		memcpy_s(xpt, sizeof(uchar) * h * w, spt, sizeof(uchar) * w * h);

		uchar* lpt = label.ptr<uchar>(((int*)select.data)[i]);
		uchar* ypt = y.ptr<uchar>(i);
		
		*ypt = *lpt;
	}
}