#include <opencv2\opencv.hpp>
#include <iostream>
#include <time.h>
#include "./header/Ptr.h"
#include "./header/NN.h"
#include "./header/ForwardAlgorithm.h"
#include "./header/MNIST_Data.h"
#include <vld.h>


using namespace std;


#define TEST		false
#define TRAIN		true



#if (TEST)

int main() {
	try {
		MNIST_Data mnist("E:\\data_set\\train-images.idx3-ubyte", "E:\\data_set\\train-labels.idx1-ubyte");
		mnist.Mining();

		NN nn;
		NN_Forward* x = NULL;

		NN_Forward* x1 = nn << new NN_Input({ NONE, 28, 28, 1 }, "input1");
		x = nn << new NN_Convolution(16, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv1_1", x1);
		x = nn << new NN_ReLU("relu1_1", x);
		x = nn << new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool1_1", x);
		x = nn << new NN_Convolution(32, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv1_2", x);
		x = nn << new NN_ReLU("relu1_2", x);
		NN_Forward* out1 = nn << new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool1_2", x);	// [NONE, 4, 4, 32]

		NN_Forward* x2 = nn << new NN_Input({ NONE, 28, 28, 1 }, "input2");
		x = nn << new NN_Convolution(16, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv2_1", x2);
		x = nn << new NN_ReLU("relu2_1", x);
		x = nn << new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool2_1", x);
		x = nn << new NN_Convolution(32, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv2_2", x);
		x = nn << new NN_ReLU("relu2_2", x);
		NN_Forward* out2 = nn << new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool2_2", x);	// [NONE, 4, 4, 32]

		x = nn << new NN_ConcatAdd("concat", { out1, out2 });
		NN_Forward* branchOut = nn << new NN_Flatten("flatten", x);

		x = nn << new NN_Dense(256, new NN_HeInit, true, "dense3_1", branchOut);
		x = nn << new NN_Dropout(0.5f, "dropout3_1", x);
		x = nn << new NN_ReLU("relu3_1", x);
		x = nn << new NN_Dense(128, new NN_HeInit, true, "dense3_2", x);
		x = nn << new NN_Dropout(0.5f, "dropout3_2", x);
		x = nn << new NN_ReLU("relu3_2", x);
		x = nn << new NN_Dense(10, new NN_HeInit, true, "dense3_3", x);
		NN_Forward* y_out1 = nn << new NN_Softmax("softmax3_1", x);

		x = nn << new NN_Dense(256, new NN_HeInit, true, "dense4_1", branchOut);
		x = nn << new NN_Dropout(0.5f, "dropout4_1", x);
		x = nn << new NN_ReLU("relu4_1", x);
		x = nn << new NN_Dense(128, new NN_HeInit, true, "dense4_2", x);
		x = nn << new NN_Dropout(0.5f, "dropout4_2", x);
		x = nn << new NN_ReLU("relu_4_2", x);
		x = nn << new NN_Dense(10, new NN_HeInit, true, "dense4_3", x);
		NN_Forward* y_out2 = nn << new NN_Softmax("softmax4_1", x);

		nn.set_io({ x1, x2 }, { y_out1, y_out2 });
		nn.build(
			new NN_SGD(0.001, 0.9),
			{
				new NN_CrossEntropy("cross_entropy", y_out1),
				new NN_CrossEntropy("cross_entropy2", y_out2)
			});
		nn.PrintForward();

		cv::Mat images, labels;
		const int batch = 64;

		for (int i = 0; i < 10; ++i) {
			mnist.GetSample(images, labels, batch);

			images.convertTo(images, CV_32FC1);
			cv::Mat _x = (images / 255.f) - 0.5f;

			cv::Mat _y = cv::Mat::zeros(batch, 10, CV_32FC1);
			for (int i = 0; i < batch; ++i) {
				uchar* lpt = labels.ptr<uchar>(i);
				float* _ypt = _y.ptr<float>(i);

				_ypt[*lpt] = 1.f;
			}

			cv::Mat loss = nn.batch_train({ _x, _x }, { _y, _y });
			printf("%d : Loss = %f, %f\n", i, loss.at<float>(0), loss.at<float>(1));
		}
	}
	catch (const char* e) {
		cout << e << endl;
		return -1;
	}
	catch (const string e) {
		cout << e << endl;
		return -1;
	}
	return 0;
}

#else
#if (!TRAIN)

int main() {
	try {
		
		MNIST_Data mnist("E:\\data_set\\t10k-images.idx3-ubyte", "E:\\data_set\\t10k-labels.idx1-ubyte");
		mnist.Mining();

		Ptr<NN> nn = new NN;
		NN_Forward* x = NULL;

		NN_Forward* x_in = x = nn->push(new NN_Input({ NONE, 28, 28, 1 }, "input"));
		x = nn->push(new NN_Convolution(16, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv_1", x));
		x = nn->push(new NN_ReLU("relu_1", x));
		x = nn->push(new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool_1", x));
		x = nn->push(new NN_Convolution(32, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv_2", x));
		x = nn->push(new NN_ReLU("relu_2", x));
		x = nn->push(new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool_2", x));	// [NONE, 4, 4, 32]
		x = nn->push(new NN_Flatten("flatten", x));
		x = nn->push(new NN_Dense(256, new NN_HeInit, true, "dense_1", x));
		x = nn->push(new NN_Dropout(0.5f, "dropout_1", x));
		x = nn->push(new NN_ReLU("relu_3", x));
		x = nn->push(new NN_Dense(128, new NN_HeInit, true, "dense_2", x));
		x = nn->push(new NN_Dropout(0.5f, "dropout_2", x));
		x = nn->push(new NN_ReLU("relu_4", x));
		x = nn->push(new NN_Dense(10, new NN_HeInit, true, "dense_3", x));
		NN_Forward* y_out = x = nn->push(new NN_Softmax("softmax", x));

		nn->set_io({ x_in }, { y_out });
		nn->LoadWeight(".\\weights\\mnist.xml");

		cv::Mat images, labels;
		
		for (int i = 0; i < 100; ++i) {
			cv::Mat img = cv::Mat::zeros(28 * 5, 28 * 5, CV_8UC1);
			mnist.GetSample(images, labels, 25);

			cv::Mat _x;
			images.convertTo(_x, CV_32FC1);
			_x = (_x / 255.f) - 0.5f;

			vector<cv::Mat> result = nn->batch_predict({ _x });
			cv::Mat labelArg = argMax(result[0], 1);

			for (int y = 0; y < 5; ++y) {
				for (int x = 0; x < 5; ++x) {
					for (int h = 0; h < 28; ++h) {
						uchar* pSrc = images.ptr<uchar>(y * 5 + x, h);
						uchar* pDst = img.ptr<uchar>((y * 28 + h));
						for (int w = 0; w < 28; ++w) {
							pDst[x * 28 + w] = pSrc[w];
						}
					}
				}
			}
			cv::imshow("IMAGES", img);
			for (int y = 0; y < 5; ++y) {
				for (int x = 0; x < 5; ++x) {
					printf("%d ", ((int*)labelArg.data)[y * 5 + x]);
				}
				printf("\n");
			}
			printf("\n");
			if (cv::waitKey() == 27) break;
		}
		cv::destroyAllWindows();

	}
	catch (const char *e) {
		cout << e << endl;
	}

	return 0;
}

#else


int main() {
	try {
		MNIST_Data train_sample("E:\\data_set\\train-images.idx3-ubyte", "E:\\data_set\\train-labels.idx1-ubyte");

		train_sample.Mining();

		Ptr<NN> nn = new NN;
		NN_Forward* x = NULL;

		NN_Forward* x_in = x = nn->push(new NN_Input({ NONE, 28, 28, 1 }, "input"));
		x = nn->push(new NN_Convolution(16, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv_1", x));
		x = nn->push(new NN_ReLU("relu_1", x));
		x = nn->push(new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool_1", x));
		x = nn->push(new NN_Convolution(32, { 5, 5 }, { 1, 1 }, { 1, 1 }, new NN_HeInit, NN_Base::VALID, true, "conv_2", x));
		x = nn->push(new NN_ReLU("relu_2", x));
		x = nn->push(new NN_Maxpool({ 2, 2 }, { 2, 2 }, NN_Base::VALID, "maxpool_2", x));	// [NONE, 4, 4, 32]
		x = nn->push(new NN_Flatten("flatten", x));
		x = nn->push(new NN_Dense(256, new NN_HeInit, true, "dense_1", x));
		x = nn->push(new NN_Dropout(0.5f, "dropout_1", x));
		x = nn->push(new NN_ReLU("relu_3", x));
		x = nn->push(new NN_Dense(128, new NN_HeInit, true, "dense_2", x));
		x = nn->push(new NN_Dropout(0.5f, "dropout_2", x));
		x = nn->push(new NN_ReLU("relu_4", x));
		x = nn->push(new NN_Dense(10, new NN_HeInit, true, "dense_3", x));
		NN_Forward* y_out = x = nn->push(new NN_Softmax("softmax", x));

		nn->set_io({ x_in }, { y_out });
		nn->build(new NN_SGD(0.001, 0.9), { new NN_CrossEntropy("cross_entropy", y_out) });
		nn->PrintForward();

		cv::Mat train_imgs, train_labels;
		const int batch = 64;

		cv::Mat losses;
		for (int i = 0; i < 10000; ++i) {
			train_sample.GetSample(train_imgs, train_labels, batch);
			
			train_imgs.convertTo(train_imgs, CV_32FC1);
			cv::Mat train_x = (train_imgs / 255.f) - 0.5f;
			cv::Mat train_y = cv::Mat::zeros(batch, 10, CV_32FC1);
		
			for (int j = 0; j < batch; ++j) {
				uchar* lpt = train_labels.ptr<uchar>(j);
				float* _ypt = train_y.ptr<float>(j);

				_ypt[*lpt] = 1.f;
			}

			vector<float> loss = nn->batch_train({ train_x }, { train_y });			
			
			losses.push_back(loss[0]);
			cout << i << " : " << "Loss = " << cv::mean(losses)[0] << endl;

		}
		nn->SaveWeight("mnist.xml");
	}
	catch (const char* e) {
		cout << e << endl;
	}

	return 0;
}

#endif
#endif










































