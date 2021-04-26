#pragma once

#include "ForwardAlgorithm.h"


/******************************************************

						   NN

*******************************************************/

class NN {
protected:
	int state;

	enum {
		CREATED,
		SETED_IO,
		BUILT_MODEL
	};

public:


	vector<NN_Forward*> _input;
	vector<NN_Forward*> _forward;
	vector<NN_Forward*> _output;
	vector<NN_Forward*> _loss;
	vector<NN_Backward*> _backward;

	int tLayer;

	NN();
	~NN();
	
	NN_Forward* push(NN_Forward* layer);
	void set_io(vector<NN_Forward*> input, vector<NN_Forward*> output);
	void build(NN_Optimizer* opt, vector<NN_Forward*> loss);
	cv::Mat batch_train(vector<cv::Mat> x, vector<cv::Mat> y_true);
	vector<cv::Mat> batch_predict(vector<cv::Mat> x);
	void SaveWeight(string path);
	void LoadWeight(string path);

	void PrintForward();
	void PrintBackward();
};