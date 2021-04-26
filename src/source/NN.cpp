#include "../header/NN.h"
#include <ctime>


/******************************************************

						   NN

*******************************************************/

NN::NN() {
	state = CREATED;
	tLayer = 0;
	
	try {
		NN_Base::CreateHandle();
	}
	catch (string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN::~NN() {
	for (NN_Forward* p : _forward) delete p;
	for (NN_Forward* p : _loss) delete p;
	for (NN_Backward* p : _backward) delete p;

	try {
		NN_Base::DestroyHandle();
		NN_Backward::ClearOptimizer();
	}
	catch (string& e) {
		cout << e << endl;
	}
}

NN_Forward* NN::push(NN_Forward* layer) {
	if (NN_Base::errorFlag) {
		for (NN_Forward* p : _forward) delete p;
		delete layer;
		layer = NULL;
	}
	else {
		_forward.push_back(layer);
		++tLayer;
	}
	return layer;
}

void NN::set_io(vector<NN_Forward*> input, vector<NN_Forward*> output) {
	_input = input;
	_output = output;
	state = SETED_IO;
}

void NN::build(NN_Optimizer* opt, vector<NN_Forward*> loss) {
	vector<string> names;
	for (NN_Forward* p : _forward) names.push_back(p->name);
	
	NN_Backward::SetOptimizer(opt);
	_loss = loss;

	if (state != SETED_IO) {
		throw StringFormat("[NN::build] ��/��� ���̾ �������� �ʾҰų� �̹� ���� ���� �Ǿ� ���ϴ�.");
	}
	else if (StrAllCompare(names)) {
		throw StringFormat("[NN::build] ������ ����� �̸��� ���� ���̾ �ֽ��ϴ�.");
	}

	for (NN_Forward* p : _loss) p->CreateBackwardModule(_backward);
	for (vector<NN_Forward*>::reverse_iterator p = _forward.rbegin(); p != _forward.rend(); ++p) {
		(*p)->CreateBackwardModule(_backward);
		if (NN_Base::errorFlag) throw "[NN::build] ��� ���� �� ������ �߻� �߽��ϴ�.";
	}
	state = BUILT_MODEL;
}

cv::Mat NN::batch_train(vector<cv::Mat> x, vector<cv::Mat> y_true) {
	if (x.size() != _input.size())
		throw StringFormat("[NN::batch_train] �Է� ��� %d�� ���� %d �̹Ƿ� ���� �ʽ��ϴ�.",
			(int)_input.size(), (int)x.size());
	else if (y_true.size() != _output.size())
		throw StringFormat("[NN::batch_train] ��� ��� %d�� ���� %d �̹Ƿ� ���� �ʽ��ϴ�.",
			(int)_output.size(), (int)y_true.size());
	else if (NN_Base::errorFlag)
		throw StringFormat("[NN::batch_train] �̹� ���̾� ���� ���� ������ �߻� �Ǿ����ϴ�.");

	for (int i = 0; i < x.size(); ++i) _input[i]->SetInput(x[i]);

	cv::Mat loss;
	NN_Base::mode = NN_Base::TRAIN_MODE;

	for (NN_Forward* p : _input) p->CalcSize();

	if (NN_Base::changeInput) {
		for (NN_Forward* p : _forward) p->CalcSize();
		for (NN_Forward* p : _loss) p->CalcSize();
		for (NN_Backward* p : _backward) p->CalcSize();
		NN_Base::changeInput = false;
	}

	for (NN_Forward* p : _forward) p->Run();
	for (int i = 0; i < y_true.size(); ++i) loss.push_back(_loss[i]->CalcLoss(y_true[i]));
	for (NN_Backward* p : _backward) p->Run();

	return loss;
}

vector<cv::Mat> NN::batch_predict(vector<cv::Mat> x) {
	if (x.size() != _input.size())
		throw StringFormat("[NN::batch_predict] �Է� ��� %d�� ���� %d �̹Ƿ� ���� �ʽ��ϴ�.",
			(int)_input.size(), (int)x.size());
	else if (NN_Base::errorFlag)
		throw StringFormat("[NN::batch_predict] �̹� ���̾� ���� ���� ������ �߻� �Ǿ����ϴ�.");

	for (int i = 0; i < x.size(); ++i) _input[i]->SetInput(x[i]);

	NN_Base::mode = NN_Base::INFERENCE_MODE;
	
	for (NN_Forward* p : _input) p->CalcSize();
	if (NN_Base::changeInput) {
		for (NN_Forward* p : _forward) p->CalcSize();
		NN_Base::changeInput = false;
	}
	for (NN_Forward* p : _forward) p->Run();

	vector<cv::Mat> _y;
	for (NN_Forward* p : _output) _y.push_back(p->GetOutput());

	return _y;
}

void NN::SaveWeight(string path) {
	if (state != BUILT_MODEL) throw StringFormat("[NN::SaveWeight] ���� ���� ���� �ʾҽ��ϴ�.");
	cv::FileStorage fs(path, cv::FileStorage::WRITE);
	
	time_t date;
	tm base_date;
	char ascTime[100];

	time(&date);
	localtime_s(&base_date, &date);
	asctime_s(ascTime, &base_date);

	fs << "date" << ascTime;
	for (NN_Forward* p : _forward) p->SaveWeight(fs);
	fs.release();

	printf("[NN::SaveWeight] %s ������ ���������� ���� �Ͽ����ϴ�.\n", path.c_str());
}

void NN::LoadWeight(string path) {
	if (state != SETED_IO) throw StringFormat("[NN::LoadWeight] ���� ���� ���� �ʾҽ��ϴ�.");
	cv::FileStorage fs(path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		throw StringFormat("[NN::LoadWeight] %s ������ ���� ���߽��ϴ�.", path.c_str());
	}

	for (NN_Forward* p : _forward) p->LoadWeight(fs);
	fs.release();

	printf("[NN::LoadWeight] %s ������ ���������� �ε� �Ͽ����ϴ�.\n", path.c_str());
}

void NN::PrintForward() {
	for (NN_Forward* p : _forward) p->PrintLayerInfo();
	for (NN_Forward* p : _loss) p->PrintLayerInfo();
}

void NN::PrintBackward() {
	for (NN_Backward* p : _backward) cout << p->layerName << endl;
}