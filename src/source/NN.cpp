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
		throw StringFormat("[NN::build] 입/출력 레이어를 설정하지 않았거나 이미 모델이 빌드 되었 습니다.");
	}
	else if (StrAllCompare(names)) {
		throw StringFormat("[NN::build] 동일한 사용자 이름을 가진 레이어가 있습니다.");
	}

	for (NN_Forward* p : _loss) p->CreateBackwardModule(_backward);
	for (vector<NN_Forward*>::reverse_iterator p = _forward.rbegin(); p != _forward.rend(); ++p) {
		(*p)->CreateBackwardModule(_backward);
		if (NN_Base::errorFlag) throw "[NN::build] 모듈 생성 중 오류가 발생 했습니다.";
	}
	state = BUILT_MODEL;
}

cv::Mat NN::batch_train(vector<cv::Mat> x, vector<cv::Mat> y_true) {
	if (x.size() != _input.size())
		throw StringFormat("[NN::batch_train] 입력 노드 %d와 샘플 %d 이므로 맞지 않습니다.",
			(int)_input.size(), (int)x.size());
	else if (y_true.size() != _output.size())
		throw StringFormat("[NN::batch_train] 출력 노드 %d와 샘플 %d 이므로 맞지 않습니다.",
			(int)_output.size(), (int)y_true.size());
	else if (NN_Base::errorFlag)
		throw StringFormat("[NN::batch_train] 이미 레이어 생성 도중 오류가 발생 되었습니다.");

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
		throw StringFormat("[NN::batch_predict] 입력 노드 %d와 샘플 %d 이므로 맞지 않습니다.",
			(int)_input.size(), (int)x.size());
	else if (NN_Base::errorFlag)
		throw StringFormat("[NN::batch_predict] 이미 레이어 생성 도중 오류가 발생 되었습니다.");

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
	if (state != BUILT_MODEL) throw StringFormat("[NN::SaveWeight] 모델이 빌드 되지 않았습니다.");
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

	printf("[NN::SaveWeight] %s 파일을 성공적으로 저장 하였습니다.\n", path.c_str());
}

void NN::LoadWeight(string path) {
	if (state != SETED_IO) throw StringFormat("[NN::LoadWeight] 모델이 빌드 되지 않았습니다.");
	cv::FileStorage fs(path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		throw StringFormat("[NN::LoadWeight] %s 파일을 열지 못했습니다.", path.c_str());
	}

	for (NN_Forward* p : _forward) p->LoadWeight(fs);
	fs.release();

	printf("[NN::LoadWeight] %s 파일을 성공적으로 로드 하였습니다.\n", path.c_str());
}

void NN::PrintForward() {
	for (NN_Forward* p : _forward) p->PrintLayerInfo();
	for (NN_Forward* p : _loss) p->PrintLayerInfo();
}

void NN::PrintBackward() {
	for (NN_Backward* p : _backward) cout << p->layerName << endl;
}