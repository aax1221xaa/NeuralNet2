#include "../header/ForwardAlgorithm.h"
#include "../header/NN_Cuda.cuh"




/******************************************************

						NN_Forward

*******************************************************/

NN_Forward::NN_Forward(const string _name, const string _layerName, const int _attr) :
	NN_Base(_name, _layerName),
	attr(_attr)
{
	backModule = NULL;
}

NN_Forward::~NN_Forward() {

}

NN_Backward* NN_Forward::Create_D_ConcatModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = NULL;

	if (next.size() > 1) {
		vector<GPU_Mem<float>*> dx;
		vector<cudnnTensorDescriptor_t> dxDesc;

		for (NN_Forward* fp : next) {
			dx.push_back(&(fp->backModule->dx));
			dxDesc.push_back(fp->backModule->dxDesc);
		}
		backward = new NN_D_ConcatAdd(dx, dxDesc);

		p.push_back(backward);
	}
	else {
		backward = next[0]->backModule;
	}
	return backward;
}

void NN_Forward::CreateBackwardModule(vector<NN_Backward*>& p) {

}

void NN_Forward::SetPrevNode(NN_Forward* prevLayer) {
	prev.push_back(prevLayer);
}

void NN_Forward::SetNextNode(NN_Forward* nextLayer) {
	next.push_back(nextLayer);
}

void NN_Forward::SaveWeight(cv::FileStorage& fs) {

}

void NN_Forward::LoadWeight(cv::FileStorage& fs) {

}

void NN_Forward::SetInput(cv::Mat _x) {

}

cv::Mat NN_Forward::GetOutput() {
	cv::Mat dst;

	if (yDim.h == 1 && yDim.w == 1) {
		dst = cv::Mat(yDim.n, yDim.c, CV_32FC1);
	}
	else {
		dst = cv::Mat({ yDim.n, yDim.h, yDim.w, yDim.c }, CV_32FC1);
	}

	CHECK_CUDA(cudaMemcpy(dst.data, y.data, y.bytes, cudaMemcpyDeviceToHost));

	return dst;
}

float NN_Forward::CalcLoss(cv::Mat _y_true) {
	return 0.f;
}




/******************************************************

					  NN_ConcatAdd

*******************************************************/

NN_ConcatAdd::NN_ConcatAdd(string _name, vector<NN_Forward*> prevLayer) :
	NN_Forward(_name, "NN_ConcatAdd", CONCAT)
{
	try {
		prev = prevLayer;
		for (NN_Forward* p : prevLayer) p->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		yDim = prev[0]->yDim;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_ConcatAdd::~NN_ConcatAdd() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_ConcatAdd::Run() {
	float alpha = 1.f;
	float beta = 1.f;

	CHECK_CUDA(cudaMemset(y.data, 0, y.bytes));
	for (NN_Forward* p : prev) {
		CHECK_CUDNN(cudnnAddTensor(
			NN_Base::handle,
			&alpha,
			p->yDesc,
			p->y.data,
			&beta,
			yDesc,
			y.data
		));
	}
}

void NN_ConcatAdd::CalcSize() {
	yDim = prev[0]->yDim;
	SetDesc(yDesc, yDim);
	ReAlloc(y, yDim);
}

void NN_ConcatAdd::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Disperse(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_ConcatAdd::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}




/******************************************************

						NN_Input

*******************************************************/

NN_Input::NN_Input(Dim dim, string _name) : 
NN_Forward(_name, "NN_Input", NN_Base::INPUT)
{
	try {
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		yDim = dim;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Input::~NN_Input() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		Release(y);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Input::Run() {
	CHECK_CUDA(cudaMemcpy(y.data, x.data, y.bytes, cudaMemcpyHostToDevice));
}

void NN_Input::SetInput(cv::Mat _x) {
	/*
	[10, 10, 10, 3]
	[-1, 10, 10, 3]
	[-1, 10, 10, 1]
	[-1, 10,  1, 1]
	[-1, -1, -1, 3]

	[10, 10, 10, 3]
	[10, 10, 10, 1]
	[10, 10, 1]
*/
	if (_x.depth() != CV_32F) throw StringFormat("[NN_Input] 입력 데이터가 CV_32F가 아닙니다.");

	x = _x;

	Dim srcSize(_x);

	if (srcSize != yDim) {
		ReAlloc(y, srcSize);
		SetDesc(yDesc, srcSize);
		yDim = srcSize;
		NN_Base::changeInput = true;
	}
}

void NN_Input::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}



/******************************************************

					  NN_Random

*******************************************************/

NN_Random::~NN_Random() {

}

cv::Mat NN_Random::InitWeight(Dim wSize) {
	int dims[4];

	dims[0] = wSize.n;
	dims[1] = wSize.h;
	dims[2] = wSize.w;
	dims[3] = wSize.c;

	return cv::Mat::zeros(4, dims, CV_32FC1);
}



/******************************************************

					NN_LeCunInit

*******************************************************/

cv::Mat NN_LeCunInit::InitWeight(Dim wSize) {
	int dims[4];

	dims[0] = wSize.n;
	dims[1] = wSize.h;
	dims[2] = wSize.w;
	dims[3] = wSize.c;

	cv::Mat weight(4, dims, CV_32FC1);
	cv::RNG rng(cvGetTickCount());

	rng.fill(weight, cv::RNG::UNIFORM, cv::Scalar::all(-sqrtf(1.f / dims[3])), cv::Scalar::all(sqrt(1.f / dims[3])));

	return weight;
}



/******************************************************

					NN_XavierInit

*******************************************************/

cv::Mat NN_XavierInit::InitWeight(Dim wSize) {
	int dims[4];

	dims[0] = wSize.n;
	dims[1] = wSize.h;
	dims[2] = wSize.w;
	dims[3] = wSize.c;

	cv::Mat weight(4, dims, CV_32FC1);
	cv::RNG rng(cvGetTickCount());

	rng.fill(weight, cv::RNG::UNIFORM, cv::Scalar::all(-sqrtf(1.f / dims[3] + dims[0])), cv::Scalar::all(sqrt(1.f / dims[3] + dims[0])));

	return weight;
}



/******************************************************

						NN_HeInit

*******************************************************/

cv::Mat NN_HeInit::InitWeight(Dim wSize) {
	int dims[4];

	dims[0] = wSize.n;
	dims[1] = wSize.h;
	dims[2] = wSize.w;
	dims[3] = wSize.c;

	cv::Mat weight(4, dims, CV_32FC1);
	cv::RNG rng(cvGetTickCount());

	rng.fill(weight, cv::RNG::UNIFORM, cv::Scalar::all(-sqrtf(6.f / dims[3])), cv::Scalar::all(sqrt(6.f / dims[3])));

	return weight;
}




/******************************************************

					  NN_Dense

*******************************************************/

NN_Dense::NN_Dense(int amount, NN_Random* random, bool _addBias, string _name, NN_Forward* prevLayer) : 
	NN_Forward(_name, "NN_Dense", NN_Base::HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim)
{
	try {
		if (xDim.c == NONE || xDim.h != 1 || xDim.w != 1) {
			throw StringFormat("[NN_Dense] 입력[%d %d %d %d]로 가중치를 생성 할 수 없습니다.",
				xDim.n, xDim.h, xDim.w, xDim.c);
		}

		addBias = _addBias;

		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
		CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

		CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
			convDesc,
			0, 0,
			1, 1,
			1, 1,
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT
		));

		yDim.n = xDim.n;
		yDim.h = 1;
		yDim.w = 1;
		yDim.c = amount;

		wDim.SetDim(amount, 1, 1, xDim.c);
		ReAlloc(w, wDim);
		cv::Mat _weight = random->InitWeight(wDim);
		delete random;

		CHECK_CUDA(cudaMemcpy(w.data, _weight.data, w.bytes, cudaMemcpyHostToDevice));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(
			wDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NHWC,
			amount,
			xDim.c,
			1,
			1
		));

		if (addBias) {
			ReAlloc(b, amount);
			CHECK_CUDA(cudaMemset(b.data, 0, b.bytes));
			
			CHECK_CUDNN(cudnnCreateTensorDescriptor(&bDesc));
			SetDesc(bDesc, 1, 1, 1, amount);		
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Dense::~NN_Dense() {
	try {
		Release(y);
		Release(w);
		Release(b);
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(bDesc));
		CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Dense::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	size_t workSize = 0;
	void *workSpace = NULL;

	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		NN_Base::handle,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		algorithm,
		&workSize
	));

	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDNN(cudnnConvolutionForward(
		NN_Base::handle,
		&alpha,
		xDesc,
		x.data,
		wDesc,
		w.data,
		convDesc,
		algorithm,
		workSpace,
		workSize,
		&beta,
		yDesc,
		y.data
	));
	
	if (addBias) {
		beta = 1.f;
		CHECK_CUDNN(cudnnAddTensor(
			NN_Base::handle,
			&alpha,
			bDesc,
			b.data,
			&beta,
			yDesc,
			y.data
		));
	}
	CHECK_CUDA(cudaFree(workSpace));
}

void NN_Dense::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Dense<NN_Dense>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Dense::CalcSize() {
	if (xDim.c != wDim.c || xDim.h != 1 || xDim.w != 1) {
		throw StringFormat("[NN_Dense] 입력[%d %d %d %d] 크기가 맞지 않습니다.",
			xDim.n, xDim.h, xDim.w, xDim.c);	
	}

	yDim.n = xDim.n;
	yDim.h = 1;
	yDim.w = 1;
	yDim.c = wDim.n;

	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);

	CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
		NN_Base::handle,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algorithm));
}

void NN_Dense::SaveWeight(cv::FileStorage& fs) {
	int dims[4] = { wDim.n, wDim.h, wDim.w, wDim.c };
	cv::Mat weight(4, dims, CV_32FC1);
	
	CHECK_CUDA(cudaMemcpy(weight.data, w.data, w.bytes, cudaMemcpyDeviceToHost));
	
	fs << name;
	fs << "{";
	fs << "attr" << "NN_Dense";
	fs << "weight" << weight;
	fs << "doBias" << addBias;
	if (addBias) {
		cv::Mat bias(wDim.n, 1, CV_32FC1);
		CHECK_CUDA(cudaMemcpy(bias.data, b.data, b.bytes, cudaMemcpyDeviceToHost));
		fs << "bias" << bias;
	}
	fs << "}";
}

void NN_Dense::LoadWeight(cv::FileStorage& fs) {
	cv::FileNode node = fs[name];

	if (node.isNone()) {
		throw StringFormat("[NN_Dense::LoadWeight] 파일에 %s노드가 없습니다.", name.c_str());
	}
	else if (node.type() != cv::FileNode::MAP) {
		throw StringFormat("[NN_Dense::LoadWeight] %s노드에 맵핑이 안되어 있습니다.", name.c_str());
	}
	else if ((string)node["attr"] != "NN_Dense") {
		throw StringFormat("[NN_Dense::LoadWeight] 현재 파일 노드가 NN_Dense가 아닙니다.");
	}

	cv::Mat weight;

	node["weight"] >> weight;
	Dim dims(weight);

	if (dims != wDim) {
		throw StringFormat("[NN_Dense::LoadWeight] file weight[%d %d %d %d] != wDim[%d %d %d %d]",
			dims.n, dims.h, dims.w, dims.c,
			wDim.n, wDim.h, wDim.w, wDim.c);
	}
	CHECK_CUDA(cudaMemcpy(w.data, weight.data, w.bytes, cudaMemcpyHostToDevice));

	bool _addBias;

	node["doBias"] >> _addBias;

	if (_addBias != addBias) {
		throw StringFormat("[NN_Dense::LoadWeight] file addBias = %s, current addBias = %s",
			_addBias ? "true" : "false", addBias ? "true" : "false");
	}
	else if (_addBias) {
		cv::Mat bias;
		
		node["bias"] >> bias;
		if (bias.rows != wDim.n) {
			throw StringFormat("[NN_Dense::LoadWeight] file bias[%d] != bDim[%d]",
				bias.rows, wDim.n);
		}
		CHECK_CUDA(cudaMemcpy(b.data, bias.data, b.bytes, cudaMemcpyHostToDevice));
	}
}

void NN_Dense::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}



/******************************************************

					   NN_Dropout

*******************************************************/

NN_Dropout::NN_Dropout(float _dropout, string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_Dropout", HIDDEN),
	x(prevLayer->y),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		modeState = NN_Base::mode;

		yDim = xDim;
		dropout = _dropout;
		reservSize = 0;
		stateSize = 0;
		reservSpace = NULL;
		stateSpace = NULL;

		xDesc = prevLayer->yDesc;
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropDesc));
		CHECK_CUDNN(cudnnDropoutGetStatesSize(
			NN_Base::handle,
			&stateSize
		));
		CHECK_CUDA(cudaMalloc(&stateSpace, stateSize));
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Dropout::~NN_Dropout() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		CHECK_CUDNN(cudnnDestroyDropoutDescriptor(dropDesc));
		Release(y);
		CHECK_CUDA(cudaFree(stateSpace));
		CHECK_CUDA(cudaFree(reservSpace));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Dropout::Run() {
	if (modeState != NN_Base::mode) {
		if (NN_Base::mode == NN_Base::TRAIN_MODE)
			CHECK_CUDNN(cudnnSetDropoutDescriptor(
				dropDesc,
				NN_Base::handle,
				dropout,
				stateSpace,
				stateSize,
				cv::getTickCount()
			));
		else if (NN_Base::mode == NN_Base::INFERENCE_MODE)
			CHECK_CUDNN(cudnnSetDropoutDescriptor(
				dropDesc,
				NN_Base::handle,
				0.f,
				stateSpace,
				stateSize,
				0
			));
		modeState = NN_Base::mode;
	}

	CHECK_CUDNN(cudnnDropoutForward(
		NN_Base::handle,
		dropDesc,
		xDesc,
		x.data,
		yDesc,
		y.data,
		reservSpace,
		reservSize
	));
}

void NN_Dropout::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);
	
	backModule = new NN_D_Dropout<NN_Dropout>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Dropout::CalcSize() {
	yDim = xDim;

	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);

	CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(xDesc, &reservSize));
	CHECK_CUDA(cudaFree(reservSpace));
	CHECK_CUDA(cudaMalloc(&reservSpace, reservSize));
}

void NN_Dropout::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}




/******************************************************

						NN_ReLU

*******************************************************/

NN_ReLU::NN_ReLU(string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_ReLU", HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));

		yDim = xDim;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_ReLU::~NN_ReLU() {
	try {
		CHECK_CUDNN(cudnnDestroyActivationDescriptor(actDesc));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		Release(y);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_ReLU::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnActivationForward(
		NN_Base::handle,
		actDesc,
		&alpha,
		xDesc,
		x.data,
		&beta,
		yDesc,
		y.data
	));
}

void NN_ReLU::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Activation<NN_ReLU>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_ReLU::CalcSize() {
	yDim = xDim;

	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);
	
	CHECK_CUDNN(cudnnSetActivationDescriptor(
		actDesc,
		CUDNN_ACTIVATION_RELU,
		CUDNN_PROPAGATE_NAN,
		0
	));
}

void NN_ReLU::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}



/******************************************************

						NN_Sigmoid

*******************************************************/

NN_Sigmoid::NN_Sigmoid(string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_Sigmoid", HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));

		yDim = xDim;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Sigmoid::~NN_Sigmoid() {
	try {
		CHECK_CUDNN(cudnnDestroyActivationDescriptor(actDesc));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		Release(y);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Sigmoid::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnActivationForward(
		NN_Base::handle,
		actDesc,
		&alpha,
		xDesc,
		x.data,
		&beta,
		yDesc,
		y.data
	));
}

void NN_Sigmoid::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Activation<NN_Sigmoid>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Sigmoid::CalcSize() {
	yDim = xDim;

	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);

	CHECK_CUDNN(cudnnSetActivationDescriptor(
		actDesc,
		CUDNN_ACTIVATION_SIGMOID,
		CUDNN_PROPAGATE_NAN,
		0
	));
}

void NN_Sigmoid::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}




/******************************************************

					  NN_Softmax

*******************************************************/

NN_Softmax::NN_Softmax(string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_Softmax", OUTPUT),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		yDim = xDim;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Softmax::~NN_Softmax() {
	try {
		Release(y);
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Softmax::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* nextBack = next[0]->backModule;

	if (next[0]->layerName == "NN_CrossEntropy") backModule = nextBack;
	else {
		backModule = new NN_D_Softmax<NN_Softmax>(*this, nextBack->dx, nextBack->dxDesc);
		p.push_back(backModule);
	}
}

void NN_Softmax::Run() {
	float alpha = 1.f;
	float beta = 0.f;

	CHECK_CUDNN(cudnnSoftmaxForward(
		NN_Base::handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		xDesc,
		x.data,
		&beta,
		yDesc,
		y.data
	));
}

void NN_Softmax::CalcSize() {
	yDim = xDim;

	SetDesc(yDesc, yDim);
	ReAlloc(y, yDim);
}

cv::Mat NN_Softmax::GetOutput() {
	cv::Mat _y;

	if (yDim.h == 1 && yDim.w == 1) {
		_y = cv::Mat::zeros(yDim.n, yDim.c, CV_32FC1);
	}
	else {
		int dims[4] = { yDim.n, yDim.h, yDim.w, yDim.c };
		_y = cv::Mat::zeros(4, dims, CV_32FC1);
	}

	CHECK_CUDA(cudaMemcpy(_y.data, y.data, y.bytes, cudaMemcpyDeviceToHost));

	return _y;
}

void NN_Softmax::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}




/******************************************************

					  NN_CrossEntropy

*******************************************************/

NN_CrossEntropy::NN_CrossEntropy(string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_CrossEntropy", LOSS),
	x(prevLayer->y),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		if (prevLayer->layerName == "NN_Softmax") {
			LossFunc = LikelieHoodCE;
		}
		else if (prevLayer->layerName == "NN_Sigmoid") {
			LossFunc = BinaryCE;
		}
		else {
			throw StringFormat("[NN_CrossEntropy] 이전 레이어가 %s입니다.",
				prevLayer->layerName.c_str());
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_CrossEntropy::~NN_CrossEntropy() {
	try {
		Release(y_true);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

float NN_CrossEntropy::CalcLoss(cv::Mat _y_true) {
	float loss = 0.f;
	float* lossArr = NULL;

	Dim tSize;
	tSize.SetDim(_y_true);

	if (tSize != xDim) {
		throw StringFormat("[NN_CrossEntropy::CalcLoss] y_pred[%d %d %d %d] != y_true[%d %d %d %d]",
			xDim.n, xDim.h, xDim.w, xDim.c,
			tSize.n, tSize.h, tSize.w, tSize.c);
	}

	CHECK_CUDA(cudaMalloc(&lossArr, sizeof(float) * xDim.n * xDim.h * xDim.w));
	CHECK_CUDA(cudaMemcpy(y_true.data, _y_true.data, y_true.bytes, cudaMemcpyHostToDevice));
	CHECK_CUDA((*LossFunc)(x.data, y_true.data, lossArr, xDim.n, xDim.h, xDim.w, xDim.c));
	CHECK_CUDA(ReduceSum(lossArr, xDim.n * xDim.h * xDim.w));
	CHECK_CUDA(cudaMemcpy(&loss, lossArr, sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(lossArr));

	return loss / xDim.n;
}

void NN_CrossEntropy::CreateBackwardModule(vector<NN_Backward*>& p) {
	backModule = new NN_D_CrossEntropy<NN_CrossEntropy>(*this);
	p.push_back(backModule);
}

void NN_CrossEntropy::CalcSize() {
	ReAlloc(y_true, xDim);
}

void NN_CrossEntropy::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		xDim.n, xDim.h, xDim.w, xDim.c);
}




/******************************************************

					  NN_Flatten

*******************************************************/

NN_Flatten::NN_Flatten(string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_Flatten", HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		if (xDim.h < 1 || xDim.w < 1 || xDim.c < 1) {
			throw StringFormat("[NN_Flatten] 입력 사이즈[%d %d %d %d]를 Flat할수 없습니다.",
				xDim.n, xDim.h, xDim.w, xDim.c);
		}
		yDim.n = xDim.n;
		yDim.h = yDim.w = 1;
		yDim.c = xDim.h * xDim.w * xDim.c;
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Flatten::~NN_Flatten() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Flatten::Run() {
	y = x;
}

void NN_Flatten::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Flatten<NN_Flatten>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Flatten::CalcSize() {
	yDim.n = xDim.n;
	yDim.h = yDim.w = 1;
	yDim.c = xDim.h * xDim.w * xDim.c;

	SetDesc(yDesc, yDim);
}

void NN_Flatten::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}



/******************************************************

					  NN_Convolution

*******************************************************/

NN_Convolution::NN_Convolution(
	int amount,
	vector<int> kSize,
	vector<int> stride,
	vector<int> dilation,
	NN_Random* random,
	int pad,
	bool _addBias,
	string _name,
	NN_Forward* prevLayer
) :
	NN_Forward(_name, "NN_Convolution", HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim) 
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		addBias = _addBias;

		if (xDim.c < 1) {
			throw StringFormat("[NN_Convolution] 입력 사이즈[%d %d %d %d]가 잘 못 되었습니다.",
				xDim.n, xDim.h, xDim.w, xDim.c);
		}

		this->stride = stride;
		this->dilation = dilation;
		this->pad = pad;

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
		CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		
		wDim.n = amount;
		wDim.h = kSize[0];
		wDim.w = kSize[1];
		wDim.c = xDim.c;
		
		ReAlloc(w, wDim);
		cv::Mat weight = random->InitWeight(wDim);
		delete random;

		CHECK_CUDA(cudaMemcpy(w.data, weight.data, w.bytes, cudaMemcpyHostToDevice));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(
			wDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NHWC,
			wDim.n,
			wDim.c,
			wDim.h,
			wDim.w
		));

		yDim.n = xDim.n;
		yDim.c = amount;

		if (xDim.h == NONE || yDim.w == NONE) {	
			yDim.h = xDim.h;
			yDim.w = xDim.w;	
		}
		else {
			if (pad == NN_Base::SAME) {
				yDim.h = xDim.h;
				yDim.w = xDim.w;

				int ph = (stride[0] * (xDim.h - 1) + (wDim.h + (wDim.h - 1) * (dilation[0] - 1)) - stride[0]) / 2;
				int pw = (stride[1] * (xDim.w - 1) + (wDim.w + (wDim.w - 1) * (dilation[1] - 1)) - stride[1]) / 2;

				CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
					convDesc,
					ph,
					pw,
					stride[0],
					stride[1],
					dilation[0],
					dilation[1],
					CUDNN_CROSS_CORRELATION,
					CUDNN_DATA_FLOAT
				));
			}
			else {
				yDim.h = (xDim.h - (wDim.h + (wDim.h - 1) * (dilation[0] - 1))) / stride[0] + 1;
				yDim.w = (xDim.w - (wDim.w + (wDim.w - 1) * (dilation[1] - 1))) / stride[1] + 1;

				CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
					convDesc,
					0,
					0,
					stride[0],
					stride[1],
					dilation[0],
					dilation[1],
					CUDNN_CROSS_CORRELATION,
					CUDNN_DATA_FLOAT
				));
			}
		}

		if (addBias) {
			ReAlloc(b, amount);
			CHECK_CUDA(cudaMemset(b.data, 0, b.bytes));
			CHECK_CUDNN(cudnnCreateTensorDescriptor(&bDesc));
			SetDesc(bDesc, 1, 1, 1, amount);
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Convolution::~NN_Convolution() {
	try {
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
		CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		Release(y);
		Release(w);
		if (addBias) {
			CHECK_CUDNN(cudnnDestroyTensorDescriptor(bDesc));
			Release(b);
		}
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Convolution::Run() {
	float alpha = 1.f;
	float beta = 0.f;
	size_t workSize = 0;
	void* workSpace = NULL;

	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		NN_Base::handle,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		algorithm,
		&workSize));
	CHECK_CUDA(cudaMalloc(&workSpace, workSize));
	CHECK_CUDNN(cudnnConvolutionForward(
		NN_Base::handle,
		&alpha,
		xDesc,
		x.data,
		wDesc,
		w.data,
		convDesc,
		algorithm,
		workSpace,
		workSize,
		&beta,
		yDesc,
		y.data));
	CHECK_CUDA(cudaFree(workSpace));

	if (addBias) {
		beta = 1.f;
		CHECK_CUDNN(cudnnAddTensor(
			NN_Base::handle,
			&alpha,
			bDesc,
			b.data,
			&beta,
			yDesc,
			y.data
		));
	}
}

void NN_Convolution::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Convolution<NN_Convolution>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Convolution::CalcSize() {
	if (pad == NN_Base::SAME) {
		int ph = (stride[0] * (xDim.h - 1) + (wDim.h + (wDim.h - 1) * (dilation[0] - 1)) - stride[0]) / 2;
		int pw = (stride[1] * (xDim.w - 1) + (wDim.w + (wDim.w - 1) * (dilation[1] - 1)) - stride[1]) / 2;

		CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
			convDesc,
			ph,
			pw,
			stride[0],
			stride[1],
			dilation[0],
			dilation[1],
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT
		));
	}
	CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
		convDesc,
		xDesc,
		wDesc,
		&yDim.n,
		&yDim.c,
		&yDim.h,
		&yDim.w
	));
	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);
	
	CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
		NN_Base::handle,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algorithm
	));
}

void NN_Convolution::SaveWeight(cv::FileStorage& fs) {
	int dims[4] = { wDim.n, wDim.h, wDim.w, wDim.c };
	cv::Mat weight(4, dims, CV_32FC1);

	CHECK_CUDA(cudaMemcpy(weight.data, w.data, w.bytes, cudaMemcpyDeviceToHost));

	fs << name;
	fs << "{";
	fs << "attr" << "NN_Convolution";
	fs << "weight" << weight;
	fs << "doBias" << addBias;
	if (addBias) {
		cv::Mat bias(wDim.n, 1, CV_32FC1);
		CHECK_CUDA(cudaMemcpy(bias.data, b.data, b.bytes, cudaMemcpyDeviceToHost));
		fs << "bias" << bias;
	}
	fs << "}";
}

void NN_Convolution::LoadWeight(cv::FileStorage& fs) {
	cv::FileNode node = fs[name];

	if (node.isNone()) {
		throw StringFormat("[NN_Convolution::LoadWeight] 파일에 %s노드가 없습니다.", name.c_str());
	}
	else if (node.type() != cv::FileNode::MAP) {
		throw StringFormat("[NN_Convolution::LoadWeight] %s노드에 맵핑이 안되어 있습니다.", name.c_str());
	}
	else if ((string)node["attr"] != "NN_Convolution") {
		throw StringFormat("[NN_Convolution::LoadWeight] 현재 파일 노드가 NN_Convolution가 아닙니다.");
	}

	cv::Mat weight;

	node["weight"] >> weight;
	Dim dims(weight);

	if (dims != wDim) {
		throw StringFormat("[NN_Convolution::LoadWeight] file weight[%d %d %d %d] != wDim[%d %d %d %d]",
			dims.n, dims.h, dims.w, dims.c,
			wDim.n, wDim.h, wDim.w, wDim.c);
	}
	CHECK_CUDA(cudaMemcpy(w.data, weight.data, w.bytes, cudaMemcpyHostToDevice));

	bool _addBias;

	node["doBias"] >> _addBias;

	if (_addBias != addBias) {
		throw StringFormat("[NN_Convolution::LoadWeight] file addBias = %s, current addBias = %s",
			_addBias ? "true" : "false", addBias ? "true" : "false");
	}
	else if (_addBias) {
		cv::Mat bias;

		node["bias"] >> bias;
		if (bias.rows != wDim.n) {
			throw StringFormat("[NN_Convolution::LoadWeight] file bias[%d] != bDim[%d]",
				bias.rows, wDim.n);
		}
		CHECK_CUDA(cudaMemcpy(b.data, bias.data, b.bytes, cudaMemcpyHostToDevice));
	}
}

void NN_Convolution::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}




/******************************************************

					  NN_Maxpool

*******************************************************/

NN_Maxpool::NN_Maxpool(vector<int> kSize, vector<int> stride, const int _pad, string _name, NN_Forward* prevLayer) :
	NN_Forward(_name, "NN_Maxpool", HIDDEN),
	x(prevLayer->y),
	xDesc(prevLayer->yDesc),
	xDim(prevLayer->yDim),
	pad(_pad)
{
	try {
		SetPrevNode(prevLayer);
		prevLayer->SetNextNode(this);

		CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
		CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

		this->kSize = kSize;
		this->stride = stride;

		yDim.n = xDim.n;
		yDim.c = xDim.c;

		if (xDim.h == NONE || xDim.w == NONE) {
			yDim.h = xDim.h;
			yDim.w = xDim.w;
		}
		else {
			if (pad == NN_Base::SAME) {
				yDim.h = cvCeil((float)(xDim.h - kSize[0]) / stride[0] + 1);
				yDim.w = cvCeil((float)(xDim.w - kSize[1]) / stride[1] + 1);
			}
			else {
				yDim.h = (xDim.h - kSize[0]) / stride[0] + 1;
				yDim.w = (xDim.w - kSize[1]) / stride[1] + 1;

				CHECK_CUDNN(cudnnSetPooling2dDescriptor(
					poolDesc,
					CUDNN_POOLING_MAX,
					CUDNN_PROPAGATE_NAN,
					kSize[0],
					kSize[1],
					0,
					0,
					stride[0],
					stride[1]));
			}
		}
	}
	catch (const string& e) {
		NN_Base::errorFlag = true;
		cout << e << endl;
	}
}

NN_Maxpool::~NN_Maxpool() {
	try {
		CHECK_CUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
		Release(y);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

void NN_Maxpool::Run() {
	float alpha = 1.f;
	float beta = 0.f;
	
	CHECK_CUDNN(cudnnPoolingForward(
		NN_Base::handle,
		poolDesc,
		&alpha,
		xDesc,
		x.data,
		&beta,
		yDesc,
		y.data
	));
}

void NN_Maxpool::CreateBackwardModule(vector<NN_Backward*>& p) {
	NN_Backward* backward = Create_D_ConcatModule(p);

	backModule = new NN_D_Maxpool<NN_Maxpool>(*this, backward->dx, backward->dxDesc);
	p.push_back(backModule);
}

void NN_Maxpool::CalcSize() {
	yDim.n = xDim.n;
	yDim.c = xDim.c;

	if (pad == NN_Base::SAME) {
		yDim.h = cvCeil((float)(xDim.h - kSize[0]) / stride[0] + 1);
		yDim.w = cvCeil((float)(xDim.w - kSize[1]) / stride[1] + 1);

		int ph = cvCeil((float)(stride[0] * (yDim.h - 1) - xDim.h + kSize[0]) / 2);
		int pw = cvCeil((float)(stride[1] * (yDim.w - 1) - xDim.w + kSize[1]) / 2);

		CHECK_CUDNN(cudnnSetPooling2dDescriptor(
			poolDesc,
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			kSize[0],
			kSize[1],
			ph,
			pw,
			stride[0],
			stride[1]));
	}
	else {
		yDim.h = (xDim.h - kSize[0]) / stride[0] + 1;
		yDim.w = (xDim.w - kSize[1]) / stride[1] + 1;
	}

	ReAlloc(y, yDim);
	SetDesc(yDesc, yDim);
}

void NN_Maxpool::PrintLayerInfo() {
	printf("[%s] yDim = [%d %d %d %d]\n",
		layerName.c_str(),
		yDim.n, yDim.h, yDim.w, yDim.c);
}