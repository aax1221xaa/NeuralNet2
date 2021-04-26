#include "../header/Dim.h"
#include "../header/Base.h"



Dim::Dim() {
	n = h = w = c = 1;
}

Dim::Dim(const cv::Mat& mat) {
	try {
		SetDim(mat);
	}
	catch (const string& e) {
		cout << e << endl;
	}
}

Dim::Dim(int _n, int _h, int _w, int _c) {
	n = _n;
	h = _h;
	w = _w;
	c = _c;
}

void Dim::SetDim(const cv::Mat& mat) {
	n = h = w = c = 1;

	if (mat.channels() == 1) {
		if (mat.dims == 2) {
			n = mat.size[0];
			c = mat.size[1];
		}
		else if (mat.dims == 3 || mat.dims == 4) {
			int* _dims[4] = { &n, &h, &w, &c };

			for (int i = 0; i < mat.dims; ++i) *_dims[i] = mat.size[i];
		}
		else {
			throw StringFormat("[Dim::SetDim] 1ä�� %d���� ġ���� ���� ���� �ʽ��ϴ�.", mat.dims);
		}
	}
	else if (mat.channels() == 3) {
		if (mat.dims == 3) {
			n = mat.size[0];
			h = mat.size[1];
			w = mat.size[2];
			c = 3;
		}
		else {
			throw StringFormat("[Dim::SetDim] 3ä�� %d���� ġ���� ���� ���� �ʽ��ϴ�.", mat.dims);
		}

	}
	else {
		throw StringFormat("[Dim::SetDim] %d���� ä�� ���� ���� ���� �ʽ��ϴ�.", mat.channels());
	}
}

void Dim::SetDim(int _n, int _h, int _w, int _c) {
	n = _n;
	h = _h;
	w = _w;
	c = _c;
}

size_t Dim::GetTotalSize() {
	return (size_t)n * h * w * c;
}

int& Dim::operator[](int idx) {
	int *arr[4] = { &n, &h, &w, &c };
	
	return *arr[idx];
}

bool Dim::operator!=(const Dim dim) {
	bool flag = false;

	if (n != dim.n || h != dim.h || w != dim.w || c != dim.c)
		flag = true;

	return flag;
}