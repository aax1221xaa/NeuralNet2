#pragma once
#include "../header/Dim.h"
#include "../header/CudaException.h"


using namespace std;






int GetMatChannels(int type);
int GetMatType(int type);
cv::Mat MeshGrid(cv::Mat x, cv::Mat y);
template <typename _T> cv::Mat GetMeshGrid(cv::Mat x, cv::Mat y);
cv::Mat aRange(int start, int end, int step = 1);
cv::Mat argMax(cv::Mat& src, int rank);
cv::Mat nrand(vector<int> dims, int low, int high);
bool StrAllCompare(vector<string> strs);