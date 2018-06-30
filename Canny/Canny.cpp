#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;

// 全局变量

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
* @函数 CannyThreshold
* @简介： trackbar 交互回调 - Canny阈值输入比例1:3
*/
void CannyThreshold(int, void*)
{
	// 使用 3x3内核降噪
	blur(src_gray, detected_edges, Size(3, 3));

	// 运行Canny算子
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	// 使用 Canny算子输出边缘作为掩码显示原图像
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}

int main()
{
	// 装载图像
	src = imread("house.jpg");

	if (!src.data)
	{
		return -1;
	}

	// 创建与src同类型和大小的矩阵(dst)
	dst.create(src.size(), src.type());

	// 原图像转换为灰度图像
	cvtColor(src, src_gray, CV_BGR2GRAY);

	// 创建显示窗口
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	// 创建trackbar
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	// 显示图像
	CannyThreshold(0, 0);

	// 等待用户反应
	waitKey(0);

	return 0;
}