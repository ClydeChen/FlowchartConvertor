//////////////////////////////////////////////////////////////////////////



#include "FlowchartConvertor.h"

int main()
{
	cv::Mat img = cv::imread("testimgs\\phototest.jpg");

	flowchart::FlowchartConvertor convertor;
	convertor.PreprocessImg(img, cv::Mat());

	return 0;
}