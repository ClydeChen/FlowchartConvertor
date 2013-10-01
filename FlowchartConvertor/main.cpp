//////////////////////////////////////////////////////////////////////////



#include "FlowchartConvertor.h"

int main()
{
	cv::Mat img = cv::imread("testimgs\\red photo.jpg");

	flowchart::FlowchartConvertor convertor;
	convertor.ProcessImage(img);

	return 0;
}