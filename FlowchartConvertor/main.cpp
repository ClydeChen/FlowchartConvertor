//////////////////////////////////////////////////////////////////////////



#include "FlowchartConvertor.h"

int main()
{
	cv::Mat img = cv::imread("testimgs\\t1.jpg");

	flowchart::FlowchartConvertor convertor;
	convertor.ProcessImage(img);

	return 0;
}