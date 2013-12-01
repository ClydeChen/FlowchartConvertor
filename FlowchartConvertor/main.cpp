//////////////////////////////////////////////////////////////////////////



#include "FlowchartConvertor.h"

int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		std::cout<<"Incorrect format: convertor.exe image_file"<<std::endl;
		return -1;
	}
	cv::Mat img = cv::imread(argv[1]);
	if(img.empty())
	{
		std::cout<<"Can't find input image."<<std::endl;
		return -1;
	}

	flowchart::FlowchartConvertor convertor;
	convertor.ProcessImage(img);

	return 0;
}