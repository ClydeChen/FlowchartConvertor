#pragma once
//////////////////////////////////////////////////////////////////////////
// Processor to detect flowchart shape and connections
// Jie Feng
//////////////////////////////////////////////////////////////////////////


#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>

namespace flowchart
{

	typedef std::vector<cv::Point> Contour;
	typedef std::vector<Contour> Contours;

	enum BasicShapeType
	{
		SHAPE_UNKNOWN, 
		SHAPE_SQUARE, 
		SHAPE_CIRCLE, 
		SHAPE_RECTANGLE, 
		SHAPE_TRIGANGLE, 
		SHAPE_PARALLELOGRAM
	};

	struct BasicShape
	{
		BasicShapeType type;
		Contour original_contour;
		Contour approx_contour;
		int area;
		int perimeter;
		cv::Rect bbox;
		cv::RotatedRect minRect;
		bool isConvex;
	};


	typedef std::vector<BasicShape> ShapeCollection;

	class FlowchartConvertor
	{
	private:

		float eps;
		int min_shape_area;

		float PointDist(cv::Point2f& pt1, cv::Point2f& pt2)
		{
			return sqrt( (pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y) );
		}

		Contour NormalizeContour(Contour& a, const cv::Point& center_pts);

	public:
		FlowchartConvertor(void);

		//////////////////////////////////////////////////////////////////////////

		// preprocessing
		bool PreprocessImg(const cv::Mat& img_in, cv::Mat& img_out);


		//////////////////////////////////////////////////////////////////////////

		// extract shape feature vector for recognition
		bool ComputeShapeFeature(const Contour& a, cv::Mat& feat);

		// extract shape contour from image
		ShapeCollection DetectShapes(const cv::Mat& img, int contour_mode, bool draw = false);

		// 
		BasicShapeType RecognizeShape(const BasicShape& query_shape);
		
		//////////////////////////////////////////////////////////////////////////

		// visualize
		void DisplayContours(const Contours& a, int canvas_width, int canvas_height);

	};
}


