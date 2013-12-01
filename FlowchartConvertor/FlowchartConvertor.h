#pragma once
//////////////////////////////////////////////////////////////////////////
// Processor to detect flowchart shape and connections
// Jie Feng
//////////////////////////////////////////////////////////////////////////


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
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

	enum ShapeFormatType
	{
		SF_CONTOUR,
		SF_SEGMENT
	};

	struct BasicShape
	{
		BasicShapeType type;
		ShapeFormatType format_type;
		Contour original_contour;
		Contour approx_contour;
		int area;
		int perimeter;
		cv::Rect bbox;
		cv::Mat mask;
		cv::RotatedRect minRect;
		bool isConvex;

		BasicShape()
		{
			type = SHAPE_UNKNOWN;
			format_type = SF_CONTOUR;
		}
	};


	typedef std::vector<BasicShape> ShapeCollection;

	class FlowchartConvertor
	{
	private:

		float eps;
		int min_shape_area;
		cv::RNG rng_gen;	// random number generator
		cv::Size newSize;

		float PointDist(cv::Point2f& pt1, cv::Point2f& pt2)
		{
			return sqrt( (pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y) );
		}

		Contour NormalizeContour(Contour& a, const cv::Point& center_pts);

		// floodfill from a start point
		bool FloodFillMask(const cv::Mat& gray_img, cv::Point& seed, float loDiff, float upDiff, cv::Mat& mask);

	public:
		FlowchartConvertor(void);

		//////////////////////////////////////////////////////////////////////////

		// preprocessing: from original image to clear edge image
		bool PreprocessImg(const cv::Mat& img_in, cv::Mat& img_out);


		//////////////////////////////////////////////////////////////////////////

		// extract shape feature vector for recognition
		bool ComputeShapeFeature(const Contour& a, cv::Mat& feat);

		// extract shape contour from preprocessed image
		ShapeCollection DetectShapes(const cv::Mat& gray_img, int contour_mode, bool draw = false);

		// 
		BasicShapeType RecognizeShape(const BasicShape& query_shape);

		// total pipeline
		bool ProcessImage(const cv::Mat& img_in);
		
		//////////////////////////////////////////////////////////////////////////

		// visualize
		void DisplayContours(const Contours& a, int canvas_width, int canvas_height);

	};
}


