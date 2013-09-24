#include "FlowchartConvertor.h"


namespace flowchart
{
	FlowchartConvertor::FlowchartConvertor(void)
	{
		rng_gen = cv::RNG(3456);
		min_shape_area = 200;
		eps = 0.1f;
		newSize = cv::Size(400,400);
	}

	Contour FlowchartConvertor::NormalizeContour(Contour& a, const cv::Point& center_pts)
	{
		Contour res_contour = a;
		cv::Point mean(0,0);
		for(size_t i=0; i<a.size(); i++)
		{
			mean.x += a[i].x;
			mean.y += a[i].y;
		}

		mean.x /= a.size();
		mean.y /= a.size();

		cv::Point diff_mean(center_pts.x - mean.x, center_pts.y-mean.y);


		for(size_t i=0; i<a.size(); i++)
		{
			res_contour[i].x = res_contour[i].x + diff_mean.x;
			res_contour[i].y = res_contour[i].y + diff_mean.y;
		}

		return res_contour;
	}

	//////////////////////////////////////////////////////////////////////////

	bool FlowchartConvertor::PreprocessImg(const cv::Mat& img_in, cv::Mat& img_out)
	{
		if(img_in.channels() != 3 && img_in.channels() != 1)
			return false;

		cv::Mat gray_img;
		if(img_in.channels() == 3)
			cv::cvtColor(img_in, gray_img, cv::COLOR_BGR2GRAY);
		else
			img_in.copyTo(gray_img);

		cv::resize(gray_img, gray_img, newSize);
		
		// smooth
		cv::GaussianBlur(gray_img, gray_img, cv::Size(3,3), 1);

		// compute edge magnitude
		cv::Mat grad_x, grad_y, grad_mag;
		cv::Sobel( gray_img, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
		cv::Sobel( gray_img, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );
		cv::magnitude(grad_x, grad_y, grad_mag);
		double minval, maxval;
		cv::minMaxLoc(grad_mag, &minval, &maxval);
		cv::normalize(grad_mag, grad_mag, 1, 0, cv::NORM_MINMAX);
		cv::Mat grad_mag_th;
		cv::threshold(grad_mag, grad_mag_th, 0.1, 1, cv::THRESH_TOZERO);
		grad_mag_th.convertTo(img_out, CV_8U, 255);

		cv::imshow("gray", gray_img);
		cv::imshow("mag", grad_mag);
		cv::imshow("mag_th", img_out);
		//cv::waitKey(0);

		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool FlowchartConvertor::ComputeShapeFeature(const Contour& a, cv::Mat& feat)
	{
		// compute normalized center distance values
		// compute mean
		cv::Point2f centerpt(0,0);
		for(size_t i=0; i<a.size(); i++)
		{
			centerpt.x += (float)a[i].x / a.size();
			centerpt.y += (float)a[i].y / a.size();
		}

		float maxdist = 0;
		cv::Mat center_dists(1, a.size(), CV_32FC1);
		for(size_t i=0; i<a.size(); i++)
		{
			float dist = \
				sqrt( (a[i].x - centerpt.x)*(a[i].x - centerpt.x)+(a[i].y - centerpt.y)*(a[i].y - centerpt.y) );
			center_dists.at<float>(0,i) = dist;
			if(dist > maxdist)
				maxdist = dist;
		}

		for(size_t i=0; i<center_dists.cols; i++)
			center_dists.at<float>(i) /= maxdist;

		// create histogram
		cv::Mat hist;
		int dbins = 10;
		int histSize[] = {dbins};
		// hue varies from 0 to 179, see cvtColor
		float dranges[] = { 0, 1 };
		const float* ranges[] = { dranges };
		// we compute the histogram from the 0-th and 1-st channels
		int channels[] = {0};
		calcHist(&center_dists, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

		normalize(hist, hist, 1, 0, cv::NORM_L1);

		/*for(int i=0; i<dbins; i++)
			cout<<hist.at<float>(i)<<" ";*/

		return true;
	}

	ShapeCollection FlowchartConvertor::DetectShapes(const cv::Mat& gray_img, int contour_mode, bool draw)
	{
		cv::Mat edgemap;
		cv::Canny(gray_img, edgemap, 100, 200);
		cv::imshow("canny", edgemap);
		cv::dilate(edgemap, edgemap, cv::Mat());
		cv::erode(edgemap, edgemap, cv::Mat());
		cv::waitKey(10);

		// connect broken lines
		//dilate(edgemap, edgemap, Mat(), Point(-1,-1));
		//erode(edgemap, edgemap, Mat(), Point(-1,-1));

		// detect contours and draw
		cv::Mat edge_copy;
		edgemap.copyTo(edge_copy);
		Contours curves;
		std::vector<cv::Vec4i> hierarchy;
		findContours( edge_copy, curves, hierarchy, contour_mode, CV_CHAIN_APPROX_SIMPLE );

		ShapeCollection res_shapes(curves.size());
		for(size_t i=0; i<curves.size(); i++)
		{
			BasicShape& cur_shape = res_shapes[i];
			cur_shape.type = SHAPE_UNKNOWN;
			cur_shape.original_contour = curves[i];
			approxPolyDP(cur_shape.original_contour, cur_shape.approx_contour, cv::arcLength(cv::Mat(cur_shape.original_contour), true)*0.02, true);
			cur_shape.minRect = minAreaRect( cur_shape.approx_contour );
			cur_shape.bbox = boundingRect(cur_shape.approx_contour);
			cur_shape.area = contourArea(curves[i]);
			cur_shape.perimeter = arcLength(curves[i], true);
			cur_shape.isConvex = isContourConvex(cur_shape.approx_contour);
		}

		// draw detected contours
		if(draw)
		{
			cv::Mat contourimg( gray_img.rows, gray_img.cols, CV_8UC3 );
			contourimg.setTo(255);
			srand( time(NULL) );
			for(size_t i=0; i<res_shapes.size(); i++)
			{
				if(res_shapes[i].area < min_shape_area)
					continue;	// remove small contours

				CvScalar cur_color = CV_RGB(rng_gen.uniform(0,255), rng_gen.uniform(0,255), rng_gen.uniform(0,255));
				drawContours(contourimg, curves, i, cur_color);
				imshow("contours", contourimg);
				cv::waitKey(0);
			}

			imshow("contours", contourimg);
			imshow("edgemap", edgemap);
			cv::waitKey(0);
		}

		return res_shapes;

	}

	BasicShapeType FlowchartConvertor::RecognizeShape(const BasicShape& query_shape)
	{
		BasicShapeType res_type = SHAPE_UNKNOWN;

		if(query_shape.original_contour.empty())
		{
			std::cerr<<"Empty shape."<<std::endl;
			return res_type;
		}

		// approximate contour
		Contour approx_shape = query_shape.approx_contour;
		float shapeArea = contourArea(approx_shape);
		float boxArea = query_shape.minRect.size.area();

		std::cout<<approx_shape.size()<<std::endl;

		// triangle
		if(approx_shape.size() >= 3 && approx_shape.size() <= 4)
		{
			// compute area ratio
			float areaRatio = shapeArea / boxArea;
			float diff = fabs(areaRatio-0.5);
			if( diff < eps )
				res_type = SHAPE_TRIGANGLE;
		}

		if( res_type == SHAPE_UNKNOWN && approx_shape.size() >= 4 && approx_shape.size() <= 5 )
		{
			// square or rectangle or parallelogram
			// check if 4 angles are ~90

			// use area to see if square or rectangle
			float areaRatio = shapeArea / (query_shape.minRect.size.width*query_shape.minRect.size.width);
			float diff = fabs(areaRatio-1);
			if( diff < eps )
				res_type = SHAPE_SQUARE;
			else
				res_type = SHAPE_RECTANGLE;

		}

		if( res_type == SHAPE_UNKNOWN )
		{
			float areaRatio = shapeArea / boxArea;
			float diff = fabs(areaRatio-3.14/4.0);
			if( diff < eps )
				res_type = SHAPE_CIRCLE;
		}

		return res_type;
	}

	bool FlowchartConvertor::ProcessImage(const cv::Mat& img_in)
	{
		cv::Mat pre_img;
		PreprocessImg(img_in, pre_img);
		
		ShapeCollection shapes = DetectShapes(pre_img, CV_RETR_TREE, false);
		Contours cons;

		cv::Mat res_img(img_in.rows, img_in.cols, CV_8UC3);
		res_img.setTo(255);
		for(size_t i=0; i<shapes.size(); i++)
		{
			BasicShape curshape = shapes[i];

			// compute contour area
			if(curshape.area < min_shape_area || !curshape.isConvex)
				continue;

			// add to collection
			cons.push_back(curshape.approx_contour);

			// create color
			cv::Scalar cur_color = CV_RGB(rng_gen.uniform(0,255), rng_gen.uniform(0,255), rng_gen.uniform(0,255));

			BasicShapeType type = RecognizeShape( curshape );
			std::string type_name;
			if(type == SHAPE_CIRCLE)
			{
				type_name = "Circle";
				// draw on result image
				int diag = (int)sqrt(1.0f*curshape.bbox.width*curshape.bbox.width+1.0f*curshape.bbox.height*curshape.bbox.height);
				circle( res_img, cv::Point(curshape.bbox.x+curshape.bbox.width/2, curshape.bbox.y+curshape.bbox.height/2),
					diag/2, cur_color );
			}
			if(type == SHAPE_RECTANGLE)
			{
				type_name = "Rectangle";
				drawContours( res_img, cons, cons.size()-1, cur_color );
				//rectangle( res_img, curshape.bbox, cur_color );
			}
			if(type == SHAPE_SQUARE)
			{
				type_name = "Square";
				rectangle( res_img, curshape.bbox, cur_color );
			}
			if(type == SHAPE_TRIGANGLE)
			{
				type_name = "Triangle";
				drawContours( res_img, cons, cons.size()-1, cur_color );
			}
			if(type == SHAPE_UNKNOWN)
			{
				type_name = "Unknown";
				drawContours( res_img, cons, cons.size()-1, cur_color );
			}

			// output text
			cv::putText( res_img, type_name, cv::Point(curshape.bbox.x, curshape.bbox.br().y+15), cv::FONT_HERSHEY_PLAIN, 0.8, cur_color );
		}

		cv::imshow("res", res_img);
		cv::waitKey(0);

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// visualization
	void FlowchartConvertor::DisplayContours(const Contours& a, int canvas_width, int canvas_height)
	{
		cv::Mat img(canvas_height, canvas_width, CV_8UC3);
		img.setTo(255);

		// draw contour
		for(size_t i=0; i<a.size(); i++)
			drawContours(img, a, i, CV_RGB(0, 255, 0), 1.5);
		// draw vertices
		for(size_t i=0; i<a.size(); i++)
			for(size_t j=0; j<a[i].size(); j++)
				circle(img, a[i][j], 1, CV_RGB(255,0,0));

		cv::imshow("contour", img);
		cv::waitKey(0);
	}
}


