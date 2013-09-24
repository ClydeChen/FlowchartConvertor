#include "FlowchartConvertor.h"


namespace flowchart
{
	FlowchartConvertor::FlowchartConvertor(void)
	{
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
		cv::waitKey(0);

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
		cv::Canny(gray_img, edgemap, 20, 100);
		cv::imshow("canny", edgemap);
		cv::waitKey(0);

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
			//res_shapes.push_back(cur_shape);
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

				drawContours(contourimg, curves, i, CV_RGB(0, 255, 0));
				//imshow("contours", contourimg);
				//cout<<i<<endl;
				//waitKey(0);
			}

			imshow("contours", contourimg);
			imshow("edgemap", edgemap);
			cv::waitKey(0);
		}

		return res_shapes;

	}

	bool FlowchartConvertor::ProcessImage(const cv::Mat& img_in)
	{
		cv::Mat pre_img;
		PreprocessImg(img_in, pre_img);

		ShapeCollection shapes = DetectShapes(pre_img, CV_RETR_LIST, true);

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


