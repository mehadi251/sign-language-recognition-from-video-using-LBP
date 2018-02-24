
#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::motempl;

/*static void help(void)
{
printf(
"\nThis program demonstrated the use of motion templates -- basically using the gradients\n"
"of thresholded layers of decaying frame differencing. New movements are stamped on top with floating system\n"
"time code and motions too old are thresholded away. This is the 'motion history file'. The program reads from the camera of your choice or from\n"
"a file. Gradients of motion history are used to detect direction of motion etc\n"
"Usage :\n"
"./motempl [camera number 0-n or file name, default is camera 0]\n"
);
}*/
// various tracking parameters (in seconds)
double MHI_DURATION = 0.0;
const double MAX_TIME_DELTA = .5;
const double MIN_TIME_DELTA = .05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)

// ring image buffer
vector<Mat> buf;
int last = 0;
double it = 1;
// temporary images
Mat mhi, orient, mask, segmask, zplane;
vector<Rect> regions;

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold, double c)
{

	double colori = (it / c);
	//colori = it*colori;
	double timestamp = (colori*255.0);
	double MHI_DURATION = (colori * 120);
	//double timestamp = it;
	it = it + 1;
	cout << timestamp << endl;
	Size size = img.size();
	int i, idx1 = last;
	Rect comp_rect;
	double count;
	double angle;
	Point center;
	double magnitude;
	Scalar color;

	// allocate images at the beginning or
	// reallocate them if the frame size is changed
	if (mhi.size() != size)
	{
		mhi = Mat::zeros(size, CV_32F);
		zplane = Mat::zeros(size, CV_8U);

		buf[0] = Mat::zeros(size, CV_8U);
		buf[1] = Mat::zeros(size, CV_8U);
	}

	cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale

	int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
	last = idx2;

	Mat silh = buf[idx2];
	absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

	threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
															 //threshold()
	updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI

															 // convert MHI to blue 8u image
	mhi.convertTo(mask, CV_8UC1, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);
	mhi.convertTo(dst, CV_8UC1);

	//dst = Mat::zeros(mask.size(), CV_8UC3);
	//insertChannel(mask, dst, 0);
	//Mat planes[] = { mask,mask,mask };
	//merge(planes, 3, dst);

	// calculate motion gradient orientation and valid orientation mask
	calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. It is not used further
	regions.clear();
	segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);

	// iterate through the motion components,
	// One more iteration (i == -1) corresponds to the whole image (global motion)
	for (i = -1; i < (int)regions.size(); i++) {

		if (i < 0) { // case of the whole image

			comp_rect = Rect(0, 0, size.width, size.height);
			color = Scalar(timestamp, timestamp, timestamp);
			magnitude = 100;
		}
		else { // i-th motion component
			comp_rect = regions[i];
			if (comp_rect.width + comp_rect.height < 200) // reject very small components
														  //continue;
				color = Scalar(timestamp, timestamp, timestamp);
			magnitude = 30;
		}

		// select component ROI
		Mat silh_roi = silh(comp_rect);
		Mat mhi_roi = mhi(comp_rect);
		Mat orient_roi = orient(comp_rect);
		Mat mask_roi = mask(comp_rect);

		// calculate orientation
		angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
		angle = 360.0 - angle;  // adjust for images with top-left origin

		count = norm(silh_roi, NORM_L1);; // calculate number of points within silhouette ROI

										  // check for the case of little motion
										  //if (count < comp_rect.width*comp_rect.height * 0.10)
										  //continue;

										  // draw a clock with arrow indicating the direction
		center = Point((comp_rect.x + comp_rect.width / 2),
			(comp_rect.y + comp_rect.height / 2));
		//circle()
		//circle(dst, center, cvRound(magnitude*1.2), color, 3, 16, 0);
		//line(dst, center, Point(cvRound(center.x + magnitude*cos(angle*CV_PI / 180)),
		//cvRound(center.y - magnitude*sin(angle*CV_PI / 180))), color, 3, 16, 0);
	}
}


int main(int argc, char** argv)
{
	VideoCapture cap1("accident1_dana.mp4");// lana liz naomi tyler
	VideoCapture cap("accident1_dana.mp4");
	Size SizeOfFrame = cv::Size(320, 240);

	// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
	// and use your predefined Size for siplicity 

	//VideoWriter video("Result5.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, true);
	buf.resize(2);
	int cntr = 0;
	double c = 0;
	Mat image, image1, motion, rotate_motion;
	for (;;)
	{


		cap1 >> image1;
		if (image1.empty())
			break;
		else {
			c++;
			cout << c << endl;
			//waitKey(10);
		}
	}
	for (;;)
	{

		cap >> image;
		if (image.empty())
			break;
		waitKey(10);

		//update_mhi(image, motion, 30, c);
		//imshow("Image", image);
		////waitKey(10);
		//normalize(motion, motion, 0.0, 255.0, NORM_MINMAX, CV_8UC1);
		//imshow("Motion", motion);
		//imshow("segment", segmask);
		imshow("mask", image);

		//video.write(motion);

		//flip(motion, rotate_motion, 1);
		// ...
		//imshow("rotate", rotate_motion);
		// saving part:
		std::string savingName = "accident1_dana" + std::to_string(++cntr) + ".jpg";
		cv::imwrite(savingName, image);
		waitKey(1);
		//imshow("")
		//if (waitKey(100) >= 0)
		//break;
	}
	//imwrite("comeon_lana.bmp", motion);
	//imwrite("comeon_lana_rotate.bmp", rotate_motion);

	return 0;
}