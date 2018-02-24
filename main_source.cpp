/*#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\videoio\videoio.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <stdio.h>*/
#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>

#include "opencv2\objdetect\objdetect.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::motempl;
using namespace cv::ml;
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
float temp[944];
int a1[59], a2[59], a3[59], a4[59], a5[59], a6[59], a7[59], a8[59], a9[59], a10[59], a11[59], a12[59], a13[59], a14[59], a15[59], a16[59];

Mat LBP(Mat src_image)
{
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);

	if (src_image.channels() == 3)
		cvtColor(src_image, Image, CV_BGR2GRAY);

	unsigned center = 0;
	unsigned center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)
	{
		for (int col = 1; col < Image.cols - 1; col++)
		{
			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 8;

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 16;

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 32;

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 64;

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 128;

			//cout << center_lbp << endl;
			lbp.at<uchar>(row, col) = center_lbp;
		}
	}
	if (affiche == true)
	{
		cv::imshow("image LBP", lbp);
		waitKey(0);
		cv::imshow("grayscale", Image);
		waitKey(0);
	}

	else
	{
		//cv::destroyWindow("image LBP");
		//cv::destroyWindow("grayscale");
	}

	return lbp;
}
static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold, double c)
{

	double colori = (it / c);
	//colori = it*colori;
	double timestamp = (colori*255.0);
	double MHI_DURATION = 255;
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
		if (count < comp_rect.width*comp_rect.height * 0.10)
			continue;

		// draw a clock with arrow indicating the direction
		center = Point((comp_rect.x + comp_rect.width / 2),
			(comp_rect.y + comp_rect.height / 2));
		//circle()
		//circle(img, center, cvRound(magnitude*1.2), color, 3, 16, 0);
		//line(dst, center, Point(cvRound(center.x + magnitude*cos(angle*CV_PI / 180)),
		//cvRound(center.y - magnitude*sin(angle*CV_PI / 180))), color, 3, 16, 0);
	}
}

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main(int argc, const char** argv)
{


	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

	VideoCapture cap6;// tyler liz naomi lana jaime
	cap6.open(0);
	time_t start, end;
	time(&start);

	//if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
	while (cap6.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		// obtain input image from source
		cap6.retrieve(frame, CV_CAP_OPENNI_BGR_IMAGE);
		// Just resize input image if you want
		resize(frame, frame, Size(320, 240));
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		char c = (char)waitKey(33);
		if (c == 27) { break; } // escape
		time(&end);
		double dif = difftime(end, start);
		printf("Elasped time is %.2lf seconds.", dif);
		if (dif == 20)
		{
			std::cout << "DONE" << dif << std::endl;
			break;
		}
	}

	//-- 2. Read the video stream
	string s= "hungry_nazia";
	VideoCapture cap1(s + ".mp4");
	VideoCapture cap4( s + ".mp4");// tyler liz naomi lana jaime
	Mat im;
	int c = 0;
	for (;;)
	{


		cap4 >> im;
		if (im.empty())
			break;
		else {
			c++;
			cout << c << endl;
			//waitKey(10);
		}
	}
	//capture.open(0);
	//if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empy input img, foreground and background image and foreground mask.
	Mat img, foregroundMask, backgroundImage, foregroundImg;

	// capture video from source 0, which is web camera, If you want capture video from file just replace //by  
	VideoCapture cap(s + "_face.wmv");
	//VideoCapture cap(0);
	// This is one of the most important thing
	// Sizes
	//Your VideoWriter Size must correspond with input video.

	// Size of your output video 
	Size SizeOfFrame = cv::Size(320, 240);

	// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
	// and use your predefined Size for siplicity 

	VideoWriter video(s + "_fore.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, false);
Mat prev, next, result;
	Mat flow(SizeOfFrame, CV_32FC2);
	// main loop to grab sequence of input files
	int i = 0;
	for (; ; ) {

		bool ok = cap.grab();

		if (ok == false) {

			std::cout << "Video Capture Fail" << std::endl;
			break;


		}
		else {

			// obtain input image from source
			cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
			// Just resize input image if you want
			resize(img, img, Size(320, 240));
			Size frameSize(320, 240);

			//VideoWriter oVideoWriter("D:/MyVideo.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true); //initialize the VideoWriter object 

			/*if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
			{
			cout << "ERROR: Failed to write the video" << endl;
			return -1;
			}*/
			// create foreground mask of proper size
			if (foregroundMask.empty()) {
				foregroundMask.create(img.size(), img.type());
			}

			// compute foreground mask 8 bit image
			// -1 is parameter that chose automatically your learning rate

			bg_model->apply(img, foregroundMask, true ? -1 : 0);

			// smooth the mask to reduce noise in image
			GaussianBlur(foregroundMask, foregroundMask, Size(9, 9), 3.5, 3.5);

			// threshold mask to saturate at black and white values
			threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
			// create black foreground image
			next = foregroundMask;
			if (i != 0)
			{
				//			calcOpticalFlowFarneback(prev, next, flow, .5, 1, 3, 3, 5, 1.1, 0);
			}
			i = i + 1;
			foregroundImg = Scalar::all(0);
			// Copy source image to foreground image only in area with white mask
			img.copyTo(foregroundImg, foregroundMask);

			//Get background image
			bg_model->getBackgroundImage(backgroundImage);

			// Show the results
			//writer the frame into the file
			prev = foregroundMask;
			imshow("foreground mask", foregroundMask);
			imshow("foreground image", foregroundImg);
			result = flow.reshape(1);
			//	imshow("flow", result);

			video.write(foregroundMask);

			int key6 = waitKey(10);
			/*if (waitKey(10) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			{
			cout << "esc key is pressed by user" << endl;
			break;
			}*/
			if (!backgroundImage.empty()) {

				imshow("mean background image", backgroundImage);
				int key5 = waitKey(40);

			}


		}

	}
	//string s = "comeon_tyler";  // liz naomi tyler lana
	VideoCapture cap2(s + "_fore.wmv");
	VideoCapture cap3(s + "_fore.wmv");


	//help();

//	Size SizeOfFrame = cv::Size(320, 240);

	// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
	// and use your predefined Size for siplicity 

	//VideoWriter video("Result5.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, true);
	buf.resize(2);
	int cntr = 0;
	double c1 = 0;
	Mat image, image1, motion;
	

	for (;;)
	{

		cap3 >> image;


		if (image.empty())
			break;
		waitKey(10);

		update_mhi(image, motion, 30, c);
		//imshow("Image", image);
		//waitKey(10);
		normalize(motion, motion, 0.0, 255.0, NORM_MINMAX, CV_8UC1);
		imshow("Motion", motion);
		std::string savingName = s + "_mhi.bmp";
		cv::imwrite(savingName, motion);
		//imshow("segment", segmask);
		imshow("mask", image);
		//imshow("orient", orient);
		//video.write(motion);
		//cntr++;

		// ...

		// saving part:

			
		/*else if (cntr == 30) {
		std::string savingName = "All_day_mhi" + std::to_string(cntr) + ".png";
		cv::imwrite(savingName, motion);
		}
		else if (cntr == 60) {
		std::string savingName = "All_day_mhi" + std::to_string(cntr) + ".png";
		cv::imwrite(savingName, motion);
		}
		else if (cntr == 90) {
		std::string savingName = "All_day_mhi" + std::to_string(cntr) + ".png";
		cv::imwrite(savingName, motion);
		}*/
	//	waitKey(1);
		//imshow("")
		//if (waitKey(100) >= 0)
		//break;
	}
	Mat hist,frame1;
	frame1 = imread(s + "_mhi.bmp");
	imshow("hijbij", frame1);
	int x = frame1.rows;
	int y = frame1.cols;
	int width1 = 0, height1 = 0, width2 = 0, height2 = 0;
	//GaussianBlur(frame1, frame1, Size(3, 3), 0.0, 0.0);
	hist = LBP(frame1);
	//hist2 = LBP(frame_rotate);
	static uchar uniform[256] = { // hardcoded 8-neighbour case
		0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
		14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
		58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
		58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
		58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
		58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
		58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
		58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
		58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
		58,58,58,50,51,52,58,53,54,55,56,57
	};
	// 59 bins, bin 58 is the noise/non-uniform slot:
	//hist = Mat::zeros(59, 1, CV_32F);

	Mat_<uchar>im1;
	float max = 0.0;

	int total[944];
	uchar uv;
	for (int z = 0; z < 944; z++) {
		total[z] = 0;
	}
	/*for (int z = 0; z<59; z++)
	{
	a[z] = 0;
	}
	for (int i = 1; i<x - 1; i++)
	{
	for (int j = 1; j<y - 1; j++)
	{
	uv = hist.at<uchar>(i,j);
	a[uniform[uv]] ++; // incr. the resp. histogram bin
	//cout << endl;
	}
	}
	for (int z = 0; z<59; z++)
	{
	cout << z << ": " << a[z] << endl;
	waitKey(100);
	}
	*/
	cout << x << " " << y << endl;
	int count = 0;
	for (int i = 1; i<x - 1; i++)
	{
		for (int j = 10; j<y - 20; j++)
		{
			if (hist.at<uchar>(i, j) < 225) count++;

		}
		cout << i << "   " << count << endl;
		if (count >= 30) {
			if (width1 == 0) width1 = i;
			width2 = i;
		}
		cout << width1 << width2 << endl;
		count = 0;
		//	waitKey(0);
	}
	for (int i = 10; i<y - 20; i++)
	{
		for (int j = 1; j<x - 1; j++)
		{
			if (hist.at<uchar>(j, i) < 225) count++;

		}
		cout << i << "   " << count << endl;
		if (count >= 25) {
			if (height1 == 0) height1 = i;
			height2 = i;
		}
		//cout << width << height << endl;
		count = 0;
		//	waitKey(0);
	}
	int p1, p2, p3, p4, p5, p6, p7, p8;
	p1 = height2 - height1;
	if (p1 % 2 == 1) { p1 = ((p1 + 1) / 2) + height1; }
	else { p1 = (p1 / 2) + height1; }
	p2 = width2 - width1;
	if (p2 % 2 == 1) { p2 = ((p2 + 1) / 2) + width1; }
	else { p2 = (p2 / 2) + width1; }
	p3 = p1 - height1;
	if (p3 % 2 == 1) { p3 = ((p3 + 1) / 2) + height1; }
	else { p3 = (p3 / 2) + height1; }
	p4 = height2 - p1;
	if (p4 % 2 == 1) { p4 = ((p4 + 1) / 2) + p1; }
	else { p4 = (p4 / 2) + p1; }
	p5 = p2 - width1;
	if (p5 % 2 == 1) { p5 = ((p5 + 1) / 2) + width1; }
	else { p5 = (p5 / 2) + width1; }
	p6 = width2 - p2;
	if (p6 % 2 == 1) { p6 = ((p6 + 1) / 2) + p2; }
	else { p6 = (p6 / 2) + p2; }
	for (int z = 0; z<59; z++)
	{
		a1[z] = 0, a2[z] = 0, a3[z] = 0, a4[z] = 0, a5[z] = 0, a6[z] = 0, a7[z] = 0, a8[z] = 0, a9[z] = 0, a10[z] = 0, a11[z] = 0, a12[z] = 0, a13[z] = 0, a14[z] = 0, a15[z] = 0, a16[z] = 0;
	}
	for (int i = width1; i<p5; i++)
	{
		for (int j = height1; j<p3; j++)
		{
			uv = hist.at<uchar>(i, j);
			a1[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p3; j<p1; j++)
		{
			uv = hist.at<uchar>(i, j);
			a2[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p1; j<p4; j++)
		{
			uv = hist.at<uchar>(i, j);
			a3[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p4; j<height2; j++)
		{
			uv = hist.at<uchar>(i, j);
			a4[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
	}
	for (int i = p5; i<p2; i++)
	{
		for (int j = height1; j<p3; j++)
		{
			uv = hist.at<uchar>(i, j);
			a5[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p3; j<p1; j++)
		{
			uv = hist.at<uchar>(i, j);
			a6[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p1; j<p4; j++)
		{
			uv = hist.at<uchar>(i, j);
			a7[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p4; j<height2; j++)
		{
			uv = hist.at<uchar>(i, j);
			a8[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
	}
	for (int i = p2; i<p6; i++)
	{
		for (int j = height1; j<p3; j++)
		{
			uv = hist.at<uchar>(i, j);
			a9[uniform[uv]] ++; // incr. the resp. histogram bin
								//cout << endl;
		}
		for (int j = p3; j<p1; j++)
		{
			uv = hist.at<uchar>(i, j);
			a10[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
		for (int j = p1; j<p4; j++)
		{
			uv = hist.at<uchar>(i, j);
			a11[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
		for (int j = p4; j<height2; j++)
		{
			uv = hist.at<uchar>(i, j);
			a12[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
	}
	for (int i = p6; i<width2; i++)
	{
		for (int j = height1; j<p3; j++)
		{
			uv = hist.at<uchar>(i, j);
			a13[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
		for (int j = p3; j<p1; j++)
		{
			uv = hist.at<uchar>(i, j);
			a14[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
		for (int j = p1; j<p4; j++)
		{
			uv = hist.at<uchar>(i, j);
			a15[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
		for (int j = p4; j<height2; j++)
		{
			uv = hist.at<uchar>(i, j);
			a16[uniform[uv]] ++; // incr. the resp. histogram bin
								 //cout << endl;
		}
	}

	for (int z = 0; z<59; z++)
	{
		total[z] = a1[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 59] = a2[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 118] = a3[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 177] = a4[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 236] = a5[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 295] = a6[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 354] = a7[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 413] = a8[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 472] = a9[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 531] = a10[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 590] = a11[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 649] = a12[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 708] = a13[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 767] = a14[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 826] = a15[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 885] = a16[z];
		//waitKey(1);
	}

	for (int z = 0; z < 944; z++) {
		if (total[z]>max)
		{
			max = total[z];
		}
	}
	for (int z = 0; z < 944; z++) {
		temp[z] = (total[z] / max);
		cout << temp[z] << " // ";
	}
	/*for (int z = 0; z < 944; z++) {
	total[z] = temp[z];
	cout << total[z]<<" \\ ";
	}*/
	float feature[48];
	for (int z = 0; z < 48; z++) { feature[z] = 0.0; }
	for (int z = 0; z < 29; z++) {
		feature[0] += temp[z];
		feature[1] += temp[z + 29];
		feature[3] += temp[z + 59];
		feature[4] += temp[z + 88];
		feature[6] += temp[z + 118];
		feature[7] += temp[z + 147];
		feature[9] += temp[z + 177];
		feature[10] += temp[z + 206];
		feature[12] += temp[z + 236];
		feature[13] += temp[z + 265];
		feature[15] += temp[z + 295];
		feature[16] += temp[z + 324];
		feature[18] += temp[z + 354];
		feature[19] += temp[z + 383];
		feature[21] += temp[z + 413];
		feature[22] += temp[z + 442];
		feature[24] += temp[z + 472];
		feature[25] += temp[z + 501];
		feature[27] += temp[z + 531];
		feature[28] += temp[z + 560];
		feature[30] += temp[z + 590];
		feature[31] += temp[z + 619];
		feature[33] += temp[z + 649];
		feature[34] += temp[z + 678];
		feature[36] += temp[z + 708];
		feature[37] += temp[z + 737];
		feature[39] += temp[z + 767];
		feature[40] += temp[z + 796];
		feature[42] += temp[z + 826];
		feature[43] += temp[z + 855];
		feature[45] += temp[z + 885];
		feature[46] += temp[z + 914];
	}
	feature[2] = temp[58];
	feature[5] = temp[117];
	feature[8] = temp[176];
	feature[11] = temp[235];
	feature[14] = temp[294];
	feature[17] = temp[353];
	feature[20] = temp[412];
	feature[23] = temp[471];
	feature[26] = temp[530];
	feature[29] = temp[589];
	feature[32] = temp[648];
	feature[35] = temp[707];
	feature[38] = temp[767];
	feature[41] = temp[825];
	feature[44] = temp[884];
	feature[47] = temp[943];
	float max2 = 0.0;
	float featuremain[48];
	for (int z = 0; z < 48; z++) {
		if (feature[z]>max2)
		{
			max2 = feature[z];
		}
	}
	for (int z = 0; z < 48; z++) {
		featuremain[z] = (feature[z] / max2);
		cout << featuremain[z] << " // ";
	}
	//histo1();
	ofstream myfile;
	myfile.open("test10.csv", ios::app);
	myfile << "\n";
	//myfile << "Writing this to a file.\n";
	for (int z = 0; z < 48; z++) {
		myfile << featuremain[z] << ",";
	}
	myfile.close();
	//histo(frame1);
	cv::Point pt1(height1, width1);
	// and its bottom right corner.
	cv::Point pt2(height2, width2);
	cv::Point pt3(p1, width1);
	cv::Point pt4(p1, width2);
	cv::Point pt5(height1, p2);
	cv::Point pt6(height2, p2);
	cv::Point pt7(p3, width1);
	cv::Point pt8(p3, width2);
	cv::Point pt9(p4, width1);
	cv::Point pt10(p4, width2);
	cv::Point pt11(height1, p5);
	cv::Point pt12(height2, p5);
	cv::Point pt13(height1, p6);
	cv::Point pt14(height2, p6);
	//cv::Rect rect(0, 0, 30 ,20);
	cv::rectangle(hist, pt1, pt2, cv::Scalar(255, 255, 0), 4);
	cv::line(hist, pt3, pt4, cv::Scalar(255, 255, 0), 2, 8, 0);
	cv::line(hist, pt5, pt6, cv::Scalar(255, 255, 0), 2, 8, 0);
	cv::line(hist, pt7, pt8, cv::Scalar(255, 255, 0), 1, 8, 0);
	cv::line(hist, pt9, pt10, cv::Scalar(255, 255, 0), 1, 8, 0);
	cv::line(hist, pt11, pt12, cv::Scalar(255, 255, 0), 1, 8, 0);
	cv::line(hist, pt13, pt14, cv::Scalar(255, 255, 0), 1, 8, 0);
	imshow("r", hist);
	waitKey(0);
	float r[2][48], t[2][48];
	const int rows = 2;
	const int cols = 48;
	ifstream file("train5.csv");
	ifstream fl("test10.csv");
	if (file.is_open()) {

		for (int i = 0; i < rows; ++i) {  // Reading Data from File
			for (int j = 0; j < cols; ++j) {
				file >> r[i][j];
				file.get(); // Throw away the comma
			}
		}
		for (int i = 0; i < rows; ++i) { // Printing the File data
			for (int j = 0; j < cols; ++j) {
				//cout << r[i][j] << ' ';
			}
			cout << '\n';
		}
	}
	if (fl.is_open()) {

		for (int i = 0; i < 2; ++i) {  // Reading Data from File
			for (int j = 0; j < cols; ++j) {
				fl >> t[i][j];
				fl.get(); // Throw away the comma
			}
		}
		for (int i = 0; i < 2; ++i) { // Printing the File data
			for (int j = 0; j < cols; ++j) {
				//cout << r[i][j] << ' ';
			}
			cout << '\n';
		}
	}
	// Data for visual representation
	int width = 512, height = 512;
	Mat image12 = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	//int labels[24] = { 1,2,3,4,5,6,2,7,6,5,4,5,1,2,3,1,3,4,6,8,8,7,7,8};
	int labels[2] = { 1,2};
	Mat labelsMat(2, 1, CV_32SC1, labels);

	//float trainingData[4][2] = { { 10, 10 },{ 255, 255 },{ 256, 256 },{ 500, 501 } };
	Mat trainingDataMat(2, 48, CV_32FC1, r);
	//Mat trainingData();
	// Set up SVM's parameters
	//SVM::Params params;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(0.0);
	svm->setC(10);
	//svm->setDegree(2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-7));

	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->train(td);
	//svm->trainAuto(td,3);
	Mat testDataMat(2, 48, CV_32FC1, t);
	int predicted[2];
	vector <string> colour;

	// Initialize vector with strings using push_back 
	// command
	colour.push_back("hungry");
	colour.push_back("chat");
	
	for (int i = 0; i <2; i++) {
		cv::Mat sample = testDataMat.row(i);
		predicted[i] = svm->predict(sample);
		cout << predicted[i] << endl;
	}
	/*for (int i = 0; i < 6; i++) {
	//int v= predicted[i];
	cout << clas[predicted[i]+1]<<endl;
	}*/
	for (int i = 0; i < 2; i++) {
		int v = predicted[i];
		cout << colour[v - 1] << endl;
	}



	// Or train the SVM with optimal parameters

	//float testData[2] = { 300,300 };


	//Mat sampleMat = (Mat_<float>(500, 1));
	//float response = svm->predict(testDataMat);
	//Mat trainingData();
	// Set up SVM's parameters

	// Train the SVM with given parameters

	//float response=svm->predict(testDataMat);
	//cout << response;
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);
	/*// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
	Mat sampleMat = (Mat_<float>(1, 2) << j, i);
	//float response = svm->predict(sampleMat);

	//if (response == 1)
	image.at<Vec3b>(i, j) = green;
	//else if (response == -1)
	image.at<Vec3b>(i, j) = blue;
	}*/

	// Show the training data


	int thickness = -1;
	int lineType = 8;
	circle(image, Point(10, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(256, 256), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(500, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = -1;
	lineType = 8;
	Mat sv = svm->getSupportVectors();

	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		//circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image12); // save the image

	imshow("SVM Simple Example", image12);
	waitKey(0);
	return 0;
}
// This is one of the most important thing
// Sizes
//Your VideoWriter Size must correspond with input video.

// Size of your output video 
Size SizeOfFrame = cv::Size(320, 240);

// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
// and use your predefined Size for siplicity 

VideoWriter video("hungry_nazia_face.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, true);
/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(05, 05));
	//face_cascade.detectMultiScale(frame_gray,faces,)
	//face_cascade.detectMultiScale()

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);
		
		ellipse(frame, center, Size(30, 30), 0, 0, 360, Scalar(0, 0, 0), -1, 8, 0);
		Rect r = Rect(0, 0, 160, 25);
		//create a Rect with top-left vertex at (10,20), of width 40 and height 60 pixels.

		rectangle(frame, r, Scalar(0, 0, 0), -1, 8, 0);
		//draw the rect defined by r with line thickness 1 and Blue color
		//rectangle(frame, cvPoint(0, 0), cvPoint(320, 50), CV_RGB(0, 0, 0), -1, 8);
		Rect r1 = Rect(0, 25, 60, 25);
		//create a Rect with top-left vertex at (10,20), of width 40 and height 60 pixels.

		rectangle(frame, r1, Scalar(0, 0, 0), -1, 8, 0);
		//draw the rect defined by r with line thickness 1 and Blue color

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		/*eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
		Point eye_center(faces[0].x + eyes[0].x + eyes[0].width / 2, faces[0].y + eyes[0].y + eyes[0].height / 2);
		int radius = cvRound((eyes[0].width + eyes[0].height)*0.25);
		circle(frame, eye_center, radius, Scalar(255, 255, 255), 4, 8, 0);
		}*/
		//subtract(frame_gray, faceROI, frame);
		//imshow("hi",faces);
		video.write(frame);
	}
	//-- Show what you got
	imshow(window_name, frame);

	waitKey(10);
}