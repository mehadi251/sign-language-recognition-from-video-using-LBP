#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <Windows.h>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\video\tracking.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{

	// Init background substractor
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empy input img, foreground and background image and foreground mask.
	Mat img, foregroundMask, backgroundImage, foregroundImg;

	// capture video from source 0, which is web camera, If you want capture video from file just replace //by  
	VideoCapture cap("All_day.mov");
	//VideoCapture cap(0);
	// This is one of the most important thing
	// Sizes
	//Your VideoWriter Size must correspond with input video.

	// Size of your output video 
	Size SizeOfFrame = cv::Size(320, 240);

	// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
	// and use your predefined Size for siplicity 

	VideoWriter video("All_day_fore.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, false);
	//VideoWriter video1("all_test_fore.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, true);
	//VideoWriter vid();

	Mat prev, next,result;
	Mat flow(SizeOfFrame,CV_32FC2);
	// main loop to grab sequence of input files
	int i = 0;
	for ( ; ; ) {

		bool ok = cap.grab();

		if (ok == false) {

			std::cout << "Video Capture Fail" << std::endl;


		}
		else {

			// obtain input image from source
			cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
			// Just resize input image if you want
			resize(img, img, Size(320, 240));
			Size frameSize(320,240);

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
				calcOpticalFlowFarneback(prev, next, flow, .5, 1, 3, 3, 5, 1.1, 0);
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
			Rect r = Rect(0, 0, 160, 25);
			//create a Rect with top-left vertex at (10,20), of width 40 and height 60 pixels.

			rectangle(foregroundMask, r, Scalar(0, 0, 0), -1, 8, 0);
			//draw the rect defined by r with line thickness 1 and Blue color
			//rectangle(frame, cvPoint(0, 0), cvPoint(320, 50), CV_RGB(0, 0, 0), -1, 8);
			Rect r1 = Rect(0, 25, 60, 25);
			//create a Rect with top-left vertex at (10,20), of width 40 and height 60 pixels.

			rectangle(foregroundMask, r1, Scalar(0, 0, 0), -1, 8, 0);
			imshow("foreground mask", foregroundMask);
			imshow("foreground image", foregroundImg);
			result = flow.reshape(1);
			imshow("flow", result);

			video.write(foregroundMask);
			//video1.write(foregroundImg);
			
			int key6 = waitKey(10);
			/*if (waitKey(10) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			{
				cout << "esc key is pressed by user" << endl;
				break;
			}*/
			if (!backgroundImage.empty()) {

				imshow("mean background image", backgroundImage);
				int key5 = waitKey(1);

			}


		}

	}


	return EXIT_SUCCESS;
}