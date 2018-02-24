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
#include "opencv2\objdetect\objdetect.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

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

	//-- 2. Read the video stream
	VideoCapture cap1;// tyler liz naomi lana jaime
	cap1.open(0);
	time_t start, end;
	time(&start);
	//if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
while (cap1.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		// obtain input image from source
		cap1.retrieve(frame, CV_CAP_OPENNI_BGR_IMAGE);
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

	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empy input img, foreground and background image and foreground mask.
	Mat img, foregroundMask, backgroundImage, foregroundImg;

	// capture video from source 0, which is web camera, If you want capture video from file just replace //by  
	VideoCapture cap("hungry_awalsir_face.wmv");
	//VideoCapture cap(0);
	// This is one of the most important thing
	// Sizes
	//Your VideoWriter Size must correspond with input video.

	// Size of your output video 
	Size SizeOfFrame = cv::Size(320, 240);

	// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
	// and use your predefined Size for siplicity 

	VideoWriter video("hungry_awalface_fore.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, false);
Mat prev, next, result;
	Mat flow(SizeOfFrame, CV_32FC2);
	// main loop to grab sequence of input files
	int i = 0;
	for (; ; ) {

		bool ok = cap.grab();

		if (ok == false) {

			std::cout << "Video Capture Fail" << std::endl;


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
	return 0;
}
// This is one of the most important thing
// Sizes
//Your VideoWriter Size must correspond with input video.

// Size of your output video 
Size SizeOfFrame = cv::Size(320, 240);

// On windows write video into Result.wmv with codec W M V 2 at 30 FPS 
// and use your predefined Size for siplicity 

VideoWriter video("hungry_awalsir_face.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, SizeOfFrame, true);
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