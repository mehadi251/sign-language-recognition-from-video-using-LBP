#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ml;

int main(int, char**)
{
	float r[2][48],t[1][48];
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

		for (int i = 0; i < 1; ++i) {  // Reading Data from File
			for (int j = 0; j < cols; ++j) {
				fl >> t[i][j];
				fl.get(); // Throw away the comma
			}
		}
		for (int i = 0; i < 1; ++i) { // Printing the File data
			for (int j = 0; j < cols; ++j) {
				//cout << r[i][j] << ' ';
			}
			cout << '\n';
		}
	}
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
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
	Mat testDataMat(1, 48, CV_32FC1, t);
	int predicted[1];
	vector <string> colour;

	// Initialize vector with strings using push_back 
	// command
	
	colour.push_back("hungry");
	colour.push_back("baseball");
	
	/*colour.push_back("accident");
	colour.push_back("adopt");
	colour.push_back("baseball");
	colour.push_back("awkward");
	colour.push_back("chat");
	colour.push_back("come on");
	colour.push_back("all");
	colour.push_back("again");*/
	
	for (int i = 0; i <1; i++) {
		cv::Mat sample = testDataMat.row(i);
		predicted[i] = svm->predict(sample);
		cout << predicted[i] << endl;
	}
	/*for (int i = 0; i < 6; i++) {
		//int v= predicted[i];
		cout << clas[predicted[i]+1]<<endl;
	}*/
	for (int i = 0; i < 1; i++) {
		int v = predicted[i];
		cout << colour[v-1] << endl;
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
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0,0,255);
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

	imwrite("result.png", image); // save the image

imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
	return 0;
}