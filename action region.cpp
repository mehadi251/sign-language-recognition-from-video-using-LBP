#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
float temp[944];
int a1[59];
int a16[59];

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
void histo()
{
	//Mat hist(255, 59, CV_32FC1);
	Mat hist1(255, 59, CV_32FC1, 255);
	for (int i = 885; i<944; i++)
	{
		line(hist1, Point((i-885),( 256.0 - (temp[i]))), Point((i-885), 256),0,2);
	}
	imshow("Input Hist", hist1);
	imwrite("histogram_featurea16.jpg", hist1);
	waitKey(0);
}
//
// Mat_<uchar> used here for convenient () operator indexing
//
/*uchar lbp(const Mat_<uchar> & img, int x, int y)
{
// this is pretty much the same what you already got..
uchar v = 0;
uchar c = img(y, x);
v += (img(y - 1, x) > c) << 0;
v += (img(y - 1, x + 1) > c) << 1;
v += (img(y, x + 1) > c) << 2;
v += (img(y + 1, x + 1) > c) << 3;
v += (img(y + 1, x) > c) << 4;
v += (img(y + 1, x - 1) > c) << 5;
v += (img(y, x - 1) > c) << 6;
v += (img(y - 1, x - 1) > c) << 7;
return v;
}*/
//your code above tries to build an lbp image first, i'm going to skip that, and calculate the histogram directly from the (uniform) lbp values.

//
// you would usually apply this to small patches of an image and concatenate the histograms
//
/*void histo(Mat src)
{
Mat gray = src;
namedWindow("Gray", 1);    imshow("Gray", gray);

// Initialize parameters
int histSize = 59;    // bin size
float range[] = { 0, 58 };
const float *ranges[] = { range };

// Calculate histogram
MatND hist;
calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

// Show the calculated histogram in command window
double total;
total = gray.rows * gray.cols;
for (int h = 0; h < histSize; h++)
{
float binVal = hist.at<float>(h);
cout << binVal<<endl;
}

// Plot the histogram
int hist_w = 512; int hist_h = 400;
int bin_w = cvRound((double)hist_w / histSize);

Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

for (int i = 1; i < histSize; i++)
{
line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
Scalar(255, 0, 0), 2, 8, 0);
}

namedWindow("Result", 1);    imshow("Result", histImage);

waitKey(0);
}*/
int main()
{
	Mat frame1, hist;

	frame1 = imread("hungry_naz.bmp");//liz tyler naomi
	//flip(frame1, frame_rotate, 1);
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
	int a1[59],a2[59],a3[59],a4[59],a5[59],a6[59],a7[59],a8[59],a9[59],a10[59], a11[59], a12[59], a13[59],a14[59], a15[59], a16[59];
	Mat_<uchar>img;
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
	cout << x <<" "<< y << endl;
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
		if (count >=25 ) {
			if (height1 == 0) height1 = i;
			height2 = i;
		}
		//cout << width << height << endl;
		count = 0;
		//	waitKey(0);
	}
	int p1, p2, p3, p4, p5, p6, p7, p8;
	p1 = height2 - height1;
	if (p1 % 2 == 1) { p1 = ((p1 + 1)/2)+height1; }
	else { p1 = (p1 / 2)+height1; }
	p2 = width2 - width1;
	if (p2 % 2 == 1) { p2 = ((p2 + 1)/2)+width1; }
	else { p2 = (p2 / 2)+width1; }
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
		a1[z] = 0,a2[z]=0,a3[z]=0,a4[z]=0,a5[z]=0,a6[z]=0, a7[z] = 0, a8[z] = 0, a9[z] = 0, a10[z] = 0, a11[z] = 0, a12[z] = 0,  a13[z] = 0, a14[z] = 0, a15[z] = 0, a16[z] = 0;
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
		total[z]=a1[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z+59]=a2[z];
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 118] = a3[z] ;
		//waitKey(1);
	}
	for (int z = 0; z<59; z++)
	{
		total[z + 177] = a4[z] ;
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
	//float temp[1][944];
	for (int z = 0; z < 944; z++) {
		if(total[z]>max)
		{
			max = total[z];
		}
	}
	for (int z = 0; z < 944; z++) {
		temp[z] = (total[z]/max)*255.0 ;
		cout << temp[z]<<" // ";
	}
	Mat histnew(1,944,CV_32FC1,temp);
	//histo();
	/*for (int z = 0; z < 944; z++) {
		total[z] = temp[z];
		cout << total[z]<<" \\ ";
	}*/
	float feature[48];
	for (int z = 0; z < 48; z++) { feature[z] = 0.0; }
	for (int z = 0; z < 29; z++) {
		feature[0] += temp[z];
		feature[1] += temp[z + 29];
		feature[3] += temp[z+59];
		feature[4] += temp[z + 88];
		feature[6] += temp[z+118];
		feature[7] += temp[z + 147];
		feature[9] += temp[z+177];
		feature[10] += temp[z + 206];
		feature[12] += temp[z+236];
		feature[13] += temp[z + 265];
		feature[15] += temp[z+295];
		feature[16] += temp[z + 324];
		feature[18] += temp[z+354];
		feature[19] += temp[z + 383];
		feature[21] += temp[z+413];
		feature[22] += temp[z + 442];
		feature[24] += temp[z+472];
		feature[25] += temp[z + 501];
		feature[27] += temp[z+531];
		feature[28] += temp[z + 560];
		feature[30] += temp[z+590];
		feature[31] += temp[z + 619];
		feature[33] += temp[z+649];
		feature[34] += temp[z + 678];
		feature[36] += temp[z+708];
		feature[37] += temp[z + 737];
		feature[39] += temp[z+767];
		feature[40] += temp[z + 796];
		feature[42] += temp[z+826];
		feature[43] += temp[z + 855];
		feature[45] += temp[z+885];
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
		featuremain[z] = (feature[z] / max2)*255.0;
		cout << featuremain[z] << " // ";
	}
	
	/*float future[240];
	for (int z = 0; z < 240; z++) { future[z] = 0.0; }
	//for (int z = 0; z < 4; z++) {
	//	future[0] += temp[z];
	//	future[1] += temp[z + 4];
	//	future[2] += temp[z + 8];
	//	future[3] += temp[z + 12];
	//	future[4] += temp[z + 16];
	//	future[5] += temp[z + 20];
	//	future[6] += temp[z + 24];
	//	future[7] += temp[z + 28];
	//	future[8] += temp[z + 32];
	//	future[9] += temp[z + 36];
	//	future[10] += temp[z + 40];
	//	future[11] += temp[z + 44];
	//	future[12] += temp[z + 48];
	//	future[13] += temp[z + 52];
	//	future[15] += temp[z+59];
	//	future[16] += temp[z + 63];
	//	future[17] += temp[z + 67];
	//	future[18] += temp[z + 71];
	//	future[19] += temp[z + 75];
	//	future[20] += temp[z + 79];
	//	future[21] += temp[z + 83];
	//	future[22] += temp[z + 87];
	//	future[23] += temp[z + 91];
	//	future[24] += temp[z + 95];
	//	future[25] += temp[z + 99];
	//	future[26] += temp[z + 103];
	//	future[27] += temp[z + 107];
	//	future[28] += temp[z + 111];
	//	future[30] += temp[z+118];
	//	future[31] += temp[z + 122];
	//	future[32] += temp[z + 126];
	//	future[33] += temp[z + 130];
	//	future[34] += temp[z + 134];
	//	future[35] += temp[z + 138];
	//	future[36] += temp[z + 142];
	//	future[37] += temp[z + 146];
	//	future[38] += temp[z + 150];
	//	future[39] += temp[z + 154];
	//	future[40] += temp[z + 158];
	//	future[41] += temp[z + 162];
	//	future[42] += temp[z + 166];
	//	future[43] += temp[z + 170];
	//	future[45] += temp[z+177];
	//	future[46] += temp[z + 181];
	//	future[47] += temp[z + 185];
	//	future[48] += temp[z + 189];
	//	future[49] += temp[z + 193];
	//	future[50] += temp[z + 197];
	//	future[51] += temp[z + 201];
	//	future[52] += temp[z + 205];
	//	future[53] += temp[z + 209];
	//	future[54] += temp[z + 213];
	//	future[55] += temp[z + 217];
	//	future[56] += temp[z + 221];
	//	future[57] += temp[z + 225];
	//	future[58] += temp[z + 229];
	//	future[60] += temp[z+236];
	//	future[61] += temp[z + 240];
	//	future[62] += temp[z + 244];
	//	future[63] += temp[z + 248];
	//	future[64] += temp[z + 252];
	//	future[65] += temp[z + 256];
	//	future[66] += temp[z + 260];
	//	future[67] += temp[z + 264];
	//	future[68] += temp[z + 268];
	//	future[69] += temp[z + 272];
	//	future[70] += temp[z + 276];
	//	future[71] += temp[z + 280];
	//	future[72] += temp[z + 284];
	//	future[73] += temp[z + 288];
	//	future[75] += temp[z+295];
	//	future[76] += temp[z + 299];
	//	future[77] += temp[z + 303];
	//	future[78] += temp[z + 307];
	//	future[79] += temp[z + 311];
	//	future[80] += temp[z + 315];
	//	future[81] += temp[z + 319];
	//	future[82] += temp[z + 323];
	//	future[83] += temp[z + 327];
	//	future[84] += temp[z + 331];
	//	future[85] += temp[z + 335];
	//	future[86] += temp[z + 339];
	//	future[87] += temp[z + 343];
	//	future[88] += temp[z + 347];
	//	future[90] += temp[z+354];
	//	future[91] += temp[z + 358];
	//	future[92] += temp[z + 362];
	//	future[93] += temp[z + 366];
	//	future[94] += temp[z + 370];
	//	future[95] += temp[z + 374];
	//	future[96] += temp[z + 378];
	//	future[97] += temp[z + 382];
	//	future[98] += temp[z + 386];
	//	future[99] += temp[z + 390];
	//	future[100] += temp[z + 394];
	//	future[101] += temp[z + 398];
	//	future[102] += temp[z + 402];
	//	future[103] += temp[z + 406];
	//	future[105] += temp[z+413];
	//	future[106] += temp[z + 417];
	//	future[107] += temp[z + 421];
	//	future[108] += temp[z + 425];
	//	future[109] += temp[z + 429];
	//	future[110] += temp[z + 433];
	//	future[111] += temp[z + 437];
	//	future[112] += temp[z + 441];
	//	future[113] += temp[z + 445];
	//	future[114] += temp[z + 449];
	//	future[115] += temp[z + 453];
	//	future[116] += temp[z + 457];
	//	future[117] += temp[z + 461];
	//	future[118] += temp[z + 465];
	//	future[120] += temp[z+472];
	//	future[121] += temp[z + 476];
	//	future[122] += temp[z + 480];
	//	future[123] += temp[z + 484];
	//	future[124] += temp[z + 488];
	//	future[125] += temp[z + 492];
	//	future[126] += temp[z + 496];
	//	future[127] += temp[z + 500];
	//	future[128] += temp[z + 504];
	//	future[129] += temp[z + 508];
	//	future[130] += temp[z + 512];
	//	future[131] += temp[z + 516];
	//	future[132] += temp[z + 520];
	//	future[133] += temp[z + 524];
	//	future[135] += temp[z+531];
	//	future[136] += temp[z + 535];
	//	future[137] += temp[z + 539];
	//	future[138] += temp[z + 543];
	//	future[139] += temp[z + 547];
	//	future[140] += temp[z + 551];
	//	future[141] += temp[z + 555];
	//	future[142] += temp[z + 559];
	//	future[143] += temp[z + 563];
	//	future[144] += temp[z + 567];
	//	future[145] += temp[z + 571];
	//	future[146] += temp[z + 575];
	//	future[147] += temp[z + 579];
	//	future[148] += temp[z + 583];
	//	future[150] += temp[z];
	//	future[151] += temp[z + 4];
	//	future[152] += temp[z + 8];
	//	future[153] += temp[z + 12];
	//	future[154] += temp[z + 16];
	//	future[155] += temp[z + 20];
	//	future[156] += temp[z + 24];
	//	future[157] += temp[z + 28];
	//	future[158] += temp[z + 32];
	//	future[159] += temp[z + 36];
	//	future[160] += temp[z + 40];
	//	future[161] += temp[z + 44];
	//	future[162] += temp[z + 48];
	//	future[163] += temp[z + 52];
	//}
	int k = 0;
	for (int z = 0; z < 943; z=z+59) {
		future[k + 0] += temp[z] + temp[z + 1] + temp[z + 2] + temp[z + 3];
		future[k + 1] += temp[z+4] + temp[z + 5] + temp[z + 6] + temp[z + 7];
		future[k + 2] += temp[z+8] + temp[z + 9] + temp[z + 10] + temp[z + 11];
		future[k + 3] += temp[z+12] + temp[z + 13] + temp[z + 14] + temp[z + 15];
		future[k + 4] += temp[z+16] + temp[z + 17] + temp[z + 18] + temp[z + 19];
		future[k + 5] += temp[z+20] + temp[z + 21] + temp[z + 22] + temp[z + 23];
		future[k + 6] += temp[z+24] + temp[z + 25] + temp[z + 26] + temp[z + 27];
		future[k + 7] += temp[z+28] + temp[z + 29] + temp[z + 30] + temp[z + 31];
		future[k + 8] += temp[z+32] + temp[z + 33] + temp[z + 34] + temp[z + 35];
		future[k + 9] += temp[z+36] + temp[z + 37] + temp[z + 38] + temp[z + 39];
		future[k + 10] += temp[z+40] + temp[z + 41] + temp[z + 42] + temp[z + 43];
		future[k + 11] += temp[z+44] + temp[z + 45] + temp[z + 46] + temp[z + 47];
		future[k + 12] += temp[z+48] + temp[z + 49] + temp[z + 50] + temp[z + 51];
		future[k + 13] += temp[z+52] + temp[z + 53] + temp[z + 54] + temp[z + 55];
		future[k + 14] += temp[z+56] + temp[z + 57];
		k = k + 15;

	}
	float max2 = 0.0;
	for (int z = 0; z < 240; z++) {
		if (future[z] > max2)
			max2 = future[z];
	}
	float futuremain[240];
	for (int z = 0; z < 240; z++) { futuremain[z] = (future[z] / max2)*255.0; }*/
	ofstream myfile;
	myfile.open("test10.csv",ios::app);
	//myfile << "\n";
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
	cv::rectangle(hist, pt1, pt2, cv::Scalar(0, 255, 0),4);
	cv::line(hist, pt3, pt4, cv::Scalar(0, 255, 0), 2, 8, 0);
	cv::line(hist, pt5, pt6, cv::Scalar(0, 255, 0), 2, 8, 0);
	cv::line(hist, pt7, pt8, cv::Scalar(0, 255, 0), 1, 8, 0);
	cv::line(hist, pt9, pt10, cv::Scalar(0, 255, 0), 1, 8, 0);
	cv::line(hist, pt11, pt12, cv::Scalar(0, 255, 0), 1, 8, 0);
	cv::line(hist, pt13, pt14, cv::Scalar(0, 255, 0), 1, 8, 0);
	imshow("r", hist);
	//imshow("ro", frame_rotate);
	imwrite("hungry_naz_region.bmp", hist);
	//cvDrawRect(hist, CvPoint(width1, height1), CvPoint(width2, height2), 255, 3, 16, 0);
	//cout << width1 << " " << width2 << " " << height1 << " " << height2 << " "<<p2<<" "<<p1<<endl;
	waitKey(0);
	return 0;
}