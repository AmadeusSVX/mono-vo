/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 4000
#define MIN_NUM_FEAT 2000

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

// no way to estimate correct scale just using monocular vo
double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{
	return 1.0;
}

void Triangulate(
	const std::vector<cv::Point2f> &vUnPt1,
	const std::vector<cv::Point2f> &vUnPt2,
	const cv::Mat rvec1,
	const cv::Mat rvec2,
	const cv::Mat tvec1,
	const cv::Mat tvec2,
	std::vector<cv::Point3f> &vpt3d
)
{
	cv::Mat R1, R2;

	cv::Rodrigues(rvec1, R1);
	cv::Rodrigues(rvec2, R2);

	cv::Mat P1(3, 4, R1.type()), P2(3, 4, R2.type());
	R1.copyTo(P1(cv::Rect(0, 0, 3, 3)));
	R2.copyTo(P2(cv::Rect(0, 0, 3, 3)));

	tvec1.copyTo(P1(cv::Rect(3, 0, 1, 3)));
	tvec2.copyTo(P2(cv::Rect(3, 0, 1, 3)));

	cv::Mat triangulated;
	cv::triangulatePoints(P1, P2, vUnPt1, vUnPt2, triangulated);

	vpt3d.resize(vUnPt1.size());

	for (int i = 0, iend = int(vUnPt1.size()); i < iend; ++i) {

		float x = triangulated.at < float >(0, i);
		float y = triangulated.at < float >(1, i);
		float z = triangulated.at < float >(2, i);
		float w = triangulated.at < float >(3, i);

		vpt3d[i].x = x / w;
		vpt3d[i].y = y / w;
		vpt3d[i].z = z / w;
	}
}


int main( int argc, char** argv )
{
	Mat img_1, img_2;
	Mat R_f, t_f; //the final rotation and tranlation vectors containing the 

	ofstream myfile;
	myfile.open ("results1_1.txt");

	double scale = 1.00;
	char filename1[200];
	char filename2[200];
	sprintf(filename1, "./dataset/sequences/00/image_0/%06d.png", 0);
	sprintf(filename2, "./dataset/sequences/00/image_0/%06d.png", 1);

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;  
	cv::Point textOrg(10, 50);

	//read the first two frames from the dataset
	Mat img_1_c = imread(filename1);
	Mat img_2_c = imread(filename2);

	if ( !img_1_c.data || !img_2_c.data )
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; return -1;
	}

	// we work with grayscale images
	cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

	// feature detection, tracking
	vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
	vector<Point3f> points3d;			     //vectors to store the coordinates of the feature points
	featureDetection(img_1, points1);        //detect features in img_1
	vector<uchar> status;
	featureTracking(img_1, img_2,points1, points2, status); //track those features to img_2

	//TODO: add a fucntion to load these values directly from KITTI's calib files
	// WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
	double focal = 718.8560;
	cv::Point2d pp(607.1928, 185.2157);
	//recovering the pose and the essential matrix
	Mat E, R, R0, tvec, rvec, tvec0, rvec0, mask;
	R0 = cv::Mat::eye(3, 3, CV_32F);
	rvec = cv::Mat(3, 1, CV_32F);
	rvec0 = cv::Mat(3, 1, CV_32F);
	cv::Rodrigues(R0, rvec0);
	tvec0 = cv::Mat::zeros(3, 1, CV_32F);

	E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points2, points1, R, tvec, focal, pp, mask);
	Rodrigues(R, rvec);

//	Triangulate(points1, points2, rvec0, rvec, tvec0, tvec, points3d);

	Mat prevImage = img_2;
	Mat currImage;
	vector<Point2f> prevFeatures = points2;
	vector<Point2f> currFeatures;

	char filename[100];

	R_f = R.clone();
	t_f = t.clone();

	clock_t begin = clock();

	namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

	Mat traj = Mat::zeros(600, 600, CV_8UC3);

	for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)
	{
		sprintf(filename, "./dataset/sequences/00/image_0/%06d.png", numFrame);
		//cout << numFrame << endl;
		Mat currImage_c = imread(filename);
		cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
		vector<uchar> status;
		featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

		E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

		Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);

		//this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
		for(int i=0;i<prevFeatures.size();i++)	
		{
  			prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  			prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  			currPts.at<double>(0,i) = currFeatures.at(i).x;
  			currPts.at<double>(1,i) = currFeatures.at(i).y;
		}

		for (int i = 0; i < prevFeatures.size(); i++)
		{
			cv::line(currImage_c, prevFeatures[i], currFeatures[i], cv::Scalar(255, 0, 0));
			cv::circle(currImage_c, currFeatures[i], 3, cv::Scalar(255, 0, 0));
		}

		scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

		//cout << "Scale is " << scale << endl;

		if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

			t_f = t_f + scale*(R_f*t);
			R_f = R*R_f;

		}
  	
		else {
			//cout << "scale below 0.1, or incorrect translation" << endl;
		}
    
		// lines for printing results
		// myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

		// a redetection is triggered in case the number of feautres being trakced go below a particular threshold
		if (prevFeatures.size() < MIN_NUM_FEAT)
		{
		//cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
		//cout << "trigerring redection" << endl;
			featureDetection(prevImage, prevFeatures);
			featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
		}

		prevImage = currImage.clone();
		prevFeatures = currFeatures;

		int x = int(t_f.at<double>(0)) + 300;
		int y = int(t_f.at<double>(2)) + 100;
		circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

		rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0));
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		imshow( "Road facing camera", currImage_c );
		imshow( "Trajectory", traj );

		waitKey(1);

	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

  //cout << R_f << endl;
  //cout << t_f << endl;

	return 0;
}