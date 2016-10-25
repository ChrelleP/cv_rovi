//____________________ INCLUDE FILES ____________________
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "stdio.h"

//____________________ NAME SPACES ____________________
using namespace cv;
using namespace std;

int main()
{
  Mat image_source = imread("../coins.jpg",CV_LOAD_IMAGE_GRAYSCALE);
  Mat image_result = image_source.clone();

  // Thresholding
  int threshold_value = 110;
  int threshold_type = 0;
  int max_BINARY_value = 255;

  GaussianBlur(image_source, image_source, Size(15,15), 2);
  threshold(image_source, image_result, threshold_value, max_BINARY_value, threshold_type);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  Mat contour_image = image_result.clone();
  findContours(contour_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( image_result.size(), CV_8UC3 );
  RNG rng(12345);
  for( int i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }

  cout << "Amount of contours: " << contours.size() << endl;
  int largest = 0;
  int smallest = 0;

  vector<vector<Point> > contours_test;

  for(int i = 0; i < contours.size(); i++)
  {
    if(contours[i].size() > 150 && contours[i].size() < 400){
      contours_test.push_back(contours[i]);
      cout << contours[i].size() << endl;
    }

  }

  /*vector<vector<Point> > coin_5;
  vector<vector<Point> > coin_10;
  vector<vector<Point> > coin_15;
  vector<vector<Point> > coin_20;
  vector<vector<Point> > coin_50;


  for(int i = 0; i < contours_test.size(); i++)
  {
    if(contours_test[i] > 150 && contours_test[i] < 200)
      coin_5.push_back(contours_test[i]);
    if(contours_test[i] > 200 && contours_test[i] < 250)
      coin_10.push_back(contours_test[i]);
    if(contours_test[i] > 150 && contours_test[i] < 250)
      coin_15.push_back(contours_test[i]);
    if(contours_test[i] > 150 && contours_test[i] < 250)
      coin_20.push_back(contours_test[i]);
    if(contours_test[i] > 150 && contours_test[i] < 250)
      coin_50.push_back(contours_test[i]);
  }*/

  Mat drawing_test = Mat::zeros( image_result.size(), CV_8UC3 );
  for( int i = 0; i< contours_test.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing_test, contours_test, i, color, 2, 8, hierarchy, 0, Point() );
  }

  Size size(round(0.5*image_result.cols),round(0.5*image_result.rows));
  resize(image_result, image_result, size);
  resize(drawing, drawing, size);
  resize(drawing_test, drawing_test, size);

  imshow("Result - Binary", image_result);
  imshow("Result - Drawing", drawing);
  imshow("Result - Drawing test", drawing_test);

  waitKey(0);

  return 0;
}
