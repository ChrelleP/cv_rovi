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
  Mat image_source = imread("../lego.jpg", CV_LOAD_IMAGE_COLOR);
  Mat image_HSV;
  Mat image_segmented;
  Mat HSV_channels[3];

  int Hue_lower = 0;
  int Hue_upper = 50;
  int Sat_lower = 55;
  int Sat_upper = 200;
  int Val_lower = 55;
  int Val_upper = 200;

  cvtColor(image_source, image_HSV, CV_BGR2HSV);

  split(image_HSV, HSV_channels);

  inRange(image_HSV, Scalar(Hue_lower, Sat_lower, Val_lower), Scalar(Hue_upper, Sat_upper, Val_upper), image_segmented);

  printf("Type: %i\n", image_HSV.type());

  imshow("Result - Source", image_source);
  imshow("Result - Hue", HSV_channels[0]);
  imshow("Result - Saturation", HSV_channels[1]);
  imshow("Result - Value", HSV_channels[2]);



  waitKey(0);

  return 0;
}
