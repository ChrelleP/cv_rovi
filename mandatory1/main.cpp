//______________ MANDATORY EXERCISE 1 | COMPUTER VISION ________________________
// Analysis of various images, and restoration using spatial and frequency
// domain filters.
//
// Made by: Mathias Thor               mthor13@student.sdu.dk
//          Christian Koed Pedersen    chped13@student.sdu.dk
//______________________________________________________________________________

//____________________ INCLUDE FILES ____________________
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "stdio.h"

//____________________ NAME SPACES ____________________
using namespace cv;
using namespace std;

//____________________ FUNCTIONS ____________________
void analyze_grey_image(Mat image)
{
  //********** MAKING HISTOGRAM **************
  // Establish the number of bins
  int histSize = 256;

  // Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  // Compute the histogram:
  calcHist( &image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histogram
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw the histogram
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
  waitKey(0);

  return;
}


//____________________ MAIN PROGRAM ____________________
int main( int argc, char** argv)
{
  //************ VARIABLES AND DATA ***************
  Mat image_source_1;
  Mat image_source_2;
  Mat image_source_3;
  Mat image_source_4_1;
  Mat image_source_4_2;
  Mat image_source_5;

  image_source_1    = imread("./Images/ImagesForStudents/Image1.png", CV_LOAD_IMAGE_GRAYSCALE);
  image_source_2    = imread("./Images/ImagesForStudents/Image2.png", CV_LOAD_IMAGE_GRAYSCALE);
  image_source_3    = imread("./Images/ImagesForStudents/Image3.png", CV_LOAD_IMAGE_GRAYSCALE);
  image_source_4_1  = imread("./Images/ImagesForStudents/Image4_1.png", CV_LOAD_IMAGE_GRAYSCALE);
  image_source_4_2  = imread("./Images/ImagesForStudents/Image4_2.png", CV_LOAD_IMAGE_GRAYSCALE);
  image_source_5    = imread("./Images/ImagesForStudents/Image5_optional.png", CV_LOAD_IMAGE_GRAYSCALE);

  Mat image_modified_1   = image_source_1.clone();
  Mat image_modified_2   = image_source_2.clone();
  Mat image_modified_3   = image_source_3.clone();
  Mat image_modified_4_1 = image_source_4_1.clone();
  Mat image_modified_4_2 = image_source_4_2.clone();
  Mat image_modified_5   = image_source_5.clone();

  //************* DISPLAY IMAGES ******************
  namedWindow( "Display window", WINDOW_AUTOSIZE );   // Create a window for display.
  imshow( "Image 1", image_source_1 );
  imshow( "Image 2", image_source_2 );
  imshow( "Image 3", image_source_3 );
  imshow( "Image 4_1", image_source_4_1 );
  imshow( "Image 4_2", image_source_4_2 );
  imshow( "Image 5", image_source_5 );

  waitKey(0);                                          // Wait for a keystroke in the window

  return 0;
}
