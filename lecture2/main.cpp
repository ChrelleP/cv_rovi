//Lecture 2 Computer Vision
//
// Adjusting colour and other stuff

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void display_grey_hist(Mat image)
{
  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
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

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat_<Vec3b> source;
    Mat grey;
    Mat grey_2;

    source = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! source.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // Grayscale
    cvtColor(source, grey, CV_BGR2GRAY);
    grey_2 = grey.clone();

    // Looping and adding
    int new_value;
    for(int i = 0; i < grey_2.rows; i++)
    {
      for(int j = 0; j < grey_2.cols; j++)
      {
          grey_2.at<uchar>(i,j) = saturate_cast<uchar>(grey_2.at<uchar>(i,j) + 50);
      }
    }

  // Display histogram
  display_grey_hist(grey);
  display_grey_hist(grey_2);

  equalizeHist(grey, grey);
  equalizeHist(grey_2, grey_2);

  display_grey_hist(grey);
  display_grey_hist(grey_2);

  // Display image
  namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", grey );                   // Show our image inside it.
  imshow( "Source", grey_2 );

  // Printing info
  printf("Source | Type: %i \t depth: %i \t channels: %i \n", source.type(), source.depth(), source.channels());
  printf("Grey | Type: %i \t depth: %i \t channels: %i \n", grey.type(), grey.depth(), grey.channels());

  waitKey(0);                                          // Wait for a keystroke in the window
  return 0;
}
