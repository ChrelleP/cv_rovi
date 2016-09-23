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

//______________ FUNCTION DECLARATIONS __________________
Mat draw_histogram(Mat);
Mat draw_magnitudeplot(Mat_<float>);
void median_filter(Mat src, Mat dst);
void dftshift(Mat_<float>&);
void resize_image(Mat&, float);
void analyse_image(Mat);
void analyse_sample(Mat);


//____________________ MAIN PROGRAM ____________________
int main( int argc, char** argv)
{
  //************ VARIABLES AND DATA ***************
  // Image1.png, Image2.png, Image3.png, Image4_1.png, Image4_2.png, Image5_optional.png,
  Mat image_source    = imread("../Images/Image1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat image_modified  = image_source.clone();

  // ************ ANALYSE IMAGE *******************
  analyse_sample(image_source);
  rectangle(image_source, Point(1350,1200), Point(1450,1300), 255, CV_FILLED); // image sample
  analyse_image(image_source);

  // ************ MODIFY IMAGE ********************
  //medianBlur(image_source_2, image_modified_2, 15); // TODO Write about the different kernel sizes

  //************* DISPLAY IMAGES ******************
  resize_image(image_modified, 0.25);
  //imshow( "Image - Modified", image_modified );

  waitKey(0);                                          // Wait for a keystroke in the window

  return 0;
}

//____________________ FUNCTIONS ____________________
void analyse_sample(Mat image)
{
  // TODO: Se slutning i 5.2 og
  // http://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv
  // Derefter indsæt det udklippede i selve histogrammet

  // Setup a rectangle to define your region of interest
  Rect sample(1200, 1100, 200, 200);

  // Crop the full image to that image contained by the rectangle myROI
  // Note that this doesn't copy the data
  Mat croppedImage = image(sample);

  imshow( "Image - Cropped", croppedImage );
  Mat hist = draw_histogram(croppedImage);
  imshow( "Hist - Cropped", hist );
}

void median_filter(Mat src, Mat dst, int kernel_size)
{
  // TODO MAKE GENERIC!
  vector<float> neighborhood (9,0);
  int k = 0;
  float median;

  for (int v = 1; v < src.cols-1; v++) {
    for (int u = 1; u < src.rows-1; u++) {
      for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) {
          neighborhood[k]=static_cast<int>(src.at<uchar>(u+i,v+j));
          k++;
        }
      }

      /// Calculate median
      size_t size = neighborhood.size();
      sort(neighborhood.begin(), neighborhood.end());

      if (size  % 2 == 0) {
          median = (neighborhood[size / 2 - 1] + neighborhood[size / 2]) / 2;
      }
      else {
          median = neighborhood[size / 2];
      }

      dst.at<uchar>(u,v)=median;
      k=0;
    }
  }
}

Mat draw_histogram(Mat image)
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
  int bin_w = cvRound( (double) (hist_w)/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw the histogram
  for( int i = 0; i < histSize; i++ )
  {
    line( histImage, Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ) ,
                     Point( bin_w*(i), hist_h ),
                     Scalar( 0, 255, 50), 1, 8, 0  );
  }

  /// Return
  return histImage;
}

void resize_image(Mat& image, float scale)
{
  Size size(round(scale*image.cols),round(scale*image.rows));
  resize(image, image, size);
}

void dftshift(cv::Mat_<float>& magnitude)
{
   const int cx = magnitude.cols/2;
   const int cy = magnitude.rows/2;

   cv::Mat_<float> tmp;
   cv::Mat_<float> topLeft(magnitude, cv::Rect(0, 0, cx, cy));
   cv::Mat_<float> topRight(magnitude, cv::Rect(cx, 0, cx, cy));
   cv::Mat_<float> bottomLeft(magnitude, cv::Rect(0, cy, cx, cy));
   cv::Mat_<float> bottomRight(magnitude, cv::Rect(cx, cy, cx, cy));

   topLeft.copyTo(tmp);
   bottomRight.copyTo(topLeft);
   tmp.copyTo(bottomRight);

   topRight.copyTo(tmp);
   bottomLeft.copyTo(topRight);
   tmp.copyTo(bottomLeft);
}

Mat draw_magnitudeplot(Mat_<float> img)
{
   // A gray image
   Mat padded;                       //expand input image to optimal size

   //Pad the image with borders using copyMakeBorders. Use getOptimalDFTSize(A+B-1). See G&W page 251,252 and 263 and dft tutorial. (Typicly A+B-1 ~ 2A is used)
   int m = cv::getOptimalDFTSize(2*img.rows);
   int n = cv::getOptimalDFTSize(2*img.cols);

   copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, Scalar::all(0));

   //Copy the gray image into the first channel of a new 2-channel image of type Mat_<Vec2f>, e.g. using merge(), save it in img_dft
   //The second channel should be all zeros.
   cv::Mat_<float> imgs[] = {img.clone(), cv::Mat_<float>(img.rows, img.cols, 0.0f)};
   cv::Mat_<cv::Vec2f> img_dft;

   merge(imgs, 2, img_dft);

   // Compute DFT using img_dft as input
   dft(img_dft, img_dft);

   // Split img_dft, you can save result into imgs
   split(img_dft, imgs);                   // imgs[0] = Re(DFT(I), imgs[1] = Im(DFT(I))

   // Compute magnitude/phase (e.g. cartToPolar), use as input imgs
   cv::Mat_<float> magnitude, phase;
   cartToPolar(imgs[0], imgs[1], magnitude, phase, false);

   // Shift magnitude quadrants for viewability, use dftshift
   dftshift(magnitude);

   // Define Logarithm of magnitude and Output image for HPF
   cv::Mat_<float> magnitudel;
   cv::Mat_<float> imgout;

   log(magnitude, magnitudel);

   // Show
   cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
   cv::normalize(magnitudel, magnitudel, 0.0, 1.0, CV_MINMAX);
   cv::normalize(phase, phase, 0.0, 1.0, CV_MINMAX);
   return magnitudel;
}

void analyse_image(Mat image)
{
  Mat histogram = draw_histogram(image);
  Mat magnitudeplot = draw_magnitudeplot(image);

  //TODO move the picture


  resize_image(image, 0.25);
  imshow( "Source Image", image );
  moveWindow("Source Image", 0, 0);

  resize_image(histogram, 0.75);
  imshow( "histogram", histogram );
  moveWindow("histogram", image.cols/2, image.rows+25);

  resize_image(magnitudeplot, 0.25);
  imshow( "magnitudeplot", magnitudeplot );
  moveWindow("magnitudeplot", image.cols, 0);


}
