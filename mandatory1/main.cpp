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
Mat analyse_sample(Mat);
Mat restore_image(Mat,int);
string get_filepath(int);
void median_filter(Mat src, Mat dst);
void dftshift(Mat_<float>&);
void resize_image(Mat&, float);
void analyse_image(Mat);



//____________________ MAIN PROGRAM ____________________
int main( int argc, char** argv)
{
  //************ VARIABLES AND DATA ***************
  // Image1.png = 1, Image2.png = 2, Image3.png = 3,
  // Image4_1.png = 41, Image4_2.png = 42, Image5_optional.png = 5,

  int image_number    = 5;
  cout << "Enter image number: ";
  cin >> image_number;
  cout << "Calculating ..." << endl;

  Mat image_source    = imread( get_filepath( image_number ), CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_restored  = image_source.clone();

  // ************ ANALYSE IMAGE *******************
  Mat histogram = draw_histogram(image_source);
  Mat magnitudeplot = draw_magnitudeplot(image_source);
  Mat sample = analyse_sample(image_source);
  Mat histogram_s = draw_histogram(sample);

  Scalar mean,stddev;
  meanStdDev(sample,mean,stddev,cv::Mat());
  cout << "The mean is:\t\t " << mean[0]  << endl;
  cout << "The std variance is:\t " << stddev[0] << endl;
  // TODO Check formulars at: http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=meanstd#meanstddev

  // ************ MODIFY IMAGE ********************
  switch (image_number) {
    case 1:
      // contraharmonic filter måske?
      break;
    case 2:
      // Måske det skal filtreres flere gange?
      medianBlur(image_source, image_restored, 7); // TODO Write about the different kernel sizes

      for (int i = 0; i < 3; i++) {
        medianBlur(image_restored, image_restored, 7); // example 5.3 s. 327
      }
      //equalizeHist( image_restored, image_restored );
      break;
    case 3:
      // Uniform noise
      // Geometric- or arithmetic mean filter
      break;
    case 41:
      // Box noise on the magnitude plot - Notch box filter (or gaussian to avoid ringing)
      break;
    case 42:
      // Either bandpassfilter or notch filters
      break;
    case 5:
      // Random :D
      // Kan ikke lige se hvad det skulle være ;)
      break;
    default:
      break;
  }

  // ************ ANALYSE RESTORED ****************
  Mat histogram_r = draw_histogram(image_restored);
  Mat magnitudeplot_r = draw_magnitudeplot(image_restored);

  //************* DISPLAY IMAGES ******************
  rectangle(image_source, Point(1345,1195), Point(1455,1305), 0, 3); // image sample

  // TODO @Christian Hvis det ikke passer til den skærm, så lav en scalar variable
  // TODO du ganger på alle resizene.. Men lad de værdier der er er nu være xD


  resize_image(image_source, 0.25);
  imshow( "Source Image", image_source );
  moveWindow("Source Image", 0, 0);

  resize_image(histogram, 0.75);
  imshow( "histogram", histogram );
  moveWindow("histogram", image_source.cols/2, image_source.rows+25);

  resize_image(magnitudeplot, 0.25);
  imshow( "magnitudeplot", magnitudeplot );
  moveWindow("magnitudeplot", image_source.cols, 0);

  resize_image(image_restored, 0.25);
  imshow( "Restored Image", image_restored );
  moveWindow("Restored Image", image_source.cols*2.5, 0);

  resize_image(histogram_r, 0.75);
  imshow( "histogram (restored)", histogram_r );
  moveWindow("histogram (restored)", image_source.cols*3, image_source.rows+25);

  resize_image(magnitudeplot_r, 0.25);
  imshow( "magnitudeplot (restored)", magnitudeplot_r );
  moveWindow("magnitudeplot (restored)", image_source.cols*3.5, 0);

  resize_image(histogram_s, 0.75);
  imshow( "histogram (sample)", histogram_s );
  moveWindow("histogram (sample)", image_source.cols*1.75, image_source.rows+25);

  resize_image(sample, 0.75);
  imshow( "sample", sample );
  moveWindow("sample", image_source.cols*2.25-histogram_s.cols/2, image_source.rows+25);

  waitKey(0); // Wait for a keystroke in the window

  return 0;
}

//____________________ FUNCTIONS ____________________
Mat analyse_sample(Mat image)
{
  Rect sample(1350, 1200, 100, 100);

  Mat croppedImage = image(sample);
  return croppedImage;
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
  int hist_w = 522; int hist_h = 400;
  int bin_w = cvRound( (double) (hist_w)/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw the histogram
  for( int i = 0; i < histSize; i++ )
  {
    line( histImage, Point( bin_w*(i)+5, hist_h - cvRound(hist.at<float>(i)) ) ,
                     Point( bin_w*(i)+5, hist_h ),
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

string get_filepath(int file_num)
{
  string path_name="Could not find the path";
  switch (file_num) {
    case 1:
      path_name = "../Images/Image1.png";
      break;
    case 2:
      path_name = "../Images/Image2.png";
      break;
    case 3:
      path_name = "../Images/Image3.png";
      break;
    case 41:
      path_name = "../Images/Image4_1.png";
      break;
    case 42:
      path_name = "../Images/Image4_2.png";
      break;
    case 5:
      path_name = "../Images/Image5_optional.png";
      break;
    default:
      break;
  }

  return path_name;
}
