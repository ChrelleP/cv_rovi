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

//______________ FUNCTION DECLARATIONS ________________
// See explanations of functions below the function, further down in the code.
Mat draw_histogram(Mat);
Mat draw_magnitudeplot(Mat_<float>);
Mat analyse_sample(Mat);
Mat restore_image(Mat,int);
string get_filepath(int);
void adaptive_median_filter(Mat src, Mat dst, int, int);
void Contraharmonic_filter(Mat src, Mat dst, int, float);
void midpoint_filter(Mat src, Mat dst, int);
void dftshift(Mat_<float>&);
void resize_image(Mat&, float);
void analyse_sp(Mat&);
void notch_highpass_butterworth(Mat& image, vector<Point>& targets, float cut_off, int order);
void intensityIncrease(Mat dst, double alhpa, int beta, bool saturateCast);
void constrast_stretch(Mat src, Mat dst);
void sharpen(Mat &image, float k, int kernel);

//____________________ MAIN PROGRAM ____________________
int main( int argc, char** argv)
{
  //_____________ VARIABLES AND DATA ______________
  // Image1.png = 1, Image2.png = 2, Image3.png = 3,
  // Image4_1.png = 41, Image4_2.png = 42, Image5_optional.png = 5,

  int image_number    = 5;
  cout << "Image1.png = 1, Image2.png = 2, Image3.png = 3," << endl;
  cout << "Image4_1.png = 41, Image4_2.png = 42" << endl;
  cout << "Enter image number: ";
  cin >> image_number;

  Mat image_source    = imread( get_filepath( image_number ), CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_restored  = image_source.clone();

  // ______________ ANALYSE IMAGE ______________
  Mat histogram       = draw_histogram(image_source);
  Mat magnitudeplot   = draw_magnitudeplot(image_source);
  Mat sample          = analyse_sample(image_source);
  Mat histogram_s     = draw_histogram(sample);
  analyse_sp(image_source);

  Scalar mean,stddev;
  meanStdDev(sample,mean,stddev,cv::Mat());
  cout << "The mean is:\t\t " << mean[0]  << endl;
  cout << "The std variance is:\t " << stddev[0] << endl;
  cout << "Calculating ..." << endl;

  double min;

  // ______________ RESTORE IMAGE ______________
  // This switch case restores the image depending on the image number chosen
  // when executing the program.
  // The cases represent each image, and the the alternate versions (different
  // filters mentioned in the report) are commented out.
  switch (image_number) {
    case 1: // IMAGE 1 - Pepper noise
      {
      Contraharmonic_filter(image_source, image_restored, 3, 1);
      constrast_stretch(image_restored, image_restored);

      // Alternate version - Using Median blur to elimnate some of the remaining
      // noise.
      /*
      Mat temp = image_restored.clone();
      medianBlur(temp, image_restored, 7);
      */

      // Alternate Version - Using bilateralFilter to remove some of the remaining
      // noise.
      /*
      temp = image_restored.clone();
      Mat new_restored;
      bilateralFilter(temp, new_restored, 9, 30, 30);
      */
      }
      break;
    case 2: // IMAGE 2 - Large amounts of Salt and small amount of pepper noise.
      // Adaptive Median Filter - The intended version
      if (true) {
        // Use an Adaptive Median Filter
        adaptive_median_filter(image_source, image_restored, 3, 11);

        // Stretch Contrast to improve image quality.
        constrast_stretch(image_restored, image_restored);

        // Count the amount of white and black pixels
        int count_w = 0;
        int count_b = 0;
        for (int y = 0; y < image_restored.rows; y++) {
          for (int x = 0; x < image_restored.cols; x++) {
            if (image_restored.at<uchar>(y,x) == 255){
              count_w++;
            } else if (image_restored.at<uchar>(y,x) == 0) {
              count_b++;
            }
          }
        }

        //Output the amount of black and white pixels.
        cout << "WHITE: " << count_w << endl;
        cout << "BLACK: " << count_b << endl;
      }

      else // Alternate Version - Using multiple passes of median filter.
      {
        // Single pass of Median (Should always be used atleast once).
        medianBlur(image_source, image_restored, 7);

        // Additional passes of median (depending on i).
        for (int i = 0; i < 3; i++) {
          medianBlur(image_restored, image_restored, 7);  // example 5.3 s. 327
        }

        // Contrast stretching to improve image quality
        constrast_stretch(image_restored, image_restored);
      }

      break;
    case 3: // IMAGE 3 - Gaussian noise
      {
      // Use Bilateral Filter to remove the noise
      bilateralFilter(image_source, image_restored, 9, 50, 50);

      // Alternate Version, using midpoint filter and sharpening.
      /*
      midpoint_filter(image_source, image_restored, 5);
      Mat temp = image_restored.clone();
      sharpen(image_restored, 1, 5);
      */

      // Stretch constrast to improve the image
      constrast_stretch(image_restored, image_restored);
      }
      break;
    case 41: // IMAGE 4_1 - Noise in the frequency domain.
      {
        // Specify the coordinates of the targeted noise in the frequency domain.
        Point target_1(206, 200);
        Point target_2(622, -604);
        vector<Point> target_freqs;
        target_freqs.push_back(target_1);
        target_freqs.push_back(target_2);

        // Use a butterworth Notch filter with the specified frequencies.
        notch_highpass_butterworth(image_restored, target_freqs, 20, 3);
      }
      break;
    case 42: // IMAGE 4_2 - Noise in frequency domain.
      {
        // Specify the coordinates of the targeted noise in the frequency domain.
        Point target_1(820, 0);
        Point target_2(590, -570);
        Point target_3(0, 810);
        Point target_4(590, 570);
        vector<Point> target_freqs;
        target_freqs.push_back(target_1);
        target_freqs.push_back(target_2);
        target_freqs.push_back(target_3);
        target_freqs.push_back(target_4);

        // Use a butterworth Notch filter with specified frequencies.
        notch_highpass_butterworth(image_restored, target_freqs, 30, 3);
      }
      break;
    case 5: // IMAGE 5 - Motion blur?
      // We believe it is motion blur, but we have not chosen to do the image restoration
      break;
    default:
      break;
  }

  // ______________ ANALYSE RESTORED ______________
  Mat histogram_r = draw_histogram(image_restored);
  Mat magnitudeplot_r = draw_magnitudeplot(image_restored);

  //______________ DISPLAY IMAGES ______________
  // It should be noted that all these images are saved in the folder "Image Results"
  // And are overwritten whenever the program is executed using a new image.

  rectangle(image_source, Point(1345,1195), Point(1455,1305), 0, 4); // image sample

  // Source image
  imwrite( "../image_results/source_image.jpg", image_source );
  resize_image(image_source, 0.25);
  imshow( "Source Image", image_source );
  moveWindow("Source Image", 0, 0);

  // Histogram of source image
  imwrite( "../image_results/histogram.jpg", histogram );
  resize_image(histogram, 0.75);
  imshow( "histogram", histogram );
  moveWindow("histogram", image_source.cols/2, image_source.rows+25);

  // Magnitudeplot of source image
  imwrite( "../image_results/magnitudeplot.jpg", magnitudeplot * 255 );
  resize_image(magnitudeplot, 0.25);
  imshow( "magnitudeplot", magnitudeplot );
  moveWindow("magnitudeplot", image_source.cols, 0);

  // The restored image
  imwrite( "../image_results/image_restored.jpg", image_restored );
  resize_image(image_restored, 0.4);
  imshow( "Restored Image", image_restored );
  moveWindow("Restored Image", image_source.cols*2.5, 0);

  // Histogram of the restored image
  imwrite( "../image_results/histogram_r.jpg", histogram_r );
  resize_image(histogram_r, 0.75);
  imshow( "histogram (restored)", histogram_r );
  moveWindow("histogram (restored)", image_source.cols*3, image_source.rows+25);

  // Magnitudeplot of the restored image
  imwrite( "../image_results/magnitudeplot_r.jpg", magnitudeplot_r * 255);
  resize_image(magnitudeplot_r, 0.25);
  imshow( "magnitudeplot (restored)", magnitudeplot_r );
  moveWindow("magnitudeplot (restored)", image_source.cols*3.5, 0);

  // Histogram of the sample
  imwrite( "../image_results/histogram_s.jpg", histogram_s );
  resize_image(histogram_s, 0.75);
  imshow( "histogram (sample)", histogram_s );
  moveWindow("histogram (sample)", image_source.cols*1.75, image_source.rows+25);

  // The sample image
  imwrite( "../image_results/sample.jpg", sample );
  resize_image(sample, 0.75);
  imshow( "sample", sample );
  moveWindow("sample", image_source.cols*2.25-histogram_s.cols/2, image_source.rows+25);

  waitKey(0); // Wait for a keystroke in the window

  return 0;
}

//____________________ FUNCTIONS ____________________
// *** Analyse Sample ***
// Retrieves a sample from the image, with a specific location.
Mat analyse_sample(Mat image)
{
  Rect sample(1350, 1200, 100, 100);

  Mat croppedImage = image(sample);
  return croppedImage;
}

// *** Contrast Strecthing ***
// Non generic stretching function, that moves the lowest value to zero and
// strecthes the rest of the pixels to maximum.
void constrast_stretch(Mat src, Mat dst)
{
  // http://what-when-how.com/embedded-image-processing-on-the-tms320c6000-dsp/contrast-stretching-image-processing/
  // http://www.nrcan.gc.ca/earth-sciences/geomatics/satellite-imagery-air-photos/satellite-imagery-products/educational-resources/9389
  double MP = 255;
  double min = 0, max = 0;
  minMaxLoc(src,&min,&max);
  cout << "Minimum intensity: " << min << endl;
  cout << "Maximum intensity: " << max << endl;
  double R = max-min;
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      dst.at<uchar>(y,x) = round( ((src.at<uchar>(y,x)-min) / R) * MP );
    }
  }
}

// *** Contraharmonic filter ***
// A Contraharmonic filter based on theory from the book
void Contraharmonic_filter(Mat src, Mat dst, int kernel_size, float Q)
{
    // Create temporary image
    Mat image_tmp;
    src.copyTo(image_tmp);

    // Make border
    int top = (int) (0.05*image_tmp.rows);  int bottom = (int) (0.05*image_tmp.rows);
    int left = (int) (0.05*image_tmp.cols); int right = (int) (0.05*image_tmp.cols);
    copyMakeBorder( src, image_tmp, top, bottom, left, right, BORDER_CONSTANT, 0);

    // Adjust kernel size for calculations
    kernel_size=kernel_size/2;

    // Calculate and apply contraharmonic filter
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
          double denominator=0,numerator=0;
          for(int s = -kernel_size; s <= kernel_size; s++){
            for(int t = -kernel_size; t <= kernel_size; t++){
                numerator += pow(image_tmp.at<uchar>(y+s+top,x+t+left),Q+1);
                denominator += pow(image_tmp.at<uchar>(y+s+top,x+t+left),Q);
            }
          }
       dst.at<uchar>(y,x) = numerator/denominator;
      }
    }
}

// *** Midpoint Filter ***
// Midpoint filter based on theory from the book.
void midpoint_filter(Mat src, Mat dst, int kernel_size)
{
    // Create temporary image
    Mat image_tmp;
    src.copyTo(image_tmp);

    // Make border
    int top = (int) (0.05*image_tmp.rows);  int bottom = (int) (0.05*image_tmp.rows);
    int left = (int) (0.05*image_tmp.cols); int right = (int) (0.05*image_tmp.cols);
    copyMakeBorder( src, image_tmp, top, bottom, left, right, BORDER_CONSTANT, 0);

    // Correct kernel size for calculations
    kernel_size=kernel_size/2;

    // Calculate and apply midpoint filter
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
          int max_value = 0;
          int min_value = 255;
          for(int s = -kernel_size; s <= kernel_size; s++){
            for(int t = -kernel_size; t <= kernel_size; t++){
                int value = image_tmp.at<uchar>(y+s+top,x+t+left);
                max_value = max(value, max_value);
                min_value = min(value, min_value);
            }
          }
       dst.at<uchar>(y,x) = floor((max_value + min_value)/2);
      }
    }
}

// *** Adaptive Median Filter ***
// Adaptive median filter based on theory from the book
void adaptive_median_filter(Mat src, Mat dst, int kernel_size_orig, const int max_kernel_size)
{
  Mat image_tmp;
  src.copyTo(image_tmp);
  int top = (int) (0.1*image_tmp.rows);  int bottom = (int) (0.1*image_tmp.rows);
  int left = (int) (0.1*image_tmp.cols); int right = (int) (0.1*image_tmp.cols);
  copyMakeBorder( src, image_tmp, top, bottom, left, right, BORDER_CONSTANT, 127);
  int z_min, z_max, z_xy, z_med;
  int kernel_size = kernel_size_orig;
  float A1, A2, B1, B2;

  for (int y = 0; y < src.rows; y++) {
    cout << y << " of " << src.rows << endl;
    for (int x = 0; x < src.cols; x++) {
      while(true){
        vector<float> neighborhood;
        for (int s = -kernel_size/2; s <= kernel_size/2; s++) {
          for (int t = -kernel_size/2; t <= kernel_size/2; t++) {
            neighborhood.push_back(image_tmp.at<uchar>(y+s+top,x+t+left));
          }
        }

        /// Calculate median
        size_t size = neighborhood.size();
        sort(neighborhood.begin(), neighborhood.end());

        z_max = neighborhood[0];
        z_min = neighborhood[size];
        z_xy = image_tmp.at<uchar>(y+top,x+left);

        if (size  % 2 == 0) {
            z_med = (neighborhood[size / 2 - 1] + neighborhood[size / 2]) / 2;
        }
        else {
            z_med = neighborhood[size / 2];
        }

        A1 = z_med - z_min;
        A2 = z_med - z_max;

        if (A1 > 0 && A2 < 0) {

          B1 = z_xy - z_min;
          B2 = z_xy - z_max;

          if (B1 > 0 && B2 < 0) {
            dst.at<uchar>(y,x)=z_xy;
            kernel_size = kernel_size_orig;
            break;
          }
          else
          {
            dst.at<uchar>(y,x)=z_med;
            kernel_size = kernel_size_orig;
            break;
          }

        }
        else if (kernel_size <= max_kernel_size){
          kernel_size+=2;
        }
        else{
          dst.at<uchar>(y,x)=z_med;
          kernel_size = kernel_size_orig;
          break;
        }
      }
    }
  }
}

// *** Draw Histogram ***
// Draws the histogram of the input image.
Mat draw_histogram(Mat image)
{
  //______________ MAKING HISTOGRAM ______________
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

  // Return
  return histImage;
}

// *** Resize Image ***
// Resizes the image with a specified scale.
void resize_image(Mat& image, float scale)
{
  Size size(round(scale*image.cols),round(scale*image.rows));
  resize(image, image, size);
}

// *** DFT shift ***
// Shifts the magnitude plot that OpenCV generates, to a magnitude plot with
// the center correctly placed in the middle of the image.
// Courtesy of: Stefan-Daniel Suvei
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

// *** Draw Magnitudeplot ***
// Draws the magnitudeplot of a gray image
Mat draw_magnitudeplot(Mat_<float> img)
{
   // A gray image
   Mat padded;

   //expand input image to optimal size
   //Pad the image with borders using copyMakeBorders. Use getOptimalDFTSize(A+B-1).
   int m = getOptimalDFTSize(2*img.rows);
   int n = getOptimalDFTSize(2*img.cols);

   copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

   //Copy the gray image into the first channel of a new 2-channel image of type Mat_<Vec2f>, e.g. using merge(), save it in img_dft
   //The second channel should be all zeros.
   Mat_<float> imgs[] = {img.clone(), cv::Mat_<float>(img.rows, img.cols, 0.0f)};
   Mat_<cv::Vec2f> img_dft;

   merge(imgs, 2, img_dft);

   // Compute DFT using img_dft as input
   dft(img_dft, img_dft);

   // Split img_dft, you can save result into imgs
   split(img_dft, imgs);                   // imgs[0] = Re(DFT(I), imgs[1] = Im(DFT(I))

   // Compute magnitude/phase (e.g. cartToPolar), use as input imgs
   Mat_<float> magnitude, phase;
   cartToPolar(imgs[0], imgs[1], magnitude, phase, false);

   // Shift magnitude quadrants for viewability, use dftshift
   dftshift(magnitude);

   // Define Logarithm of magnitude and Output image for HPF
   Mat_<float> magnitudel;
   Mat_<float> imgout;

   log(magnitude, magnitudel);

   // Normalize for better visualization
   normalize(img, img, 0.0, 1.0, CV_MINMAX);
   normalize(magnitudel, magnitudel, 0.0, 1.0, CV_MINMAX);
   normalize(phase, phase, 0.0, 1.0, CV_MINMAX);

   return magnitudel;
}

// *** Analyse Salt and Pepper noise ***
// Analyse the amount of salt and pepper noises in an image.
// Returns the percentages of white and black pixels in the image.
// This method can not dinstinguish an intended black or white pixel from one
// that is a result of impulse noise.
void analyse_sp(Mat& image)
{
  float total_pixels = image.cols * image.rows;
  float salt        = 0;
  float pepper      = 0;
  float p_salt    = 0;
  float p_pepper  = 0;

  for(int i = 0; i < image.cols; i++){
    for(int j = 0; j < image.rows; j++){
      int current_pixel = image.at<uchar>(i, j);

      if(current_pixel == 0)
        pepper ++;
      if(current_pixel == 255)
        salt ++;
    }
  }

  if(salt > 0)
    p_salt = salt / total_pixels;
  if(pepper > 0)
    p_pepper = pepper / total_pixels;

  printf("P of salt: %f \t P of pepper: %f \n", p_salt, p_pepper);
}

// *** Butterworth Notch Filter ***
// A butterworth notch filter based on theory from the book.
void notch_highpass_butterworth(Mat& image, vector<Point>& targets, float cut_off, int order)
{
  // Pad image borders
  int padded_rows = getOptimalDFTSize(2*image.rows);
  int padded_cols = getOptimalDFTSize(2*image.cols);

  int imgCols = image.cols;
  int imgRows = image.rows;

  copyMakeBorder(image, image, 0, padded_rows-image.rows ,0 ,padded_cols-image.cols, BORDER_CONSTANT, Scalar(0));

  // Transform image to frequency domain
  Mat_<float> image_parts[] = {image.clone(), Mat_<float>(image.rows, image.cols, 0.0f)};
  Mat_<Vec2f> image_dft;

  merge(image_parts, 2, image_dft);
  dft(image_dft, image_dft);

  split(image_dft, image_parts);

  Mat_<float> magnitude, phase;
  cartToPolar(image_parts[0], image_parts[1], magnitude, phase, false);

  dftshift(magnitude);

  // Create notch filters
  Mat_<float> filter_notch = magnitude.clone();
  Mat_<float> filter_notch_values = magnitude.clone();

  int u_range = magnitude.rows;
  int v_range = magnitude.cols;

  for(int u = 0; u < u_range; u++){
    for(int v = 0; v < v_range; v++){
      // Variables for calculations
      float H_NR = 1;
      float H_NR_first;
      float H_NR_second;
      float H_NP;

      for(int k = 0; k < targets.size(); k++){
        float D_pos = sqrt( powf(u - u_range/2 - targets[k].x, 2) + powf(v - v_range/2 - targets[k].y, 2));
        float D_neg = sqrt( powf(u - u_range/2 + targets[k].x, 2) + powf(v - v_range/2 + targets[k].y, 2));

        H_NR_first  = 1 / (1 + powf(cut_off / D_pos, 2*order) );
        H_NR_second = 1 / (1 + powf(cut_off / D_neg, 2*order) );

        H_NR *= H_NR_first * H_NR_second;
      }

      H_NP = H_NR;

      // Multiply the filter with the magnitude plot
      filter_notch.at<float>(u,v) = magnitude.at<float>(u,v) * H_NP;
      filter_notch_values.at<float>(u,v) = H_NP;
    }
  }

  // Visualize the filter, to see if the notches are placed correctly
  normalize(filter_notch_values, filter_notch_values, 0.0, 1.0, CV_MINMAX);
  resize_image(filter_notch_values, 0.20);
  imshow("magnitude", filter_notch_values);

  // Inverse transform the image back to the spatial domain
  // Shift back quadrants of the spectrum
  dftshift(filter_notch);

  // Compute complex DFT output from magnitude/phase
  polarToCart(filter_notch, phase, image_parts[0], image_parts[1]);

  // Merge DFT into one image and restore
  Mat_<float> imgout;
  cv::merge(image_parts, 2, image_dft);
  cv::dft(image_dft, imgout, cv::DFT_INVERSE + cv::DFT_SCALE + cv::DFT_REAL_OUTPUT);

  //Cut away the borders
  imgout = imgout(cv::Rect(0,0,imgCols,imgRows));
  image = imgout.clone();
}

// *** Unsharp Masking ***
// Sharpen the image using unsharp masking
void sharpen(Mat &image, float k, int kernel)
{
  Mat blurred = image.clone();
  float amount = k;

  // Get a blurred version of the picture
  GaussianBlur(blurred, blurred, Size(kernel, kernel), 0, 0);

  // Following the formular Sharpened = Original + ( Original - Blurred ) * Amount
  // Ref: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
  addWeighted(image, 1 + amount, blurred, -amount, 0, image);

}

// *** Get filepath ***
// Used to get the filepath where the images are located.
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
