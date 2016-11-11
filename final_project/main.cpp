//______________ FINAL PROJECT ROVI | COMPUTER VISION ________________________
// Description
//
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

//_____________________ DEFINES _______________________
#define HSV         3
#define GRAY        2

#define RED         1
#define BLUE        2

#define COLOR_INPUT         1
#define COLOR_INPUT_HARD    2
#define LINE_INPUT          3
#define LINE_INPUT_HARD     4
#define CORNY_INPUT         5
#define CORNY_INPUT_HARD    6

#define PI                  3.14159265359

// _____________ GLOBAL VARIABLES ____________________
Mat src, src_threshold;
Mat dst, output_threshold;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 255;
int highThreshold;
int const max_highThreshold = 255;
int ratio = 3;
int kernel_size;
int const max_kernel_size = 21;
int low_hueThreshold;
int high_hueThreshold;
int low_satThreshold;
int high_satThreshold;
int low_valThreshold;
int high_valThreshold;
int const max_colorThreshold = 255;

char* window_name = "Output Sequence";

//______________ FUNCTION DECLARATIONS ________________
// See explanations of functions below the function, further down in the code.
void load_data(vector<Mat> &input, String &path, int type = 1);
vector<Mat> color_segmentation(vector<Mat> &input, int type);
vector<vector<vector<Point> > > find_circle_contours(vector<Mat> &input, int perimeter_thresh, int circle_thresh);
vector<vector<Point> > find_centers(vector<vector<vector<Point> > > &input_contours);
void CannyThreshold(int, void*);
void ColorThreshold(int, void*);

//____________________ MAIN PROGRAM ____________________
int main( int argc, char** argv)
{
  // color_input = 1, color_input_hard = 2, line_input = 3,
  // line_input_hard = 4, corny_input = 5, corny_input_hard = 6.

  int sequence_number;
  cout << "____________ Final Project ROVI - Computer vision ______________" << endl << endl;
  cout << "Choose a sequence of images..." << endl;
  cout << "----------------------------------------------------------------" << endl;
  cout << "Color_input = 1, Color_input_hard = 2, Line_input = 3," << endl;
  cout << "Line_input_hard = 4, Corny_input = 5, Corny_input_hard = 6." << endl;
  cout << "----------------------------------------------------------------" << endl;
  cout << "Enter sequence number: ";
  cin >> sequence_number;

  //_____________ VARIABLES AND DATA ______________
  vector<Mat> input_sequence;
  vector<Mat> output_sequence;

  // ________________ Object Recognition ____________________
  switch(sequence_number){
    case COLOR_INPUT:
        {
          //_________ LOAD DATA __________
          String color_path("../sequences/marker_color/*.png");
          load_data(input_sequence, color_path, HSV);

          // Segment the blue color in the images
          vector<Mat> blue_output = color_segmentation(input_sequence, BLUE);

          // Find contours that belong to circles
          vector<vector<vector<Point> > > circles = find_circle_contours(blue_output, 100, 0.7);

          // Find the center of the contours
          vector<vector<Point> > centers = find_centers(circles);

          // Set the output sequence and draw the centers on the images.
          output_sequence = input_sequence;

          for(int i = 0; i < output_sequence.size();i++)
          {
            for(int j = 0; j < centers[i].size(); j++)
            {
                circle(output_sequence[i], centers[i][j], 5, Scalar(255, 255, 255));
            }
          }
        }
        break;
    case COLOR_INPUT_HARD:
        {
          // LOAD DATA
          String color_path_hard("../sequences/marker_color_hard/*.png");
          load_data(input_sequence, color_path_hard, HSV);

          // Segment the blue color in the images
          vector<Mat> blue_output = color_segmentation(input_sequence, BLUE);

          // Find contours that belong to circles
          vector<vector<vector<Point> > > circles = find_circle_contours(blue_output, 100, 0.7);

          // Find the center of the contours
          vector<vector<Point> > centers = find_centers(circles);

          // Set the output sequence and draw the centers on the images.
          output_sequence = input_sequence;

          for(int i = 0; i < output_sequence.size();i++)
          {
            for(int j = 0; j < centers[i].size(); j++)
            {
                circle(output_sequence[i], centers[i][j], 5, Scalar(255, 255, 255));
            }
          }

        }
        break;
    case LINE_INPUT:
        {
        // LOAD DATA
        String line_path("../sequences/marker_thinline/*.png");
        load_data(input_sequence, line_path, GRAY);

        output_sequence = input_sequence;

        }
        break;
    case LINE_INPUT_HARD:
        {
        // LOAD DATA
        String line_path_hard("../sequences/marker_thinline_hard/*.png");
        load_data(input_sequence, line_path_hard, GRAY);

        output_sequence = input_sequence;
        }
        break;
    case CORNY_INPUT:
        {
        // LOAD DATA
        String corny_path("../sequences/marker_corny/*.png");
        load_data(input_sequence, corny_path, GRAY);

        }
        break;
    case CORNY_INPUT_HARD:
        {
        // LOAD DATA
        String corny_path_hard("../sequences/marker_corny_hard/*.png");
        load_data(input_sequence, corny_path_hard, GRAY);

        }
        break;
  }

  // --------- Output the video --------
  /*VideoWriter output_vid;
  output_vid.open("./output_vid.avi", CV_FOURCC('M','J','P','G'), 5, Size(1024, 768));
  for(int i = 0; i < output_sequence.size(); i++)
  {
    output_vid.write(output_sequence[i]);
  }
  output_vid.release();*/

  namedWindow( window_name, CV_WINDOW_NORMAL );

  // ------- Show results -----
  for(int i = 0; i < output_sequence.size(); i++)
  {
    //src_threshold = output_sequence[i];

    //createTrackbar( "Max Threshold:", window_name, &highThreshold, max_highThreshold, CannyThreshold );
    //createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    //createTrackbar( "Smooth amount:", window_name, &kernel_size, max_kernel_size, CannyThreshold );
    //CannyThreshold(0,0);

  /*createTrackbar( "Hue_min:", window_name, &low_hueThreshold, max_colorThreshold, ColorThreshold );
    createTrackbar( "Hue_max:", window_name, &high_hueThreshold, max_colorThreshold, ColorThreshold );
    createTrackbar( "Sat_min:", window_name, &low_satThreshold, max_colorThreshold, ColorThreshold );
    createTrackbar( "Sat_max:", window_name, &high_satThreshold, max_colorThreshold, ColorThreshold );
    createTrackbar( "Val_min:", window_name, &low_valThreshold, max_colorThreshold, ColorThreshold );
    createTrackbar( "Val_max:", window_name, &high_valThreshold, max_colorThreshold, ColorThreshold );
    ColorThreshold(0,0);*/

    if(i == output_sequence.size() - 1){
      i = 0;
    }
    imshow(window_name, output_sequence[i]);
    if(waitKey(250) >= 0) break; // Increase x of waitKey(x) to slow down the video
  }

  waitKey(0);

  return 0;
}

//____________________ FUNCTIONS ___________________
// *** Load data ***
// Loads the image data into a vector of Mat's, and converts to either gray or HSV
void load_data(vector<Mat> &input, String &path, int type)
{
  vector<String> fn;
  glob(path, fn, true); // true = Recursive

  for (size_t k = 0; k < fn.size(); k++)
  {
       Mat im_in = imread(fn[k]);
       Mat im_out;

       if(type == 2){
         cvtColor(im_in, im_out, CV_BGR2GRAY);
       }
       else if(type == 3){
         cvtColor(im_in, im_out, CV_BGR2HSV);
       }
       else{
         im_out = im_in;
       }

       input.push_back(im_out);
  }
}

// *** Color segmentation ***
// Example on syntax for function
vector<Mat> color_segmentation(vector<Mat> &input, int type)
{
  vector<Mat> output;

  int Sat_lower = 30;
  int Sat_upper = 255;
  int Val_lower = 25;
  int Val_upper = 230;
  int Hue_lower = 0;
  int Hue_upper = 255;

  if(type == RED){
    Hue_lower = 0;
    Hue_upper = 50;
  }
  else if(type == BLUE){
    Hue_lower = 110;
    Hue_upper = 120;
  }

  for(int i = 0; i < input.size(); i++)
  {
     output.push_back(input[i]);
     inRange(input[i], Scalar(Hue_lower, Sat_lower, Val_lower), Scalar(Hue_upper, Sat_upper, Val_upper), output[i]);
  }

  return output;
}

// *** Find Circle Contours ***
// Example on syntax for function
vector<vector<vector<Point> > > find_circle_contours(vector<Mat> &input, int perimeter_thresh, int circle_thresh)
{
  vector<vector<vector<Point> > > result_contours;

  for(int i = 0; i < input.size(); i++)
  {
    vector<vector<Point> > contours;
    vector<vector<Point> > circle_contours;
    vector<Vec4i> hierarchy;

    /// Find contours
    findContours( input[i], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Extract correct contours
    for( int j = 0; j < contours.size(); j++ )
    {
      // Calculate parameters
      double area             = abs(contourArea(contours[j], true));
      double perimeter        = arcLength(contours[j], 1);
      double circle_constant  = (4 * PI * area) / (perimeter*perimeter);

      if(perimeter > 100 && circle_constant > 0.7)
      {
        circle_contours.push_back(contours[j]);
      }
    }

    result_contours.push_back(circle_contours);
  }

  return result_contours;

}

// *** Find Centers ***
// Example on syntax for function
vector<vector<Point> > find_centers(vector<vector<vector<Point> > > &input_contours)
{
  vector<vector<Point> > circle_centers(input_contours.size());

  for(int i = 0; i < input_contours.size(); i++) // For every frame
  {
    for(int j = 0; j < input_contours[i].size(); j++) // For every contour in frame
    {
      Moments circle_moments = moments(input_contours[i][j], false);
      int center_u = floor(circle_moments.m10/circle_moments.m00);
      int center_v = floor(circle_moments.m01/circle_moments.m00);

      circle_centers[i].push_back(Point(center_u, center_v));
    }
  }

  return circle_centers;
}

// *** Color Threshold ***
// Example on syntax for function
void ColorThreshold(int, void*)
{
  inRange(src_threshold, Scalar(low_hueThreshold, low_satThreshold, low_valThreshold), Scalar(high_hueThreshold, high_satThreshold, high_valThreshold), output_threshold);
  dst = Scalar::all(0);

  src_threshold.copyTo( dst, output_threshold);
  imshow( window_name, dst );
}

// *** Canny Threshold ***
// Example on syntax for function
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  if (kernel_size > 0)
    blur( src_threshold, output_threshold, Size(kernel_size,kernel_size) );
  else
    blur( src_threshold, output_threshold, Size(1,1) );


  /// Canny detector
  Canny( output_threshold, output_threshold, lowThreshold, highThreshold, 3 );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src_threshold.copyTo( dst, output_threshold);
  imshow( window_name, dst );
}
