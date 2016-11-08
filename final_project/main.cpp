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

//______________ FUNCTION DECLARATIONS ________________
// See explanations of functions below the function, further down in the code.
void load_data(vector<Mat> &input, String &path, int type = 1);
vector<Mat> color_segmentation(vector<Mat> &input, int type);

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

        vector<Mat> blue_output = color_segmentation(input_sequence, BLUE);
        output_sequence = blue_output;
        }
        break;
    case COLOR_INPUT_HARD:
        {
        // LOAD DATA
        String color_path_hard("../sequences/marker_color_hard/*.png");
        load_data(input_sequence, color_path_hard, HSV);
        vector<Mat> blue_output = color_segmentation(input_sequence, BLUE);
        output_sequence = blue_output;

        }
        break;
    case LINE_INPUT:
        {
        // LOAD DATA
        String line_path("../sequences/marker_thinline/*.png");
        load_data(input_sequence, line_path, GRAY);

        }
        break;
    case LINE_INPUT_HARD:
        {
        // LOAD DATA
        String line_path_hard("../sequences/marker_thinline_hard/*.png");
        load_data(input_sequence, line_path_hard, GRAY);

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

  cout << "Test" << endl;
  // ------- Show results -----
  for(int i = 0; i < output_sequence.size(); i++)
  {
    imshow("Output Video", output_sequence[i]);
    if(waitKey(500) >= 0) break; // Increase x of waitKey(x) to slow down the video
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

//____________________ FUNCTIONS ___________________
// *** Color segmentation ***
// Example on syntax for function
vector<Mat> color_segmentation(vector<Mat> &input, int type)
{
  vector<Mat> output;

  int Sat_lower = 55;
  int Sat_upper = 200;
  int Val_lower = 55;
  int Val_upper = 200;
  int Hue_lower = 0;
  int Hue_upper = 255;

  if(type == RED){
    Hue_lower = 0;
    Hue_upper = 50;
  }
  else if(type == BLUE){
    Hue_lower = 100;
    Hue_upper = 120;
  }

  for(int i = 0; i < input.size(); i++)
  {
     output.push_back(input[i]);
     inRange(input[i], Scalar(Hue_lower, Sat_lower, Val_lower), Scalar(Hue_upper, Sat_upper, Val_upper), output[i]);
  }

  return output;
}
