//Lecture 1 Computer Vision
//
// Adjusting colour and other stuff

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat_<Vec3b> source;
    Mat grey;
    Mat canny;
    source = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! source.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // Grayscale
    cvtColor(source, grey, CV_BGR2GRAY);
    

    // Creating the box
    for(int i = 100; i < 220; i++)
    {
      for(int j = 350; j < 440; j++)
      {
          source(i,j)[0] = 0;
          source(i,j)[1] = 255;
          source(i,j)[2] = 0;
      }
    }

    // Display image
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", grey );                   // Show our image inside it.
    imshow( "Source", source, grey);

    // Printing info
    printf("Source | Type: %i \t depth: %i \t channels: %i \n", source.type(), source.depth(), source.channels());
    printf("Grey | Type: %i \t depth: %i \t channels: %i \n", grey.type(), grey.depth(), grey.channels());

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
