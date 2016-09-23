/*
 *  ex3_template.cpp
 *  Exercise 3 - The Frequency domain and filtering
 *
 *  Created by Stefan-Daniel Suvei on 19/09/16.
 *  Copyright 2016 SDU Robotics. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>

using namespace cv;

void run(const std::string& filename, bool highpass);
void dftshift(cv::Mat_<float>& magnitude);

int main(int argc, char **argv) {
   // EXERCISE 1
   run("lena.bmp", false);
   run("lena_face.bmp", false);
   run("lena_hair.bmp", false);
   run("lena_hat.bmp", false);

   // EXERCISE 2
   run("lena.bmp", true);

   return 0;
}

void run(const std::string& filename, bool highpass) {
   // A gray image
   cv::Mat_<float> img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
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

   if(highpass) {
      // High-pass filter: remove the low frequency parts in the middle of the spectrum, you can use Rect()
      rectangle(magnitude,
                Point(magnitude.cols/2 - 25, magnitude.rows/2 - 25),
                Point(magnitude.cols/2 + 25, magnitude.rows/2 + 25),
                0,
                CV_FILLED
              );

      // Take logarithm of modified magnitude (log()), save result into magnitudel
      log(magnitude, magnitudel);

      rectangle(magnitudel,
                Point(magnitudel.cols/2 - 25, magnitudel.rows/2 - 25),
                Point(magnitudel.cols/2 + 25, magnitudel.rows/2 + 25),
                0,
                CV_FILLED
              );

      // Shift back magnitude quadrants of the spectrum, use dftshift();
      dftshift(magnitude);

      // Compute complex DFT output from magnitude/phase (polarToCart()), store result in imgs
      polarToCart(magnitude,phase,imgs[0],imgs[1],false);

      // Merge DFT (imgs) into one image
      merge(imgs, 2, img_dft);

      //Restore, use dft with DFT_INVERSE flag, save result in imgout
      dft(img_dft, imgout, cv::DFT_INVERSE + cv::DFT_SCALE + cv::DFT_REAL_OUTPUT);

   } else {
     log(magnitude, magnitudel);
   }

   // Show
   cv::normalize(img, img, 0.0, 1.0, CV_MINMAX);
   cv::normalize(magnitudel, magnitudel, 0.0, 1.0, CV_MINMAX);
   cv::normalize(phase, phase, 0.0, 1.0, CV_MINMAX);
   cv::imshow("Input", img);
   cv::imshow("Magnitude", magnitudel);
   if(highpass) {
      cv::normalize(imgout, imgout, 0.0, 1.0, CV_MINMAX);
      cv::imshow("Output", imgout);
   }
   cv::waitKey();
}

void dftshift(cv::Mat_<float>& magnitude) {
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
