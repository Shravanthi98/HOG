#include "HOG.h"
#include <fstream>
#include <vector>

int main(int argc, char* argv[])
{
	// To store the path of the image.
	std::string image_path;
	// User Input.
	image_path = argv[1];
	
	//Reading the input image in grayscale format as HOG works best for gray scale images.
	cv::Mat image = imread(image_path, IMREAD_GRAYSCALE);
	
	//Resizing the image to 128x64, i.e. the standard HOG window size.
	cv::Size win_size(64, 128);
	cv::resize(image, image, win_size);
	
	// To compute the gradient magnitudes and directions.
	cv::Mat magnitude;
	cv::Mat angle;
	computeMagAngle(image, magnitude, angle);
	
	// HOG Descriptor function call
	cv::Mat HogFeatures;
	computeHOG(magnitude, angle, HogFeatures);
	
	// Text display.
	std::cout << HogFeatures << endl;

	return 0;
}
