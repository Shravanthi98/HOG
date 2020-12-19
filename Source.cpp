#include "HOG.h"
#include <fstream>
#include <vector>
#include "opencvtest.h"

int main(int argc, char* argv[])
{
	// To store the path of the image.
	std::string image_path;
	// User Input.
	image_path = argv[1];
	
	//Reading the input image in grayscale format as HOG works best for gray images.
	cv::Mat image = imread(image_path, IMREAD_GRAYSCALE);
	
	//Resizing the image to 128x64, i.e. the standard HOG window size.
	cv::Size win_size(64, 128);
	cv::resize(image, image, win_size);
	
	// To compute the gradient magnitudes and directions.
	cv::Mat magnitude;
	cv::Mat angle;
	computeMagAngle(image, magnitude, angle);
	
	// HOG Descriptor function call
	cv::Mat wHogFeature;
	computeHOG(magnitude, angle, wHogFeature);
	
	// Output Display based on user input.
	char *value = argv[2];
	int number = atoi(value);
	
	// Text display.
	if (number==1)
		std::cout << wHogFeature << endl;

	// Graphical display.
	else if (number == 2)
	{
		// Converting a matrix to a vector of descriptor values.
		std::vector<float> Vec(wHogFeature.begin<float>(), wHogFeature.end<float>());
		//Hog Visualization
		cv::Mat hog_image = get_hogdescriptor_visual_image(image, Vec,
			cv::Size(64, 128),
			cv::Size(8, 8),
			10,
			5);
		
		imshow("Hog Visualization", hog_image);
		waitKey(0);
	}

	// Both text and graphical display.
	else if (number == 3)
	{
		std::cout << wHogFeature << endl;
		std::vector<float> Vec(wHogFeature.begin<float>(), wHogFeature.end<float>());
		//Hog Visualization
		cv::Mat hog_image = get_hogdescriptor_visual_image(image, Vec,
			cv::Size(64, 128),
			cv::Size(8, 8),
			10,
			5);

		imshow("Hog Visualization", hog_image);
		waitKey(0);
	}
	
	return 0;
}
