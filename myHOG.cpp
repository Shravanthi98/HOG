// Include all the necessary headers and libraries.
#include "HOG.h"
#include <cmath>
#include <array>

// Computing the gradient magnitudes and directions.
void computeMagAngle(cv::InputArray source, cv::Mat& mag, cv::Mat& ang)
{
	// Getting the image pixel values.
	cv::Mat image = source.getMat();
	// Normalizing the image.
	image.convertTo(image, CV_32F, 1 / 255.0);
	
	//** Calculate gradients using Sobel operator.
	cv::Mat gradx, grady;
	Sobel(image, gradx, CV_32F, 1, 0, 1);
	Sobel(image, grady, CV_32F, 0, 1, 1);

	//** Calculate gradient magnitude and direction.
	cartToPolar(gradx, grady, mag, ang, 1);
	// Make the gradient directions unsigned, i.e. between 0 to 180 degrees.
	cv::subtract(ang, 180.0f, ang, (ang > 180.0f));
}

// Constrast Normalization over the blocks.
// L2-Hys Normalization = L2-norm + Clipping + L2-norm.
void contrast_normalize(cv::Mat &final_feat_vec)
{
	// To hold the temporary normalization results.
	cv::Mat norm_temp(36, 1, CV_32F);
	float temp1, temp2;

	// L2 Normalization.
	for (int iter4 = 0; iter4 < 3780; iter4 = iter4 + 36)
	{
		final_feat_vec(cv::Range(iter4, iter4 + 36), cv::Range(0, 1)).copyTo(norm_temp);
		temp1 = norm_temp.dot(norm_temp);
		temp2 = sqrt(temp1);
		norm_temp = (temp2 == 0) ? (norm_temp * 0) : (norm_temp * (1 / temp2));

		// Clipping with max = 0.2 using cv::threshold function.
		cv::threshold(norm_temp, norm_temp, 0.2, 0.2, cv::THRESH_TRUNC);

		// Re-normalizing it using L2-norm.
		temp1 = norm_temp.dot(norm_temp);
		temp2 = sqrt(temp1);
		norm_temp = (temp2 == 0) ? (norm_temp * 0) : (norm_temp * (1 / temp2));
		norm_temp.copyTo(final_feat_vec(cv::Range(iter4, iter4 + 36), cv::Range(0, 1)));
	}

}

// Block Normalization with a block size = (16, 16), block stride = (8,8).
void block_normalize(cv::Mat temp_fv[16][8], cv::Mat& final_fv_temp)
{
	// Concatenating 4 Histograms, i.e. 9(per histogram)*4 = 36.
	// This happens 15 times along the rows and 7 times along the columns.
	for (int rows = 0; rows < 15; rows++)
	{
		for (int cols = 0; cols < 7; cols++)
		{
			final_fv_temp.push_back(temp_fv[rows][cols]);
			final_fv_temp.push_back(temp_fv[rows][cols + 1]);
			final_fv_temp.push_back(temp_fv[rows + 1][cols]);
			final_fv_temp.push_back(temp_fv[rows + 1][cols + 1]);
		}
	}
}

// HOG Descriptor Main function.
void computeHOG(cv::InputArray magnitude, cv::InputArray angle, cv::Mat& final_fv)
{
	// Magnitude and Angle values
	const cv::Mat magMat = magnitude.getMat();
	const cv::Mat angMat = angle.getMat();

	// HOG Parameters
	const cv::Size win_size(64, 128);
	const cv::Size block_size(16, 16);
	const cv::Size block_stride(8, 8);
	const cv::Size cell_size(8, 8);

	// Validate the magnitude and angle matrix dimensions.
	// magMat, angMat = 128x64.
	if (magMat.rows != angMat.rows || magMat.cols != angMat.cols)
	{
		return;
	}

	// 8x8 Magnitude and Angle matrices for each cell.
	cv::Mat mag(cell_size.width, cell_size.height, CV_32F);
	cv::Mat ang(cell_size.width, cell_size.height, CV_32F);

	// Feature vector of length, 9 for each cell.
	cv::Mat feat_vec = cv::Mat::zeros(cv::Size(9, 1), CV_32F);
	cv::Mat temp_fv[16][8];
	cv::Mat final_fv_temp;
	
	// Histogram of 9 bins with a bin length of 20 degrees.
	constexpr std::array<int, 9> hist{ 0, 20, 40, 60, 80, 100, 120, 140, 160 };
	// Indices for the histogram vector
	int index, index1, index2, index_temp;

	// variables used in the calculation of histogram of a single cell.
	float mag_pixel, ang_pixel, weight1, weight2, diff1, diff2, current_diff, diff_rightbin, diff_leftbin;
	int counter1 = 0, counter2 = 0;

	//** Computing feature vector of length, 9 for all the cells in an image.
	// Moving and storing in Row-major order to make it cache-friendly.
	for (int iter1 = 0; iter1 < win_size.height; iter1 = iter1+cell_size.height)
	{
		for (int iter2 = 0; iter2 < win_size.width; iter2 = iter2+cell_size.width)
		{
			// Copying 8x8 cells from magnitude and angle matrices of an image to mag and ang.
			// (mag, ang) - 8x8 magnitude and angle matrices of a cell.
			magMat(cv::Range(iter1, iter1 + 8), cv::Range(iter2, iter2 + 8)).copyTo(mag);
			angMat(cv::Range(iter1, iter1 + 8), cv::Range(iter2, iter2 + 8)).copyTo(ang);

			// Iterating through all the rows and col's of 8x8 cell.
			// Row-major order.
			for (int row = 0; row < mag.rows; row++)
			{
				for (int col = 0; col < mag.cols; col++)
				{
					// Magnitude and Angle value per pixel in 8x8 cell.
					mag_pixel = mag.at<float>(row, col);
					ang_pixel = ang.at<float>(row, col);

					// Check if the angle value directly falls into one of the 9 bins.
					// If true, the complete magnitude value goes into that bin.
					if (fmod(ang_pixel, 20) == 0)
					{
						// Index of that angle between 0 to 9.
						index = (ang_pixel / 20);
						// If the angle = 180, map it to 0 degrees as it is circular.
						if(index==9)
							feat_vec.at<float>(0, 0) += mag_pixel;
						else
							feat_vec.at<float>(0, index) += mag_pixel;
					}
					// When the magnitude gets divided between two bins.
					else
					{
						// Identify the two bins with the least difference with angle value to divide the magnitude.
						// Difference between the angle and histogram bin values.
						diff1 = abs(ang_pixel - hist[0]);
						index1 = 0;
						// Comparing the angle value to all the bin values and considering two bins with lowest difference.
						// First bin with lowest difference.
						for (int iter3 = 1; iter3 < 9; iter3++)
						{
							current_diff = abs(ang_pixel - hist[iter3]);
							if (current_diff < diff1)
							{
								index1 = iter3;
								diff1 = current_diff;
							}
						}
						// When bin index = 8 i.e. angle = 180, maps to 0.
						index_temp = (index1 == 8) ? 0 : (index1 + 1);
						// When bin index = 8 i.e. angle = 180, subtract the angle with 180.
						diff_rightbin = (index1 == 8) ? abs(ang_pixel - 180) : abs(ang_pixel - hist[index_temp]);
						diff_leftbin = (index1 == 0)? abs(ang_pixel - 180): abs(ang_pixel - hist[index1 - 1]);
						
						// Selecting the second bin with lowest difference adjacent to first bin(left or right).
						if (diff_rightbin < diff_leftbin)
						{
							// Second bin = right bin to the first bin.
							index2 = index_temp;
							diff2 = diff_rightbin;
						}
						else
						{
							// Second bin = left bin to the first bin.
							index2 = index1-1;
							diff2 = diff_leftbin;
						}
						//** Weight-voting.
						// Magnitude weights for each of the two bins based on the ratio.
						weight1 = ((diff2 / 20) * mag_pixel);
						weight2 = ((diff1 / 20) * mag_pixel);

						// Accumulate the corresponding magnitudes to the two bins.
						feat_vec.at<float>(0, index1) += weight1;
						feat_vec.at<float>(0, index2) += weight2;
					}
				}
			}
			// Copying the feature vector of the cell to an array to concatenate later.
			feat_vec.copyTo(temp_fv[counter1][counter2]);
			// Initializing the matrices to 0 to start over with a new cell.
			mag = cv::Mat::zeros(cell_size.width, cell_size.height, CV_32F);
			ang = cv::Mat::zeros(cell_size.width, cell_size.height, CV_32F);
			feat_vec = cv::Mat::zeros(1, 9, CV_32F);
			// To track the index (column) of the cell (cell number) in an image.
			counter2++;
		}
		// To track the index (row) of the cell (cell number) in an image.
		counter1++;
		counter2 = 0;
	}

	//** Block Normalization with a block size = (16, 16), block stride = (8,8).
	block_normalize(temp_fv, final_fv_temp);

	// Final Feature Vector of length, 3780.
	final_fv = final_fv_temp.reshape(1, 3780);
	
	//** Constrast Normalization over the blocks.
	contrast_normalize(final_fv);

	//** Mat final_fv - contains the final feature vector of length, 3780.
}


