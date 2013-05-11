#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "compute_sift.h"
#include "clustering.h"



std::vector<cv::Mat> GatherFeatures(const std::vector<std::string>& imageNames, std::vector<cv::Mat>& imageSIFT);
std::vector<int> ComputeHistogram(std::vector<cv::Mat> dictionary, cv::Mat siftImage);
std::vector<std::vector<int> > ConvertImagesToHistograms(std::vector<cv::Mat> codebook, std::vector<cv::Mat> imageSIFTs);
std::vector<double> ReweightHistogram(std::vector<int> hist, std::vector<int> occ_words, int Nimages);
void WriteHistogram(std::vector<std::vector<double> > hists);
double CompareHistograms(std::vector<double> hist1, std::vector<double> hist2);

#endif // HISTOGRAM_H
