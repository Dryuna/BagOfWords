#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "param.h"


std::vector<cv::Mat> ClusterFeatures(int k, const std::vector<cv::Mat>& features);
double ComputeDistance(const cv::Mat& v1, const cv::Mat& v2);
std::vector<cv::Mat> ClusterFeaturesANN(int k, const std::vector<cv::Mat>& features);


#endif // CLUSTERING_H
