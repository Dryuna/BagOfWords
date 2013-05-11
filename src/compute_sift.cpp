#include "compute_sift.h"
using namespace std;
using namespace cv;

Mat computeSifts(const string& fileName)
{
    const Mat input = cv::imread(fileName.c_str(), 0); //Load as grayscale
    if(input.empty())
        cout<<"ERROR: Image "<<fileName<<" was not read"<<endl;
    Mat descriptors;
    SiftFeatureDetector detector;
    vector<cv::KeyPoint> keypoints;
    detector.detect(input, keypoints);
    SiftDescriptorExtractor extractor;
    extractor.compute(input, keypoints, descriptors);
    // cout<<descriptors<<endl;
    return descriptors;
}
