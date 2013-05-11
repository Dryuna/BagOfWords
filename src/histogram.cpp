#include "histogram.h"
#include <math.h>
using namespace std;
using namespace cv;

/*extracts SIFTs features from images. Returns SIFTs from all dataset */
vector<Mat> GatherFeatures(const vector<string>& imageNames, vector<Mat>& imageSIFT)
{
    vector<Mat> features;
    for(int j=0; j<imageNames.size(); j++)
    {
        Mat siftImage = computeSifts(imageNames[j]);
        imageSIFT.push_back(siftImage);
        for(int i=0; i<siftImage.rows; i++)
        {
            Mat imageRow = siftImage.row(i);
            features.push_back(imageRow);
        }
        cout<<"Num of Sifts after "<<j<<" is: "<<features.size()<<endl;
    }
    cout<<"Num of Sifts: "<<features.size()<<endl;
    return features;
}

/* compute histogram for the image according to the dictionary*/
vector<int> ComputeHistogram(vector<Mat> dictionary, Mat siftImage)
{
    vector<Mat> features;
    vector<int> hist(dictionary.size(),0);

    /* transforming Mat into vector<Mat> */
    for(int i=0; i<siftImage.rows ; i++)
        features.push_back(siftImage.row(i));

    Mat dictionaryMat;
    for(int i=0; i<dictionary.size(); i++)
        dictionaryMat.push_back(dictionary[i]);

    cv::flann::KDTreeIndexParams indexParams(5);
    cv::flann::Index kdtree(dictionaryMat, indexParams);

    for(int i=0; i<features.size(); i++)
    {

        int min_idx;
        Mat nn_idx(1,1,DataType<int>::type);
        Mat nn_dist(1,1,DataType<int>::type);

        kdtree.knnSearch(features[i], nn_idx, nn_dist, 1, cv::flann::SearchParams(64));
        min_idx = nn_idx.at<int>(0,0);
        if(min_idx == -1)
            cout<<"ERROR: in ComputeHistogram: idx out of the range"<<endl;
        hist[min_idx]+=1;
    }
    return hist;
}

template <class Htype>
void WriteHistogram(vector<vector<Htype> > hists, string fileName)
{
    ofstream out(fileName.c_str());
    for(int i=0; i<hists.size(); i++)
    {
        for(int j=0;j<hists[i].size(); j++)
        {
            out<<hists[i][j]<<" ";
        }
        out<<endl;
    }
    out.close();
}

/* creates a data set where training images are represent as histograms */
vector<vector<int> > ConvertImagesToHistograms(vector<Mat> codebook, vector<Mat> imageSIFTs)
{
    vector<vector<int> > Hists;
    if(imageSIFTs.size() == 0)
        cout<<"WARNING: imageSIFTs were not initialized"<<endl;
    for(int i=0; i<imageSIFTs.size(); i++)
    {
        Hists.push_back(ComputeHistogram(codebook,imageSIFTs[i]));
    }
    WriteHistogram<int>(Hists, "hist.dat");
    return Hists;
}



/*
 * nid - hist[i];
 * nd - sum over hist // all bins
 * N - number of images in dataset Nimages
 * ni - how many images have mean>0
 */

vector<double> ReweightHistogram(vector<int> hist, vector<int> occ_words, int Nimages)
{
    vector<double> weights;
    if(hist.size() != occ_words.size())
        cout<<"ERROR: wrong dim. Reweighting cannot be done\n";

    int nd=0;
    for(int i=0; i<hist.size(); i++)
        nd +=hist[i];

    for(int i=0; i< hist.size(); i++)
    {
        int ni = occ_words[i];
        int nid = hist[i];
        double value;
        if(ni == 0)
        {
            cout<<"ERROR: Word "<<i<<" is not occuring in any image\n";
            exit(1);
        }
        else
            value = (double)nid/nd * log((double)Nimages/ni);

        weights.push_back(value);
    }

    return weights;

}


double CompareHistograms(vector<double> hist1, vector<double> hist2)
{
    double res;
    double sumAB = 0,sumA = 0, sumB = 0;
    if(hist1.size()!= hist2.size())
        cout<<"ERROR: Histograms are of different size"<<endl;
    for(int i=0; i<hist1.size(); i++)
    {
        sumAB += hist1[i]*hist2[i];
        sumA += hist1[i]*hist1[i];
        sumB += hist2[i]*hist2[i];
    }

    res = sumAB /(sqrt(sumA) * sqrt(sumB));

    return res;
}


