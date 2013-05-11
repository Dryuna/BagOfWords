/*
 * main.cpp
 *
 *  Created on: Nov 9, 2012
 *      Author: olga
 */

#include "compute_sift.h"
#include "clustering.h"
#include "histogram.h"
#include "ListDir.h"
#include <limits.h>
#include "param.h"
using namespace std;
using namespace cv;

const string outputPath = "/home/olga/workspace/my_works/BagOfWords/results";

# define K 1000// k-means


vector<string> imageNamesTrain;
vector<string> imageNamesTest;
vector<int> occ_features; // number of images in which feature(mean) i occurs
vector<vector<double> > imageHistsRE;
vector<Mat> codebook;

/*
 * After training phase:
 * imageSIFTs - vector<Mat> vector of Mat - matrix of SIFTs for each image
 * imageHists - vector of corresponding image histograms
 * imageNames - vector of names of images that where used for training
 * codebook - vector of means(k-means) or codewords
 * HistRE - re-weighted histograms
 */

vector<int> ComputeMeanOccurance(const vector<vector<int> >& imageHists);
void TrainingPart(vector<Mat> & imageSIFTs, vector<vector<int> >& imageHists)
{
    double time_start=clock(), time;
    ofstream out("Log.txt");
    out<<"Log file for CameraLocalization project"<<endl<<endl;
    out<<"\tImage names: "<<endl;
    for(int i=0;i<imageNamesTrain.size(); i++)
    {
        cout<<imageNamesTrain[i]<<endl;
        out<<imageNamesTrain[i]<<endl;
    }
    vector<Mat> features = GatherFeatures(imageNamesTrain, imageSIFTs);
    time = clock();
    out<<"Number of SIFTS: "<<features.size()<<" number of images "<<imageNamesTrain.size()<<endl;
    out<<"Extracting features time: "<<(time - time_start)/CLOCKS_PER_SEC<<" s"<<endl;
    cout<<"Extracting features time: "<<(time - time_start)/CLOCKS_PER_SEC<<" s"<<endl;

    cout<<"Clustering is in progress..."<<endl;
    //codebook = ClusterFeatures(K, features);
    codebook = ClusterFeaturesANN(K, features);

    cout<<"Codebook is created\n";
    out<<"Clustering time: "<<(clock() - time)/CLOCKS_PER_SEC<<" s"<<endl;
    cout<<"Clustering time: "<<(clock() - time)/CLOCKS_PER_SEC<<" s"<<endl;
    cout<<endl;

    imageHists = ConvertImagesToHistograms(codebook,imageSIFTs);
    cout<<"Histogram from training images were computed\n";

    occ_features = ComputeMeanOccurance(imageHists);
    /* re-weight training image histograms */
    for(int i=0; i<imageHists.size(); i++)
         imageHistsRE.push_back(ReweightHistogram(imageHists[i], occ_features, imageNamesTrain.size()));


    time = clock();
    cout<<"Time of training: "<<(time-time_start)/CLOCKS_PER_SEC<<" s"<<endl;
    out<<"Time of training: "<<(time-time_start)/CLOCKS_PER_SEC<<" s"<<endl;
    out.close();
}

template <class Vtype>
void PrintVector(vector<Vtype> v)
{
    for(int i=0; i<v.size(); i++)
        cout<<v[i]<<" ";
    cout<<endl;
}


vector<int> ComputeMeanOccurance(const vector<vector<int> >& imageHists)
{
    vector<int> occurance(imageHists[0].size(),0);
    for(int i=0;i<imageHists.size(); i++)
    {
        for(int j=0; j<imageHists[i].size(); j++)
        {
            if(imageHists[i][j] > 0)
                occurance[j]+=1;
        }
    }
    return occurance;
}

    /* returns Nsim similar images to query one */
void QueryImage(string imageName, int Nsim)
{
    cout<<"Query image: "<<imageName<<endl;

    vector<int> SimilarImage;
    Mat features = computeSifts(imageName);
    vector<int> imageHist = ComputeHistogram(codebook, features);
    vector<double> imageHistRE = ReweightHistogram(imageHist, occ_features, imageHistsRE.size());

   imageName.erase(imageName.end()-4, imageName.end());
   imageName.erase(imageName.begin(), imageName.begin()+imageName.find_last_of('/'));
   string fileName = outputPath+imageName+(string)".txt";
   cout<<fileName<<endl;
   ofstream out(fileName.c_str());
   out<<"Query image: "<<imageName<<endl;

    vector<double> sim; //vector of similarities of query image to each image in database;
    for(int i=0; i<imageHistsRE.size(); i++)
    {
        double distance_query_to_database_i = CompareHistograms(imageHistRE, imageHistsRE[i]);
        sim.push_back( distance_query_to_database_i);
    }
    cout<<"Similarity: "<<endl;
    PrintVector<double>(sim);


   /* for finding the best Nsim matches */
   vector<double> sim_tmp = sim;
   vector<int> simImages;
   for(int i=0; i<Nsim; i++)
   {
       vector<double>::iterator idx_iter = max_element(sim_tmp.begin(), sim_tmp.end());
       int idx = idx_iter - sim_tmp.begin();
       simImages.push_back(idx);
       *(idx_iter)=INT_MIN;
   }
   out<<"Similar images were found"<<endl;
   out<<"Similar images are:"<<endl;
   for(int i=0; i<simImages.size(); i++)
   {
       out<<i+1<<") "<<imageNamesTrain[simImages[i]]<<" "<<sim[simImages[i]]<<endl;
       cout<<i+1<<") "<<imageNamesTrain[simImages[i]]<<" "<<sim[simImages[i]]<<endl;
   }
   out.close();
}


/* test query images for the whole database */
void TestingPart(int Nsim)
{
    cout<<"\n\tTesting begins"<<endl;

    // const int NUM_QUERY_ENTRIES = imageNamesTest.size();
    // const int NUM_DB_ENTRIES = imageNamesTrain.size();


    for(int i=0; i<imageNamesTest.size(); i++)
    {
        // cout<<"Processing image: "<<imageNamesTest[i]<<endl;
        QueryImage(imageNamesTest[i], Nsim);
        cout<<"Similarity values were computed"<<endl;
    }
}

/* gives the possibility to divide training and testing part */
void WriteTraining()
{
    ofstream out("train/codebook.txt");
    for(int i=0; i<codebook.size(); i++)
        out<<codebook[i]<<endl;
    out.close();
    out.open("train/hists.txt");
    for(int i=0; i<imageHistsRE.size(); i++)
    {
        for(int j=0; j<imageHistsRE[i].size(); j++)
            out<<imageHistsRE[i][j]<<" ";
        out<<endl;
    }
    out.close();
    out.open("train/occurance.txt");
    for(int i=0; i<occ_features.size(); i++)
    {
        out<<occ_features[i]<<" ";
    }
        out.close();
}



int main()
{
    vector<Mat> imageSIFTs;
    vector<vector<int> > imageHists;
    imageNamesTrain = ListDirectories("/home/olga/Documents/Work/BagOfWords/train");
    imageNamesTest = ListDirectories("/home/olga/Documents/Work/BagOfWords/test");
    // imageNamesTrain = load_pxgps_log("/home/olga/Documents/Work/the2_awesome_students/example_DB.pxgps");
    // imageNamesTest = load_pxgps_log("/home/olga/Documents/Work/the2_awesome_students/example_QRY.pxgps");

//    cout<<"Training"<<endl;
//    for(int i=0; i<imageNamesTrain.size(); i++)
//        cout<<imageNamesTrain[i]<<endl;

//    cout<<"Testing"<<endl;
//    for(int i=0; i<imageNamesTest.size(); i++)
//        cout<<imageNamesTest[i]<<endl;
////    exit(0);

    TrainingPart(imageSIFTs,imageHists);
    //WriteTraining();

    if(codebook.size() == 0 || imageHistsRE.size() == 0 || occ_features.size()==0)
        cout<<"\nWARNING: Some variable after training were not initialized\n";

    TestingPart(2);

    cout<<"\nSuccess"<<endl;
    return 0;
}


