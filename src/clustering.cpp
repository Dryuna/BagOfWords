#include "clustering.h"
using namespace cv;
using namespace std;


#define Epsilon 1 //end criterion
#define MAX_ITER 100



/* compute Euclidean distance between 2 feature vectors */
double ComputeDistance(const Mat& v1, const Mat& v2)
{
    if(v1.size() != v2.size())
        return 100000;
    Mat v = v1-v2;
    double dist = v.dot(v);
    dist = sqrt(dist);
    return dist;
}

/* criterion for stop iterating */
bool MeanIsStable(vector<Mat>& m_old, vector<Mat>& m_new)
{
    double dist=0;

    for(int i=0; i<m_old.size(); i++)
    {
        dist += ComputeDistance(m_old[i], m_new[i]);
    }
    dist = dist/m_old.size();
    cout<<"dist: "<<dist<<endl;
    if(dist > Epsilon)
        return false;
    else
        return true;
}

/* input: vector of features
 * output: vector of means
 * assuming Mat be vector everywhere in this function
 */
vector<Mat> ClusterFeatures(int k, const vector<Mat>& features)
{
    vector<Mat> means;
    vector<Mat> old_means;
    vector<int> means_idx;
    srand(time(0));
    for(int i=0; i<k; i++)
    {
        int r = rand()% features.size();
        means.push_back(features[r]);
    }

    int iter=0;
    do
    {
        /* compute assignment of the points to the clusters */

        vector<Mat> assign;
        for(int f=0; f<features.size(); f++)
        {
            int min_idx = -1; /* idx of the closest mean for feature[f]*/
            double min_dist = 100000;

            for(int m=0; m<means.size(); m++)
            {
                double dist = ComputeDistance(features[f],means[m]);
                if(min_dist > dist)
                {
                    min_dist = dist;
                    min_idx = m;
                }
            }
            Mat assign_tmp = Mat::zeros(1, k, DataType<int>::type);
            assign_tmp.at<int>(0,min_idx) = 1;
            assign.push_back(assign_tmp);
        }


        /* Recompute means */


        vector<Mat> new_means;
        means_idx.clear();
        Mat tmp;
        for(int i=0; i<k; i++)
        {
            new_means.push_back(tmp);
            means_idx.push_back(0);
        }

        for(int i=0; i<assign.size();i++)
        {

            Mat prob = assign[i];
            for(int j=0; j<means.size(); j++)
            {
                if(prob.at<int>(0,j) > 0)
                {
                    if(new_means[j].empty())
                    {
                        features[i].copyTo(new_means[j]);//new_means[j] = f;
                    }
                    else
                    {
                        Mat tmp;
                        features[i].copyTo(tmp);
                        new_means[j] = new_means[j] +tmp;
                    }
                    means_idx[j] = means_idx[j]+1;
                }
            }
        }

        /* normalize means */
        for(int i=0; i<new_means.size(); i++)
        {
            new_means[i] = new_means[i]/means_idx[i];
        }

        old_means = means;
        means = new_means;
        iter++;
    }while(!MeanIsStable(means, old_means) && iter<MAX_ITER);

    /* check if there are means with no points associated*/
    cout<<"Cheking for empty means"<<endl;
    for(int i=0; i<means_idx.size(); i++)
    {
        if(means_idx[i] == 0)
        {
            //cout<<"Size before: "<<means_idx.size()<<" "<<means.size()<<endl;
            means_idx.erase(means_idx.begin()+i);
            means.erase(means.begin() + i);
            i--; /* checking once again for 0 0 situations*/
            //cout<<"Size after: "<<means_idx.size()<<" "<<means.size()<<endl;
        }
    }

    cout<<"Number of iteration to converge: "<<iter<<endl;
    return means;
}

vector<Mat> ClusterFeaturesANN(int k, const vector<Mat>& features)
{
    vector<Mat> means;
    vector<int> means_idx(k,0);
    vector<Mat> old_means;

    srand(time(0));
    for(int i=0; i<k; i++)
    {
        int r = rand()% features.size();
        means.push_back(features[r]);
    }
    Mat means_tmp;
    for(int i=0; i<means.size(); i++)
        means_tmp.push_back(means[i]);
    int iter=0;
    cv::flann::KDTreeIndexParams indexParams(4);

    do
    {
        /* compute assignment of the points to the clusters */
        cv::flann::Index kdtree(means_tmp, indexParams);
        vector<Mat> assign;
        for(int f=0; f<features.size(); f++)
        {
            int min_idx;
            Mat nn_idx(1,1,DataType<int>::type);
            Mat nn_dist(1,1,DataType<int>::type);

            kdtree.knnSearch(features[f], nn_idx, nn_dist, 1);
            min_idx = nn_idx.at<int>(0,0);
            Mat assign_tmp = Mat::zeros(1, k, DataType<int>::type);
            assign_tmp.at<int>(0,min_idx) = 1;
            assign.push_back(assign_tmp);
        }


        /* Recompute means */

        Mat tmp;
        vector<Mat> new_means(k,tmp);
        means_idx.assign(means_idx.size(),0);

        for(int i=0; i<assign.size();i++)
        {

            Mat prob = assign[i];
            for(int j=0; j<means.size(); j++)
            {
                if(prob.at<int>(0,j) > 0)
                {
                    if(new_means[j].empty())
                    {
                        features[i].copyTo(new_means[j]);//new_means[j] = f;
                    }
                    else
                    {
                        Mat tmp;
                        features[i].copyTo(tmp);
                        new_means[j] = new_means[j] +tmp;
                    }
                    means_idx[j] = means_idx[j]+1;
                }
            }
        }

        /* normalize means */
        for(int i=0; i<new_means.size(); i++)
        {
            new_means[i] = new_means[i]/means_idx[i];
        }

        old_means = means;
        means = new_means;
        Mat new_means_tmp;
        for(int i=0;i< means.size(); i++)
            new_means_tmp.push_back(means[i]);
        means_tmp = new_means_tmp;

        iter++;
    }while(!MeanIsStable(means, old_means) && iter<MAX_ITER);

    /* check if there are means with no points associated*/
    cout<<"Cheking for empty means"<<endl;
    for(int i=0; i<means_idx.size(); i++)
    {
        if(means_idx[i] == 0)
        {
            //cout<<"Size before: "<<means_idx.size()<<" "<<means.size()<<endl;
            means_idx.erase(means_idx.begin()+i);
            means.erase(means.begin() + i);
            i--; /* checking once again for 0 0 situations*/
            //cout<<"Size after: "<<means_idx.size()<<" "<<means.size()<<endl;
        }
    }
    cout<<"Size after: "<<means_idx.size()<<" "<<means.size()<<endl;

    cout<<"Number of iteration to converge: "<<iter<<endl;
    return means;
}

