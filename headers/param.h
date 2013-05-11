#ifndef PARAM_H
#define PARAM_H

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

typedef struct
{
    std::string fname;
    double timestamp;
    int gps_idx;
}pix_entry;

class matching_entry
{
    public:
     std::vector< std::pair<double, int> > mtch;
     std::vector< int  > tp,tn,fp,fn;

  template<typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mtch;
    }

};


#endif // PARAM_H
