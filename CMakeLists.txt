cmake_minimum_required(VERSION 2.8)
PROJECT( bow)
FIND_PACKAGE( OpenCV REQUIRED )

set (MY_SRC src/main.cpp src/clustering.cpp src/compute_sift.cpp src/histogram.cpp)
include_directories( headers)

ADD_EXECUTABLE(bow ${MY_SRC})
TARGET_LINK_LIBRARIES( bow ${OpenCV_LIBS})
