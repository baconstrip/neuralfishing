#ifndef IMAGES_HPP
#define IMAGES_HPP
#include "Eigen/Dense"

using Eigen::Matrix;

class MNISTImg
{
public:
    Matrix<double, Eigen::Dynamic, 1> data{784};
    char label;

};
#endif