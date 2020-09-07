#ifndef LAYER_HPP
#define LAYER_HPP

#include <memory>
#include <stdlib.h>

#include "Eigen/Dense"

template<size_t layerSize, size_t previousLayer> class Layer {
public:
    virtual std::unique_ptr<Eigen::MatrixXd> evaluate(std::unique_ptr<Eigen::MatrixXd>) = 0;
};
#endif