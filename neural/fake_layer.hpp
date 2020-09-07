#ifndef FAKE_LAYER_HPP
#define FAKE_LAYER_HPP

#include <memory>
#include <stdlib.h>

#include "layer.hpp"

#include "Eigen/Dense"

using Eigen::Matrix;

template<size_t layerSize> class FakeLayer: public Layer<layerSize, 0> {
public: 
    Matrix<double, layerSize, 1> data;

    std::unique_ptr<Matrix<double, Eigen::Dynamic, 1>> evaluate(std::unique_ptr<Matrix<double, Eigen::Dynamic, 1>> do_not_use) override {
        return std::make_unique<Matrix<double, Eigen::Dynamic, 1>>(this->data);
    }
};

#endif 