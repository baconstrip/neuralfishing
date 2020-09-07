#ifndef MNIST_INPUT_LAYER_HPP
#define MNIST_INPUT_LAYER_HPP

#include <stdlib.h>
#include <memory>

#include "layer.hpp"
#include "images.hpp"

#include "Eigen/Dense"

using Eigen::Matrix;

class MNISTInputLayer: public Layer<28 * 28, 0> {
private:
    std::shared_ptr<std::vector<MNISTImg>> imgs_;
    size_t counter_ = 0;
    size_t batchSize_ = 0;

public: 
    std::unique_ptr<Eigen::MatrixXd> evaluate(std::unique_ptr<Eigen::MatrixXd> do_not_use) override;
    void randomise();
    std::shared_ptr<Eigen::MatrixXd> lastActivation;

    MNISTInputLayer(size_t batchSize, 
        std::shared_ptr<std::vector<MNISTImg>> imgs): 
            imgs_{imgs}, batchSize_{batchSize} {};
    bool advance();
    void reset() { this->counter_ = 0; }

    size_t counter() { return this->counter_; };
};
#endif