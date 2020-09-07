#include "mnist_input_layer.hpp"
#include <iostream>

std::unique_ptr<Eigen::MatrixXd> MNISTInputLayer::evaluate(std::unique_ptr<Eigen::MatrixXd> do_not_use){
    Eigen::MatrixXd data(784, this->batchSize_);
    for (int i = 0; i < this->batchSize_; i++) {
        data.col(i) = (*this->imgs_.get())[this->counter_*this->batchSize_ + i].data;
    }
    this->lastActivation = std::make_shared<Eigen::MatrixXd>(data);
    return std::make_unique<Eigen::MatrixXd>(data);
}

bool MNISTInputLayer::advance() {
    if (this->counter_ == this->imgs_->size()) {
        this->counter_ = 0;
        return true;
    }
    this->counter_++;
    return false;
}