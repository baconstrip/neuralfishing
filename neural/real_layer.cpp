#include "real_layer.hpp"
#include "../math/util.hpp"

template<size_t neuronCount, size_t previousLayerSize> 
std::unique_ptr<Eigen::MatrixXd> RealLayer<neuronCount, previousLayerSize>::
        evaluate(std::unique_ptr<Eigen::MatrixXd> input) {

    Eigen::MatrixXd activation = input->transpose() * this->weights;
    Eigen::MatrixXd output = (activation.transpose().colwise() + this->biases);
    this->lastPreActivation = std::make_shared<Eigen::MatrixXd>(output);

    std::unique_ptr<Eigen::MatrixXd> activated = this->activation_(std::make_unique<Eigen::MatrixXd>(output));

    const auto temp = *activated;
    this->lastActivation = std::make_shared<Eigen::MatrixXd>(temp);
    return activated;
}

template<size_t neuronCount, size_t previousLayerSize> 
void RealLayer<neuronCount, previousLayerSize>::randomise() {
    for (int i = 0; i< neuronCount; i++) {
        for (int j = 0; j < previousLayerSize; j++) {
            this->weights(j,i) = this->randomiser_();
        }
        this->biases(i) = this->biaser_();
    }
}