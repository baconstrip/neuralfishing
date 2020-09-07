#ifndef REAL_LAYER_HPP
#define REAL_LAYER_HPP

#include <stdlib.h>
#include <memory>
#include <iostream>

#include "layer.hpp"

#include "Eigen/Dense"

using Eigen::Matrix;

template<size_t neuronCount, size_t previousLayerSize> 
class RealLayer : public Layer<neuronCount, previousLayerSize> {
private:
    std::unique_ptr<Eigen::MatrixXd> 
        (*activation_)(std::unique_ptr<Eigen::MatrixXd>);
    double (*randomiser_)();
    double (*biaser_)();

public: 
    Eigen::MatrixXd weights{previousLayerSize, neuronCount};
    Eigen::VectorXd biases{neuronCount};

    std::shared_ptr<Eigen::MatrixXd> lastActivation;
    std::shared_ptr<Eigen::MatrixXd> lastPreActivation;

    std::unique_ptr<Eigen::MatrixXd> evaluate(std::unique_ptr<Eigen::MatrixXd> input);
    void randomise();

    RealLayer(double (*rand)(), double (*bias)(), std::unique_ptr<Eigen::MatrixXd> 
        (*activation)(std::unique_ptr<Eigen::MatrixXd>)):
        randomiser_{rand}, activation_{activation}, biaser_{bias}{};
};
#endif