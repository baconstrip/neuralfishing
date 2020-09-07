#ifndef UTIL_HPP 
#define UTIL_HPP

#include "Eigen/Dense"

#include <memory>
#include <iostream>

namespace Activations
{
    std::unique_ptr<Eigen::MatrixXd> 
        Linear(std::unique_ptr<Eigen::MatrixXd> v);
    std::unique_ptr<Eigen::MatrixXd> 
        ReLU(std::unique_ptr<Eigen::MatrixXd> v);
    std::unique_ptr<Eigen::MatrixXd>
        SoftMax(std::unique_ptr<Eigen::MatrixXd> v);
} // namespace Activations

namespace Helpers
{
    std::unique_ptr<Eigen::MatrixXd> CrossEntropy(
        Eigen::MatrixXd& prediction,
        Eigen::MatrixXd& real 
    );
}

namespace Derivatives 
{
    std::unique_ptr<Eigen::MatrixXd> ReLUDerv(Eigen::MatrixXd& v);
}

void MatrixSize(std::string name, const Eigen::MatrixXd& m);
std::unique_ptr<Eigen::MatrixXd> Capture(std::string name, std::unique_ptr<Eigen::MatrixXd> m);
void MatrixDump(std::string name, Eigen::MatrixXd& m);
#endif