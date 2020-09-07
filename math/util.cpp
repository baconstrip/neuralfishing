#include "util.hpp"
namespace {
    inline double ReLUHelper(double in) {
        if (in > 0)
        {
            return in;
        }
        else
        {
            return 0;
        }
    }
    inline double ReLUDervHelper(double in) {
        if (in > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

std::unique_ptr<Eigen::MatrixXd> Activations::Linear(std::unique_ptr<Eigen::MatrixXd> v) {
    // Pass through the values without modifying them.
    return std::move(v);
}

std::unique_ptr<Eigen::MatrixXd> Activations::ReLU(std::unique_ptr<Eigen::MatrixXd> v) {
    return std::make_unique<Eigen::MatrixXd>(v->unaryExpr(&ReLUHelper));
}

std::unique_ptr<Eigen::MatrixXd> Activations::SoftMax(std::unique_ptr<Eigen::MatrixXd> v) {
    Eigen::MatrixXd exponented = v->array().exp().matrix();
    Eigen::MatrixXd denom = exponented.colwise().sum();
    for (int i = 0; i < v->cols(); i ++) {
        exponented.col(i) /= denom(i);
    }

    return std::make_unique<Eigen::MatrixXd>(exponented);
}

std::unique_ptr<Eigen::MatrixXd> Helpers::CrossEntropy(Eigen::MatrixXd& prediction, Eigen::MatrixXd& real)
{
    const auto samples = real.size();
    return std::make_unique<Eigen::MatrixXd>((prediction - real) / samples);
}

std::unique_ptr<Eigen::MatrixXd> Derivatives::ReLUDerv(Eigen::MatrixXd& v) {
    return std::make_unique<Eigen::MatrixXd>(v.unaryExpr(&ReLUDervHelper));
}

void MatrixSize(std::string name, const Eigen::MatrixXd& m) {
    std::cout << "Matrix " << name << " size: " << m.rows() << "x" << m.cols()
        << std::endl << std::endl;
}

std::unique_ptr<Eigen::MatrixXd> Capture(std::string name, std::unique_ptr<Eigen::MatrixXd> m) {
    MatrixDump(name, *m);
    return std::move(m);
}

void MatrixDump(std::string name, Eigen::MatrixXd& m) {
    Eigen::IOFormat enthiccen;
    enthiccen.precision = 6;
    std::cout << "-----------------------------------------\n\n";
    std::cout << "Captured Matrix \"" << name << "\" ("<< m.rows() << "x" << m.cols() << "), value:\n\n" << m.format(enthiccen) << std::endl << std::endl;
    std::cout << "-----------------------------------------\n";
}