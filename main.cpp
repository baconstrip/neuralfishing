#include "Eigen/Dense"

#include "math/util.hpp"
#include "neural/images.hpp"
#include "neural/real_layer.cpp"
#include "neural/fake_layer.hpp"
#include "neural/mnist_input_layer.cpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <stdint.h>
#include <random>

using Eigen::MatrixXd;
using namespace std;

const uint32_t MAGIC_LABELS = 2049;
const uint32_t MAGIC_IMAGES = 2051;

inline double translateToDouble(uint8_t x) {
    return ((((double)x))/255.0);
}

inline void printImage(Eigen::MatrixXd& img) {

    for (int i = 0; i < 28; i++) {
        printf("\n");
        for (int j = 0; j < 28; j++) {
            printf("%s", img(i * 28 + j) > 0 ? "1" : "0");
        }
    }
    printf("\n");
}

inline void printImage(MNISTImg& img) {
    Eigen::MatrixXd data = img.data;
    printImage(data);
}


void readLEBytes(ifstream &stream, uint32_t count, uint64_t *dest)
{
    for (uint32_t i = 0; i < count; i++)
    {
        *dest <<= 8;
        char temp = 0;
        stream.readsome(&temp, 1);
        *dest = *dest + (uint8_t)temp;
    }
}

const vector<MNISTImg> loadImages(ifstream& images, ifstream& labels) {
    uint64_t labelCount = 0;
    uint64_t imageCount = 0;

    readLEBytes(images, 4, &imageCount);
    readLEBytes(labels, 4, &labelCount);
    if (imageCount != labelCount)
    {
        cout << "Count mismatch";
        exit(1);
    }
    else
    {
        cout << "Got " << imageCount << " images" << endl;
    }

    vector<uint8_t> labelVec;

    for (int i = 0; i < imageCount; i++)
    {
        char temp = 0;
        labels.readsome(&temp, 1);
        labelVec.push_back(temp);
    }

    // Burn two values
    for (int i = 0; i < 2; i++)
    {
        uint64_t temp;
        readLEBytes(images, 4, &temp);
    }

    vector<MNISTImg> imageData;

    for (int i = 0; i < imageCount; i++) {
        MNISTImg img;
        char singleImage[28 * 28];
        images.readsome(singleImage, sizeof(singleImage));

        for (int j = 0; j < 28 * 28; j++) {
            img.data(j) = translateToDouble(singleImage[j]); 
        }
        
        img.label = labelVec[i]; 
        imageData.push_back(img);
        if (i == 0 || i == 1) {
            printImage(img);
            printf("Label: %d\n", img.label);
        }
    }

    return imageData;
}


const ifstream openData(const char* path, bool image) {
    ifstream d(path);
    if (!d.is_open())
    {
        cout << "Failed to opening training data";
        exit(1);
    }

    
    uint64_t magic = 0;
    readLEBytes(d, 4, &magic);
    if (magic != 2051 && image || magic != 2049 && !image)
    {
        cout << "Bad magic number for file";
        exit(1);
    }

    return d;
}

std::normal_distribution<double> normie_dist(0, 0.3);
std::mt19937 random_generator(time(0));

double fishinRandom() {
    return normie_dist(random_generator);
}

double baseBias() {
    return 0.1;
}

const size_t batchSize = 50000;
const double learningRate = 0.1;
const size_t epochCount = 5000;

int main()
{

    std::clock_t c_start = std::clock();

    auto trainingData = openData("data/train-images-idx3-ubyte", true);
    auto trainingLabels = openData("data/train-labels-idx1-ubyte", false);
    auto testingData = openData("data/t10k-images-idx3-ubyte", true);
    auto testingLabels = openData("data/t10k-labels-idx1-ubyte", false);

    const auto trainingImages = loadImages(trainingData, trainingLabels);
    const auto testingImages = loadImages(testingData, testingLabels);

    auto c_end = std::clock();

    cout << std::fixed << std::setprecision(2) << "CPU time to load images: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << "ms\n";


    cout << "Successfully loaded input data" << endl;

    auto imgPtr = std::make_shared<vector<MNISTImg>>(trainingImages);

    printImage((*imgPtr.get())[5]);

    // Testing neural net

    MNISTInputLayer imgs_layer(batchSize, imgPtr);

    RealLayer<64, 28*28> layer1(fishinRandom, baseBias, Activations::ReLU);
    RealLayer<16, 64> layer2(fishinRandom, baseBias, Activations::ReLU);
    RealLayer<10, 16> outLayer(fishinRandom, baseBias, Activations::SoftMax);

    layer1.randomise();
    layer2.randomise();
    outLayer.randomise();

    for (int epoch = 0; epoch < epochCount; epoch++) {
        for (int i = 0; i < 1; i ++) {
            if (epoch % 100 == 0) {
                cout << endl << endl << "-------- 100 complete -------------" << endl << endl;
            }
            // if (i == 0){
            //     MatrixDump("w3", outLayer.weights);
            // }
            c_start = std::clock();
            auto result = outLayer.evaluate(
                layer2.evaluate(
                layer1.evaluate(
                imgs_layer.evaluate(NULL)
            )));

            c_end = std::clock();
            //cout << std::fixed << std::setprecision(2) << "CPU time to infer one image: " << 1000000.0 * (c_end-c_start) / CLOCKS_PER_SEC << "us\n";

            Eigen::MatrixXd expected(10,batchSize);
            expected.setZero();
            for (int j = 0; j < batchSize; j++) {
                expected(imgPtr->at((imgs_layer.counter() * batchSize) + j).label, j) = 1.0;
            }

            // cout << "expected: " << endl << expected << endl;
            // cout << "result: " << endl << *result << endl;
            
            // auto imgLayerAct = imgs_layer.evaluate(NULL);
            // Eigen::MatrixXd firstImg = imgLayerAct->col(1);
            // printImage(firstImg);


            Eigen::VectorXd indicies(batchSize);
            for (int j = 0; j < batchSize; j++) {
                indicies(j) = imgPtr->at((imgs_layer.counter() * batchSize) + j).label;
            }

            Eigen::VectorXd predictions(batchSize);
            for (int j = 0; j < batchSize; j++) {
                double max = -2;
                size_t idx = -1;
                for (int k = 0; k < 10; k++) {
                    if ((*result)(k, j) > max) {
                        max = (*result)(k, j);
                        idx = k;
                    }
                }

                predictions(j) = idx;
            }

            Eigen::VectorXd correctAsZero = predictions - indicies;
            size_t incorrect = correctAsZero.cwiseAbs().count();
            cout << "Number incorrect: " << incorrect << endl;


            Eigen::VectorXd costs(batchSize);
            for (int j = 0; j < batchSize; j++) {
                costs(j) = (*result)(indicies(j));
            }

            auto error = (-costs.array().log()).sum() / (double)batchSize;
            cout << "Error: " << error << endl;

            // auto cost = -(expected.dot(result->array().log().matrix().transpose()));
            // std::cout << "Result: ";
            // std::cout << *result.get() << endl;
            //std::cout << "Error: ";
            //std::cout << cost << endl;

            const auto samples = expected.size();

            Eigen::MatrixXd dZ3 = *result - expected;

            auto dW3 = (1./samples) * (dZ3 * layer2.lastActivation->transpose());
            auto db3 = (1./samples) * dZ3.rowwise().sum();

            auto dA2 = outLayer.weights * dZ3;
            auto dZ2 = dA2.cwiseProduct( *Derivatives::ReLUDerv(*layer2.lastPreActivation).release());
            auto dW2 = (1./samples) * (dZ2 * layer1.lastActivation->transpose());
            auto db2 = (1./samples) * dZ2.rowwise().sum();

            auto dA1 = layer2.weights * dZ2;
            auto dZ1 = dA1.cwiseProduct( *Derivatives::ReLUDerv(*layer1.lastPreActivation).release());
            auto dW1 = (1./samples) * (dZ1 * imgs_layer.lastActivation->transpose());
            auto db1 = (1./samples) * dZ1.rowwise().sum();

            outLayer.weights -= learningRate * dW3.transpose();
            outLayer.biases -= learningRate * db3;

            layer2.weights -= learningRate * dW2.transpose();
            layer2.biases -= learningRate * db2;

            layer1.weights -= learningRate * dW1.transpose();
            layer1.biases -= learningRate * db1;

//  ---------------------- Old backprop
            // // --- begin backprop values calc
            // Eigen::IOFormat enthiccen;
            // enthiccen.precision= 6;

            // const auto samples = expected.size();
            // auto a3_delta = (*result - expected).transpose();
            // //auto a3_delta = Helpers::CrossEntropy(*result, expected).release()->transpose();
            // auto z2_delta = a3_delta * outLayer.weights.transpose();
            // auto a2_delta = z2_delta.cwiseProduct(Derivatives::ReLUDerv(*layer2.lastActivation).release()->transpose());
            // auto z1_delta = a2_delta * layer2.weights.transpose();
            // auto a1_delta = z1_delta.cwiseProduct( Derivatives::ReLUDerv(*layer1.lastActivation).release()->transpose());

            // // --- begin value nudging 

            // auto w3_factor = (*layer2.lastActivation * a3_delta) / samples;
            // auto b3_factor = a3_delta.colwise().sum() / samples;
            
            // auto w2_factor = *layer1.lastActivation * a2_delta;
            // auto b2_factor = a2_delta.colwise().sum() / samples;

            // auto w1_factor = *imgs_layer.lastActivation * a1_delta;
            // auto b1_factor = a1_delta.colwise().sum() / samples;

            // outLayer.weights -= learningRate * w3_factor;
            // outLayer.biases -= learningRate * b3_factor;

            // layer2.weights -= learningRate * w2_factor;
            // layer2.biases -= learningRate * b2_factor;

            // layer1.weights -= learningRate * w1_factor;
            // layer1.biases -= learningRate * b1_factor;
//  ---------------------- Old backprop

            imgs_layer.advance();
        }
        imgs_layer.reset();

        cout << "\t\tEpoch complete!\n\n";
    }
}
