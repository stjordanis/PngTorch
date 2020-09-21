#include <iostream>
#include <chrono>
#include <iostream>
#include <time.h>
#include <typeinfo>
#include <png++/png.hpp>
#include <thread>
#include <future>
using namespace std;
using namespace std::chrono;
class VisionUtils {
public:
    VisionUtils();
    void tensorDIMS(const torch::Tensor &tensor);
    torch::Tensor pngToTorch(png::image<png::rgb_pixel> &image, c10::Device &device);
    png::image<png::rgb_pixel> torchToPng(torch::Tensor &tensor_);
};

VisionUtils::VisionUtils() {}

//Adapted from https://github.com/koba-jon/pytorch_cpp/blob/master/utils/visualizer.cpp
torch::Tensor VisionUtils::pngToTorch(png::image<png::rgb_pixel> &image, c10::Device &device){
    size_t width = image.get_width();
    size_t height = image.get_height();
    unsigned char *pointer = new unsigned char[width * height * 3];
    for (size_t j = 0; j < height; j++){
        for (size_t i = 0; i < width; i++){
            pointer[j * width * 3 + i * 3 + 0] = image[j][i].red;
            pointer[j * width * 3 + i * 3 + 1] = image[j][i].green;
            pointer[j * width * 3 + i * 3 + 2] = image[j][i].blue;
        }
    }
    torch::Tensor tensor = torch::from_blob(pointer, {image.get_height(), image.get_width(), 3}, torch::kUInt8).clone().to(device);  // copy
//    torch::Tensor tensor = torch::from_blob(pointer, {image.get_height(), image.get_width(), 3}, torch::kUInt8).clone();  // copy
    tensor = tensor.permute({2, 0, 1});  // {H,W,C} ===> {C,H,W}
    delete[] pointer;
    return tensor;
}

png::image<png::rgb_pixel> VisionUtils::torchToPng(torch::Tensor &tensor_){

    torch::Tensor tensor = tensor_.detach().cpu().permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
    size_t width = tensor.size(1);
    size_t height = tensor.size(0);
    unsigned char *pointer = tensor.data_ptr<unsigned char>();
    png::image<png::rgb_pixel> image(width, height);
    for (size_t j = 0; j < height; j++){
        for (size_t i = 0; i < width; i++){
            image[j][i].red = pointer[j * width * 3 + i * 3 + 0];
            image[j][i].green = pointer[j * width * 3 + i * 3 + 1];
            image[j][i].blue = pointer[j * width * 3 + i * 3 + 2];
        }
    }
    return image;
}


void VisionUtils::tensorDIMS(const torch::Tensor &tensor) {
    auto t0 = tensor.size(0);
    auto s = tensor.sizes();
    cout << "D=:" << s << "\n";
}