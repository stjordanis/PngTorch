//#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <iostream>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>
#include "../include/utils/vision_utils.hpp"
using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    auto VU= VisionUtils();
    torch::Device device(torch::kCUDA );
    // Input PNG-image
    png::image<png::rgb_pixel> imageI("clion.png");
    // Convert png::image into torch::Tensor
    torch::Tensor tensor = VU.pngToTorch(imageI, device); //Note: we are allocating on the GPU
    std::cout << "C:" << tensor.size(0) << " H:" << tensor.size(1) << " W:" << tensor.size(2) << std::endl;
    // Convert torch::Tensor into png::image
    png::image<png::rgb_pixel> imageO = VU.torchToPng(tensor);
    // Input PNG-image
    imageO.write("clion-output001.png");
    return 0;
}