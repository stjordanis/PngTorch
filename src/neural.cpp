//#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include <iostream>
#include <chrono>

#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>
#include "../include/utils/vision_utils.hpp"
using namespace std;
using namespace std::chrono;
#define kIMAGE_SIZE 512
#define kCHANNELS 3
#define kTOP_K 3

//temporarily disable gradient recording
//NoGradGuard guard;


int main(int argc, char *argv[]) { //./../style_model_cpp.pt ./../imgs/RGB_00_001.jpg
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

//    auto videoLambda =[&](const std::string&  modelName, c10::Device device, const std::string&  vidName) {
//        VU.processVideo(modelName, device, vidName);
//    };
//
//    // Loading your model
//    const std::string s_model_name0 = "style_model_cpp.pt";
//    const std::string s_model_name1 = "erfnet_fs.pt";
//    const std::string s_model_name2 = "vgg19_layers.pt"; // Forward= D=:[1, 512, 22, 40]
//    auto vid1Callback2 =std::async(std::launch::async, [&](){
//        videoLambda(s_model_name0, device,  s_image_name0);
//    });
    return 0;
}