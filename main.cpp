#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
//#include <opencv2/opencv.hpp>


int main() {
//    auto a = torch::tensor({{1,2},{3,4}});
//
//    std::cout << a << std::endl;
//    std::cout << "hello" << std::endl;

//    cv::VideoCapture cap(0);
//    while (1) {
//        cv::Mat frame;
//        cap >> frame;
//        imshow("opencv", frame);
//        cvWaitKey(1);
//    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("traced_mynet_model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    auto image = torch::randn({2, 784});

    // Create a vector of inputs.
    /**
     用来添加元素对应网络模型forward后面的参数
     */
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);

    auto rst = module.forward(inputs).toTensor();

    std::cout << rst.sizes() << std::endl;
    return 0;
}
