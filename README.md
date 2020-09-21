
<h4 align="center">A CMake based integration of the libpng / png++ libraries with the Libtorch C++ Deep Learning Library</h4>
      
<p align="center">
  <a href="#about">About</a> •
  <a href="#credits">Credits</a> •
  <a href="#installation">Installation</a> •  
  <a href="#fexamples">Examples</a> •  
  <a href="#author">Author</a> •  
  <a href="#license">License</a>
</p>

<h1 align="center">  
  <img src="https://github.com/QuantScientist/PngTorch/blob/master/asstes/logo.png?raw=true"></a>
</h1>


---

## About

<table>
<tr>
<td>
  
**PngTorch++** is an **integration** of the well-known  **_libpng_** (https://github.com/libpng) library 
and my favourite Deep Learning Library Libtorch: the **_PyTorch_** C++ frontend.
In many occations, one wants to avoid using OpenCV, just because of the large overhead. 
This was the case when I started writing: https://github.com/QuantScientist/Siv3DTorch 

For motivation, see: 
https://github.com/pytorch/vision/issues/2691
https://github.com/koba-jon/pytorch_cpp/issues/6
 
By including a single header file, `#include <torch/script.h>`, the integration allows one to easily 
read images and convert them into PyTorch C++ front-end tensors and vice versa.  


<p align="right">
<sub>(Preview)</sub>
</p>

</td>
</tr>
</table>

## Credits 
* PyTorch CPP examples by koba-jon https://github.com/koba-jon/pytorch_cpp.
 
* PyTorch CPP examples + CMake build: https://github.com/prabhuomkar/pytorch-cpp/

## A simple example 
The folowing example reads a PNG from teh file system, converts it into a `torch::tensor` and then saves 
the tensor as an PNG image on the file system.  
 

Full source code:

```cpp

```


## Features

|                            | 🔰 PngTorch++ CMake  | |
| -------------------------- | :----------------: | :-------------:|
| PyTorch CPU tensor to PNG        |         ✔️                 
| PyTorch GPU tensors to PNG       |         ✔️                 
| Libtorch C++ 1.6           |         ✔️                 


## Examples

* A Simple example, mainly for testing the integration. Allocates a tensor on the GPU.

![PngTorch++ Code](https://github.com/QuantScientist/PngTorch/blob/master/assets/simple001.gif?raw=true)
 
* Load a trained PyTorch model in C++ (**see pth folder**), load an Image in C++, run a trained pytorch model on it and save the output.
 


## Requirements:
* Windows 10 and Microsoft Visual C++ 2019 16.4, Linux is not supported at the moment.
* NVIDIA CUDA 10.2. I did not test with any other CUDA version. 
* PyTorch / LibTorch c++ version 1.6.  
* 64 bit only.  
* CMake 3.18  
* libpng, png++ 

Please setup CLion as follows:
![PngTorch++ Code](https://github.com/QuantScientist/PngTorch/blob/master/assets/clion.png?raw=true)

## Installation 

#### Downloading and installing steps LIBTORCH C++:
* **[Download]()** the latest version of Libtorch for Windows here: https://pytorch.org/.
![PngTorch++ Code](https://github.com/QuantScientist/PngTorch/blob/master/assets/libtorch16.png?raw=true)

* **Go** to the following path: `mysiv3dproject/`
* Place the **LiBtorch ZIP** folder (from .zip) inside the **project** folder as follows `mydproject/_deps/libtorch/`:

The **CMake file will download this automatically for you**. Note: only a GPU is supported.  
Credits: https://github.com/prabhuomkar/pytorch-cpp/
  

#### Downloading and installing steps lippng:
* **[Download]()** 
* Under the lib directory,I included the lib file for PNG and ZLIB for windows, 
the CMake file will link against them during runtime.
   
```cmake
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES}  ${OpenCV_LIBRARIES} "${CMAKE_CURRENT_LIST_DIR}/lib/libpng/libpng16.lib" "${CMAKE_CURRENT_LIST_DIR}/lib/zlib/zlib.lib")
```
 

## Inference
For inference, you have to copy all the **Libtorch DLLs** to the location of the executable file. For instance:
![PngTorch++ Code](https://github.com/QuantScientist/PngTorch/blob/master/assets/vc-inference.png?raw=true)

This is **done automatically** for you in the CMake file. 
 
## Contributing

Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.

## Author
Shlomo Kashani, Author of the book _Deep Learning Interviews_ www.interviews.ai: entropy@interviews.ai 

## Citation

If you find the code or trained models useful, please consider citing:

```
@misc{PngTorch++,
  author={Kashani, Shlomo},
  title={PngTorch++2020},
  howpublished={\url{https://github.com/QuantScientist/PngTorch/}},
  year={2020}
}
```

## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-orange.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- Copyright © [Shlomo](https://github.com/QuantScientist/).

# References
- https://github.com/koba-jon/pytorch_cpp 
- https://www.jianshu.com/p/6fe9214431c6
- https://github.com/lsrock1/maskrcnn_benchmark.cpp
- https://gist.github.com/Con-Mi/4d92af62adb784a5353ff7cf19d6d099
- https://lernapparat.de/pytorch-traceable-differentiable/
- http://lernapparat.de/static/artikel/pytorch-jit-android/thomas_viehmann.pytorch_jit_android_2018-12-11.pdf
- https://github.com/walktree/libtorch-yolov3
