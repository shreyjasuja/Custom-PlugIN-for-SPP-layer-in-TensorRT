# Custom PlugIN for SPP layer in TensorRT

## Brief Summary


1. NVIDIA TensorRT™ is an SDK for high-performance deep learning inference.
2. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications
3. TensorRT-based applications perform up to 40 times faster than CPU-only platforms during inference. 
4. With TensorRT, we can optimize neural network models trained in all major frameworks, calibrate for lower precision with high accuracy, and finally deploy to hyperscale data centers, embedded, or automotive product platforms.

<p align="center">
  <img src="trt-info.png">
</p>

### Custom Plug-In 

1. Custom plugins(layers) extends the functionality of Tensorrt layers with specific needs which are not supported.
2. Implemented using the IPluginV2Ext class in C++ and Python API.
3. A custom layer is implemented by extending the IPluginCreator class and one of TensorRT’s base classes for plugins.
4. We will be extending IPluginV2Ext base class on Spatial Pyramid Pooling
layer of deep neural networks.

### Spatial Pyramid Pooling Layer

Existing deep convolutional neural networks (CNNs) require a fixed-size (e.g. 64X64) input image. This requirement may decrease the recognition accuracy for the images or sub-images of an arbitrary size/scale.
Using SPP is more significant in CNNs. Using SPP , we compute the feature maps from the entire image only once, and then pool features in arbitrary regions (sub-images) to generate fixed-length representations for training the detectors.
The layer is used between convolution and and fully connected layers so that we can generate same length output after convolution layers for FC layers.

### Plug-In Files

Find the plugin files for SPP layers as:
SPP_Pool.h
SPP_Pool.cpp

### References


[1]   NVIDIA TensorRT  Developer Guide 
Available:[Click Here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) 
Access date: 10-12-2019.
[2] Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Microsoft Research, China ,Xi’an Jiaotong University, China, University of Science and Technology of China ] Access date: 20-12-2019


