# MobileNetV1
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result




---

## 简介
MobileNetV1 是由谷歌团队于 2017 年提出的**轻量化深度卷积神经网络**，相关成果发表于《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》。针对 AlexNet、VGG 等传统卷积网络参数量庞大、计算复杂度高、无法在移动端与嵌入式设备部署的痛点，MobileNetV1 从卷积运算底层重构设计逻辑，在精度可控损失的前提下，极大压缩模型体积与计算量，成为轻量化视觉模型的开山之作。其核心创新为**深度可分离卷积**，将标准卷积拆解为深度卷积与逐点卷积分步运算，同时引入**宽度乘数α**与**分辨率乘数ρ**两个缩放超参数，灵活实现模型精度与推理速度的动态权衡。该模型首次系统性解决深度学习在手机、无人机、边缘设备等低算力硬件落地的难题，以极低浮点运算量完成图像分类、目标检测、人脸识别等视觉任务，奠定了移动端轻量化卷积网络的核心设计思想，直接推动后续 MobileNetV2、MobileNetV3、ShuffleNet 等轻量模型的迭代发展。
## 架构
MobileNetV1 整体为**全卷积堆叠+轻量化设计的端到端卷积神经网络**，整体分为「基础卷积初始模块」「深度可分离卷积特征提取模块」和「全局池化+全连接分类模块」三大核心部分，原论文标准输入为224×224分辨率的3通道RGB图像，适配通用图像分类任务，具体结构与核心设计如下：
- **基础初始模块**：网络首层采用标准3×3普通卷积、步长为2，完成原始图像基础浅层特征提取与下采样，搭配BN批量归一化与ReLU激活函数，保证浅层特征表达能力，为后续轻量化卷积做特征预处理。
- **轻量化特征提取模块（核心）**：全程堆叠**深度可分离卷积块**，分为两层串行结构：第一层深度卷积仅在单通道内做空间特征卷积，不进行通道融合；第二层1×1逐点卷积负责跨通道信息交互与通道升降维。网络整体堆叠13组深度可分离卷积，交替设置步长完成下采样，逐步压缩特征图尺寸、提升通道维度，高效提取边缘、纹理、语义等多层级特征，相比传统卷积减少8~9倍计算量。
- **分类输出模块**：后端采用全局平均池化替代冗余全连接降维，压缩高维特征图，减少参数量；末端单层全连接层映射分类维度，原论文ImageNet任务输出1000维类别得分，全程无冗余隐藏层，进一步控制模型体积。

该架构彻底重构传统卷积计算范式，以深度可分离卷积为核心轻量化手段，配合通道缩放、全局池化等优化策略，在保证特征提取能力的同时，实现模型极致轻量化，是边缘端、移动端计算机视觉任务的经典基础骨架网络。


**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要调整下采样步长、防止特征图尺寸过小），但核心架构**深度可分离卷积块堆叠 + 宽度乘数缩放 + 全局平均池化**完全保留，严格复现原版MobileNetV1核心设计思想。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
MobileNetV1 is a lightweight deep convolutional neural network proposed by the Google team in 2017, published in the paper *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. Aiming at the shortcomings of traditional networks such as AlexNet and VGG—huge parameter volume, high computational complexity, and difficulty in deployment on mobile and embedded devices—MobileNetV1 reconstructs the convolution operation logic at the underlying level. It greatly reduces model size and computational cost with controllable accuracy loss, becoming the pioneering lightweight vision model. Its core innovation is **depthwise separable convolution**, which splits standard convolution into depthwise convolution and pointwise convolution. Meanwhile, two scaling hyperparameters, width multiplier α and resolution multiplier ρ, are introduced to flexibly balance model accuracy and inference speed. It systematically solves the problem of deploying deep learning on low-computing hardware such as mobile phones and edge devices, laying the core foundation for mobile lightweight CNNs and promoting the iterative development of subsequent lightweight models.

## Architecture
The overall structure of MobileNetV1 is an end-to-end convolutional neural network based on full convolution stacking and lightweight optimization. It is divided into three core parts: the initial basic convolution module, the depthwise separable convolution feature extraction module, and the global pooling & fully connected classification module. The original paper adopts 224×224 RGB images as standard input for general image classification tasks.
- **Initial Basic Module**: The first layer adopts standard 3×3 ordinary convolution with stride=2, completing shallow feature extraction and downsampling, combined with BN and ReLU activation to ensure basic feature expression.
- **Lightweight Feature Extraction Module (Core)**: Thirteen groups of depthwise separable convolution blocks are stacked throughout the network. Depthwise convolution extracts spatial features within a single channel, and 1×1 pointwise convolution realizes cross-channel information fusion and channel dimension adjustment. Alternate downsampling is completed by adjusting stride to extract multi-scale semantic features with extremely low computational consumption.
- **Classification Output Module**: Global average pooling is used to reduce dimension and compress high-dimensional feature maps, effectively reducing redundant parameters. The final fully connected layer maps to the classification dimension and outputs category prediction scores, maintaining a concise and efficient output structure.

**Note:** We use the CIFAR-10 dataset with 10 classification categories. Since the 32×32 image size of CIFAR-10 is much smaller than the 224×224 input in the original paper, slight adjustments are made to the downsampling stride to avoid excessive feature compression. However, the core design of **depthwise separable convolution, width multiplier and global average pooling** is completely consistent with the original MobileNetV1.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

---
## 原文章 | Original article
Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[EB/OL]. arXiv preprint arXiv:1704.04861, 2017.
