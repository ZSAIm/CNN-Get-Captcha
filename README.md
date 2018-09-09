# CaptchaReconition-CNN
### Base on 
* __CNN__ 
* __tensorflow__
* __python 3.5__

# 基于卷积神经网络的验证码识别

## 说明
* 由于本程序使用模拟方式生成验证码数据集，最终所得预测准确率不会很高（取决于验证码的模拟程度）。
* 本项目的模型识别准确率大概能达到``80%``。
* 本项目例子训练总用时不到一个小时。(GPU - GT940M)

## 目标验证码
![target-0](https://github.com/ZSAIm/CaptchaReconition-CNN/blob/master/images/target-captcha-0.gif)
![target-1](https://github.com/ZSAIm/CaptchaReconition-CNN/blob/master/images/target-captcha-1.gif)
![target-2](https://github.com/ZSAIm/CaptchaReconition-CNN/blob/master/images/target-captcha-2.gif)

## 项目包含
* __``/model/*``__	: 经过训练的模型，其中包含两个模型。
* __``/dataset/*``__ : 训练集存放目录。
* __``arialbd.ttf``__ : 模拟训练集所用字体。
* __``target-captcha.gif``__ : 目标验证码。
* __``generate_dataset.py``__ : 生成模拟验证码训练集，并转存到TFRecord。
* __``img_process.py``__ : 简单图像处理函数。
* __``batching.py``__ : batch数据并输出作为训练数据。
* __``train.py``__ : 训练和增强训练的操作。
* __``inference.py``__ : 卷积神经网络的结构。
* __``constant.py``__ : 共用常量（方便调整）。

## 安装模块
* __``tensorflow``__ : pip install tensorflow
* __``matplotlib``__ : pip install matplotlib
* __``numpy``__ : pip install numpy
* __``PIL``__ : pip install pillow

## 引用项目
无

## 建议

### 如需进一步提高识别率，切合目标验证码，可通过一下方式实现。
1. 人工标记一定数量的训练集，然后再稍稍提高学习率进行训练（不要训练过猛，避免过拟合）。
2. 适当增加神经网路的复杂度，并适当修改验证码生成参数。

## LICENSE
Apache-2.0
