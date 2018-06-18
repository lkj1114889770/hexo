---
title: '基于深度学习的目标检测技术学习笔记(R-CNN系列)'
date: 2018-04-20 11:03:17
tags:
	- 深度学习
	- 目标检测
---
图像的目标检测（object detection）主要包括两个任务，一是要标注出目标物体的位置（localization），而是要识别出目标物体的类别（classification）。通俗来说，就是解决图像中多个目标在哪里，是什么的一个问题。这个问题的涉及，主要是目前参加了天池大赛的一个目标识别方面的问题，所以阅读了一些相关方面的文献，在此做一个学习总结，主要来介绍R-CNN（Regions with CNN features）系列的算法。
<!-- more -->

传统的目标检测算法一般是基于滑动窗口选中图中的某一部分作为候选区域，然后提取候选区域的特征，利用分类器（如常见的SVM)进行识别。2014年提出的region proposal+CNN代替传统目标检测使用的滑动窗口+特征工程的方法，设计了R-CNN算法，开启了基于深度学习的目标检测的大门。
## R-CNN
![](https://i.imgur.com/2YwfkRz.png)

R-CNN算法流程为：

1. 输入图像，根据SS（selective search）算法提取2000个左右的region proposal（候选框）
2. 将候选框crop/wrap为固定大小后输入CNN中，得到固定维度的输出特征
3. 对提取的CNN特征，利用SVM分类器分类得到对应类别
4. 边界回归（bouding-box regression），用线性回归模型修正候选框的位置

R-CNN使得识别的精度和速度都有了提升，但是也存在很大问题，每次候选框都需要经过CNN操作，计算量很大，有很多重复计算；训练步骤繁多。

## Fast R-CNN
R-CNN需要每次将候选框resize到固定大小作为CNN输入，这样有很多重复计算。SPP-net的主要思想是去掉了原始图像上的crop/warp等操作，换成了在卷积特征上的空间金字塔池化层（Spatial Pyramid Pooling，SPP）。

![](https://i.imgur.com/21xD4eJ.png)

SPP Net对整幅图像只进行一次CNN操作得到特征图，这样原图中的每一个候选框都对应于特征图上大小不同的某一区域，通过SPP可以将这些不同大小的区域映射为相同的维度，作为之后的输入，这样就能保证只进行一次CNN操作了。SPP包含一种可伸缩的池化层，输出固定尺寸特征。

基于SPP的思想，Fast R-CNN加入了一个ROI Pooling，将不同大小输入映射到一个固定大小的输出。R-CNN之前的操作是目标识别（classification）以及边界回归（bouding-box regression）分开进行。Fast R-CNN做的改进就是将这两个过程合并在一起，这两个任务共享CNN特征图，即成为了一个multi-task模型。

![](https://i.imgur.com/31zXWV3.png)

多任务自然对应multi-loss，损失函数包括分类误差以及边框回归误差。
L*cls*为分类误差：

![](https://i.imgur.com/MR7Wgbl.png)

分类误差只考虑对应的类别被正确分类到的概率，即P*l*为label对应的概率，当P*l*=1时，Loss为0，即正确分类的概率越大，loss越小。

L*reg*为边框回归误差：

![](https://i.imgur.com/3zsjUKR.png)

对预测的边框四个位置描述参数与真实分类对应边框的四个参数偏差进行评估作为损失函数，g函数为smooth L1函数，这样对于噪声点不敏感，鲁棒性强，在|x|>1时，变为线性，降低噪声影响。

![](https://i.imgur.com/plKB3T1.png)

![](https://i.imgur.com/UEijJWR.png)

这样加权得到的最终损失函数为：

![](https://i.imgur.com/rcLlJWB.png)

foreground理解为前景，即对应有目标物体，这个时候需要考虑边框回归误差；background为背景，没有包含目标物品，所以不需考虑边框回归误差。

## Faster R-CNN
Faster R-CNN对Fast R-CNN又进行了改进，使得Faster。主要是将候选框的选取也引入到网络中，代替了之前SS选取候选框的方式，即引入了RPN（Region Proposal Network），将找候选框的工作也交给了神经网络了。

![](https://i.imgur.com/Aso7UhH.png)

提到RPN网络，就不能不说anchors，即锚点，对应的是一组矩形框，在时间中有3种形状width:height = [1:1, 1:2, 2:1]，对应3中尺寸，多以共计9个矩形框。

![](https://i.imgur.com/LyWLO9K.jpg)

这个矩形框对应的是原始输入图像里面的，并非是卷积特征图上的。即对卷积特征图生每一个点，可以对应原始图上的一个anchors，为其配备9个框作为原始检测框，当然一开始肯定四不准确的，可以在后续的bounding box regression可以修正检测框位置。

![](https://i.imgur.com/9PmRpnN.png)

为了生成区域建议框，在最后一个共享的卷积层输出的卷积特征映射上滑动小网络，这个网络全连接到输入卷积特征映射的nxn的空间窗口上。每个滑动窗口映射到一个低维向量上（对于ZF是256-d，对于VGG是512-d，每个特征映射的一个滑动窗口对应一个数值）。这个向量输出给两个同级的全连接的层——包围盒回归层（reg）和包围盒分类层（cls）。论文中n=3由于小网络是滑动窗口的形式，所以全连接的层（nxn的）被所有空间位置共享（指所有位置用来计算内积的nxn的层参数相同）。这种结构实现为nxn的卷积层，后接两个同级的1x1的卷积层（分别对应reg和cls）。 
在每一个滑动窗口的位置，我们同时预测k个区域建议，所以reg层有4k个输出，即k个box的坐标编码。cls层输出2k个得分，即对每个建议框是目标/非目标的估计概率（为简单起见，是用二类的softmax层实现的cls层，还可以用logistic回归来生成k个得分）。k个建议框被相应的k个称为anchor的box参数化。每个anchor以当前滑动窗口中心为中心，并对应一种尺度和长宽比，我们使用3种尺度和3种长宽比，这样在每一个滑动位置就有k=9个anchor。对于大小为WxH（典型值约2,400）的卷积特征映射，总共有WHk个anchor。

Faster R-CNN的损失函数为：

![](https://i.imgur.com/ytSoqQU.png)

这里，i是一个mini-batch中anchor的索引，Pi是anchor i是目标的预测概率。如果anchor为正，Pi* 就是1，如果anchor为负，Pi* 就是0。ti是一个向量，表示预测的包围盒的4个参数化坐标，ti* 是与正anchor对应的GT（groundtruth）包围盒的坐标向量。Pi* L*reg*这一项意味着只有正anchor（Pi* =1）才有回归损失，其他情况就没有（Pi* =0）。cls层和reg层的输出分别由{pi}和{ti}组成，这两项分别由N*cls*和N*reg*以及一个平衡权重λ归一化。
边框回归损失函数，用采取类似fast R-CNN介绍的方法。具体地，学习的时候，对于四个参数进行如下处理：

![](https://i.imgur.com/w449Eaq.png)

x，y，w，h指的是包围盒中心的（x,y）坐标、宽、高。变量x，xa，x* 分别指预测的包围盒、anchor的包围盒、GT的包围盒（对y，w，h也是一样）的x坐标，可以理解为从anchor包围盒到附近的GT包围盒的包围盒回归。

Fast R-CNN训练依赖于固定的目标建议框，而Faster R-CNN中的卷积层是共享的，所以RPN和Fast R-CNN都不能独立训练，论文中提出的是4步训练算法，通过交替优化来学习共享的特征。 


1.  训练RPN，该网络用ImageNet预训练的模型初始化，并端到端微调用于区域建议任务。
2.  利用第一步的RPN生成的建议框，由Fast R-CNN训练一个单独的检测网络，这个检测网络同样是由ImageNet预训练的模型初始化的，这时候两个网络还没有共享卷积层。
3.  用检测网络初始化RPN训练，但我们固定共享的卷积层，并且只微调RPN独有的层，现在两个网络共享卷积层了。
4.  保持共享的卷积层固定，微调Fast R-CNN的fc层。这样，两个网络共享相同的卷积层，构成一个统一的网络。