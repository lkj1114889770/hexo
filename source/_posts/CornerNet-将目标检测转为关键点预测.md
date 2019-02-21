---
title: 'CornerNet:将目标检测转为关键点预测'
date: 2018-09-26 14:22:54
tags:
	- 深度学习
	- 目标检测
	- CV
---
论文链接：[https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)
代码链接：[https://github.com/umich-vl/CornerNet
](https://github.com/umich-vl/CornerNet)

CornerNet是ECCV2018上的一篇文章，与以往的Anchor机制的目标检测方法不同，这篇文章借鉴了人体关键点检测的思路，将目标检测转为关键点检测（Dectecting Objects as Paired Keypoints），是一种不一样的新思路，阅读了这篇文章，做个笔记。
<!-- more -->

## Introduction
目前目标检测算法的主要思路还是设置大量的Anchor作为预选框，通过训练的方式获取最后的bounding box，这样就带来两个问题：
1. 大量的Anchor只有少部分和gt有比较大的overlap，从而带来正负样本巨大的不均衡的问题，减慢训练过程
2. Anchor的设置本身也是需要超参数的(形状、个数怎么设置)，在multi-scale的时候会更加明显。

作者因此提出了一种新的one-stage解决方法，将目标检测转为一堆关键点检测，the top-left corner 和bottom-right corner，使用卷积神经网络来为一个类别预测heatmap获取top-left corners,同样预测另一个heatmap获取bottom-right corners,还预测embedding vector对顶点进行分组，确定是否属于同一个目标，如下图所示。

![](/img/CornerNet/figure_1.png)

另一个创新点是提出了corner pooling，一种为了更好地获取corner的新的pooling layer。以top-left corner pooling 为例，如下图所示对每个channel，分别提取特征图的水平和垂直方向的最大值，然后求和。

![](/img/CornerNet/figure_2.png)

论文认为corner pooling之所以有效，是因为（1）目标定位框的中心难以确定，和边界框的4条边相关，但是每个顶点只与边界框的两条边相关，所以corner 更容易提取。（2）顶点更有效提供离散的边界空间，实用O(wh)顶点可以表示O(w2h2) anchor boxes。

## CornetNet
CornerNet使用CNNs来预测两组heatmaps为每个物品类别来表示corner的位置，使用embedding vector来表示corner是否属于同一个物品，同时为了产生更加紧密的bounding box，也预测了offset。通过heatmap, embedding vector，offsets,通过后处理的方法就可以获得最后的bounding box。作者提出的算法总体框架如下图所示：

![](/img/CornerNet/figure_3.png)

使用了hourglass network作为backbone network，紧接的两个模块分别用于预测top-left corners和bottom-right corners，每一个模块有独立的corner pooling，然后得到heatmaps, embeddings, offsets.

### Detecting Corners
论文预测了两组heatmap，每一个heatmap包含C channels(C是目标类别，不包括background),每一个channel是二进制mask，表示相应的corner位置。

![](/img/CornerNet/figure_4.png)


对于每个顶点，只有一个groun truth，其他位置都是负样本。在训练过程中为了减少负样本数量，在每个gt顶点设定的半径r区域内都是正样本，如上图所示，半径r的确定根据所学的Iou决定。使用unnormalized 2D Gaussian来减少的半径r范围内的loss，基本思想就是构造高斯函数，中心就是gt位置，离这个中心越远衰减得越厉害，即：

![](/img/CornerNet/figure_5.png)

pcij表示类别为c，坐标是（i,j）的预测热点图，ycij表示相应位置的ground-truth，是经过2D Gaussian的输出值，用来调整权值，论文提出变体Focal loss表示检测目标的损失函数：

![](/img/CornerNet/figure_6.png)

由于采样过程中的量化带来的misaligment，预测offset来调整corner的位置：

![](/img/CornerNet/figure_7.png)

训练中用smooth L1 Loss来计算：

![](/img/CornerNet/figure_8.png)

### Grouping Corner
这个部分是用来决定一对corner是否来自同一object。具体的做法就是对卷积特征进行embedding（1x1的conv），得到corner的embedding vector，我们希望同属于同一个object的一对 corner的距离尽可能小，不属于的距离尽可能大！所以有两个loss，push loss和pull loss，从名字上来说，pull吧同一个目标的corner拉近，push把不同目标的推远。

![](/img/CornerNet/figure_9.png)

etk,elk分别是属于 top-left corner和botto-right corner的embedding，ek是他们的平均值，△在论文中设置为1.

### Corner Pooling

![](/img/CornerNet/figure_10.png)

为了检测某一个点是否是corner，需要从行和列分别检查，以top-left这个点为例，计算过程分为三部分：
1. 从上到下做max pooling
2. 从右到左做max pooling
3. 然后合并（相加）
如下图中，从下往上计算，每一列都能得到一个单调非递减的结果，相当于对corner的先验做了编码。对于object来说，如果要去找最上边的位置，需要从下到上检查这一列的最大值，最大值的位置是corner的可能存在的位置。

![](/img/CornerNet/figure_11.png)

实际计算公式为：


![](/img/CornerNet/figure_12.png)

这样整个预测框架如下图所示：


![](/img/CornerNet/figure_13.png)

### Predict details
1. 在corner heatmap上用3x3的max poolings做NMS，选择top 100的top-left和top 100的bottom-right
2. 通过预测的offset来调整位置
3. 计算top-left和bottom-right的embedding 的L1 distance，筛掉距离大于0.5或者是不属于同一类别的一对corner。
4. 计算top-left和bottom-right的score的平均值作为最终的score。