---
title: K-means Clustering
date: 2017-09-22 14:56:04
tags:
	- 机器学习
	- 无监督学习
---
分类通常是一种监督式学习算法，事先都知道标签信息，但是实际上很多情况下都不知道标签信息，这个时候就经常用到聚类算法（Clustering），属于无监督学习的一种，本文介绍无监督学习中典型的一种k-means聚类算法。
<!-- more -->

## conventional k-means
k-means聚类算法的思想很简单，就是将数据相似度最大的聚集在一起为一类，如何衡量数据之间的相似度，通常用欧几里得距离来表示：

![](https://i.imgur.com/lhYz2GX.png)

有时候也用余弦向量来度量：

![](https://i.imgur.com/6DqsyIq.png)

算法过程也比较简单：
1. 从数据集D中随机取k个元素，作为初始k个簇的中心
2. 计算剩下的元素到k个中心的距离，并将其归类到离自己最近的簇中心点对应的簇
3. 重新计算每个簇的中心，采用簇中所有元素各个维度的算数平均数
4. 若新的簇的中心不变，或者在变化阈值内，则聚类结束，否则重新回到第2步。

## k-means++
k-means算法的一个弊端就是，初始选取的随机k个中心会对最终实际聚类效果影响很大，基于此对于初始k个点选取进行了改进，即k-measn++算法。
基本思想就是，选取的初始k个点的距离尽可能远。
1. 首先随机选择一个点作为第一个簇中心点
2. 计算其余点与最近的一个簇中心点的距离D(x)保存在一个数组，并累加得到和sum（D(X))。
3. 再在（0，1）取一个随机值Random，sum*Random对应的D(x)区间即为选中的下一个聚类中心（因为D(X)越大，被选中的概率越大）。
4. 重复2 3步骤知直到k个初始聚类中心都被找出来
5. 再进行上面的k-means聚类算法。

## Kernel k-means
当数据无法线性可分的时候，k-means算法也无法进行分类，类似于SVM，将分类空间推广到更广义的度量空间，即为kernel k-means.

![](https://i.imgur.com/FkIdHw1.jpg)

将点从原来的空间映射到更高维度的特征空间，则距离公式变成：

![](https://i.imgur.com/QSX5K1g.png)

常见的核函数有：
Linear Kernel:

![](https://i.imgur.com/Pamhtvp.png)

Polynomial Kernel:

![](https://i.imgur.com/8c4UlBi.png)

Gaussian Kernel：

![](https://i.imgur.com/Q0fK08M.png)


