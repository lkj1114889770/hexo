---
title: PCL的python库安装for Ubuntu16.04
date: 2018-05-11 19:43:04
tags:
---

PCL（Point Cloud Library）是包含各种点云算法的大型跨平台开源C++编程库，是吸纳了大量点云相关算法，包括点云获取、滤波、分割、特征提取、曲面重建以及可视化等各种算法，然而现在我主要使用的是python语言，网上目前又有公布的python_pcl实现库[python_pcl实现库](https://github.com/strawlab/python-pcl) ，然而针对Ubuntu16.04按照官方给的方法没有能够实现安装，踩了无数坑之后，博客记录一种简单且成功安装的方法。

<!-- more -->

## PCL安装
不用编译源码，一行命令直接apt安装，顺带安装各种依赖的乱七八糟的库
	
	sudo apt-get install libpcl-dev 

再安装一些pcl可视化等软件包

	sudo apt-get install pcl_tools


## 安装 python_pcl
首先下载python_pcl源文件

	git clone https://github.com/strawlab/python-pcl.git
	
编译、安装

	python setup.py build_ext -i
	python setup.py install

在此之前常出现的一个编译问题是cython版本问题，所以在执行上一步之前首先：

	pip install cython==0.25.2
	
## 解决常出现的链接失败的问题
由于我的默认python为anaconda3的python，可能是anaconda3自带的链接库的问题，所以出现了如下错误：

	./lib/libgomp.so.1: version `GOMP_4.0' not found (required by /home/lkj/anaconda3/lib/python3.5/site-packages/xgboost-0.6-py3.5.egg/xgboost/libxgboost.so) 
	
上面的意思是anaconda3/lib/libgomp.so.1中没有‘GOMP_4.0'，这个可以使用strings命令查看libgomp.so.1这个文件，显示并无4.0版本，因此寻找其他路径的链接库替代，用locate命令搜索系统中所有的libgomp.so.1，得到：
![](https://i.imgur.com/tSz0fdU.png) 
然后用strings查看这些文件信息，

	/usr/lib/x86_64-linux-gnu/libgomp.so.1 |grep GOMP

发现x86_64-linux-gnu/libgomp.so.1包含GOMP_4.0
![](https://i.imgur.com/Hi1QcUV.png) 
因此可以删掉原有的libgomp.so.1，重新做一个新的链接。

	ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 libgomp.so.1 
	
然后再次在python里面import pcl,又提示libstdc++.so.6出现类似的问题，对上述做类似处理，如果还有链接库的问题，也可以用同样的方法处理,至此实现了python的pcl库安装。