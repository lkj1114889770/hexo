---
title: 数据处理—Encoding Categorical Value
date: 2017-08-30 15:41:09
tags:
	- 机器学习
---

在机器学习的数据处理中，常常有些特征的值为类型变量（categorical variable），即这些特征对应的值是一些文本，为了便于后期的模型建立，常常将这些文本属性的值转换成数值，即对categorical variable的编码处理。
<!-- more -->

## Data Set ##

数据集假设为：

    import pandas as pd  
	df = pd.DataFrame([  
	            ['green', 'M', 11],   
	            ['red', 'L', 22],   
	            ['blue', 'XL', 33]])  
	  
	df.columns = ['color', 'size', 'prize'] 

>            color size  prize 
>	      0  green    M   11     		
>	      1    red    L   22  
>	      2   blue   XL   33

## Replace ##
一种方法是使用DataFrame的自带函数功能，替换函数replace。

	encoding_num={"color":{"green":0,"red":1,"blue":2},"size":{"M":0,"L":1,"XL":2}}
	df.replace(encoding_num,inplace=True)

	df
	Out: 
   	color  size  prize
	0      0     0   10.1
	1      1     1   13.5
	2      2     2   15.3

## One Hot Encoding ##
One-hot Encoding，又称为一位有效编码，即对每个状态采取一位进行编码。假设某个特征有N个特征值，则需要N位进行编码，且任意时候只有一位有效。

	pd.get_dummies(df)
	Out[28]: 
	   prize  color_blue  color_green  color_red  size_L  size_M  size_XL
	0     11           0            1          0       0       1        0
	1     22           0            0          1       1       0        0
	2     33           1            0          0       0       0        1

这种编码方式弊端就是，当某个特征对应特征值很多时，就需要很多位进行编码，使得数据列数过大。

## Label Encoding ##
这种编码方式，主要是基于pandas的Categorical模块，将某一列转换成category类型，然后使用category value来编码。

	df["color"]=df["color"].astype('category')
	df["size"]=df["size"].astype('category')
	df["color"]=df["color"].cat.codes
	df["size"]=df["size"].cat.codes
	
	df
	Out[37]: 
	   color  size  prize
	0      1     1     11
	1      2     0     22
	2      0     2     33

