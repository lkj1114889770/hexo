---
title: 决策树算法介绍及python实现
date: 2017-08-22 22:13:14
tags:
	- 机器学习
	- python
---
决策树算法是机器学习中经典的算法之一，既可以作为分类算法，也可以作为回归算法。在做开始入门机器学习这方面内容时，自然就接触到了这方面的知识。因此，本文对决策树算法进行了一些整理，首先对决策树算法的原理进行介绍，并且用python对决策树算法进行代码实现。
<!-- more -->

## 决策树算法的概述
决策树算法的思想很类似于我们写代码经常用到的if，else if，else，用什么特征来判断if，这就是决策树算法的精髓所在。
<div align=center>
	<img src="http://i.imgur.com/5xkMuwo.jpg" width="400" height="400" / alt="决策树示例">
</div>
上图是一个结构简单的决策树，用于判断贷款用户是否具备偿款能力，根据不同的条件，从根节点拥有房产开始，根据特征不断判断不断分裂到子节点，叶子节点代表最终的分类结果，树枝分裂的地方就是要根据特征来判断。由此可知，决策树最关键的地方在于树枝分叉的地方，即合理选择特征，从根节点开始，经历子节点的分支后，最终到达叶子。

从上面来说，决策树算法的基本步骤为：
1.从根节点开始
2.遍历所有特征作为一种分裂方式，找到最好的分裂特征
3.分类成两个或者多个节点
4.对子节点同样进行2-3的操作，其实就是第一个递归建树的操作，直到到达叶子节点，即得到分类结果。
常见的决策树算法有ID3、C4.5、CART等，下面将分别进行介绍。

## ID3算法 ##
理论上来说，其实可以有很多种树能够完成这个分类问题，但是如何将这个分类做的比较优呢？有个叫昆兰的大牛将信息论中的熵引入到决策树算法，解决了这个问题。

首先，什么是熵？熵的概念源于在物理的热力学，主要是用于描述一个热力学系统的无序程度。信息论的创始人香农将熵引入信息论中，表示不确定性的度量，直观理解为信息含量越多，熵越大，也就是分布越无序。熵的数值定义为：

![](http://i.imgur.com/CPbE2UB.gif)

X为样例的集合，P(xi)为样例xi的出现概率。
分类特征的选择应该使得分类后更加有序，熵减小，在这里引入了熵的增益这个概念。

![](http://i.imgur.com/z1tT5ej.gif)

其中，Sv为采用特征A分类后的，某一个类别样例数占原有的样例数的比例，H(Sv)该分类后类别的熵，用原来的上减去特征A分类后的每个类别的熵乘于比例权重之后的和。这个时候便得到了选取分类特征的方法，即选取使得熵的增益为最大的特征A作为树枝分裂的特征。ID3就是基于熵增益最大的决策树算法。

### ID3算法计算实例 ###
下面以一个经典的打网球的例子说明如何构建决策树。例子中打网球（play）主要由天气（outlook)、温度（temperature）、湿度（humidity)、风（windy）来确定，样本数据如下：
<div align=center>
	<img src="http://i.imgur.com/i4FloIa.png" width="400" height="270" / alt="样本数据">
</div>
在本例中S有14个样例，目标变量是是否打球，play=yes or no,yes有9个样例，no有5个样例，首先计算从根节点的信息熵：
H(S)=-(9/14*log(9/14))-(5/14*log(5/14))=0.28305
从根节点开始，有四个特征（outlook temperature humidity windy)可以用来进行树枝分裂，首先需要计算这四个特征的信息熵增益，以outlook为例进行计算。

特征A（outlook）有三个不同的取值{sunny,overcast,rainy}，这样将原先的数据集S分为3类，其中
sunnny有5个样本，2个play=yes，3个play=no
overcast有4个样本，4个play=yes
rainy有5个样本，3个play=yes，2个play=no
根据上面的公式，按outlook分类后的熵为：
H（S,A)=5/14sunny熵+4/14overcast熵+5/14rainy熵
&emsp;&emsp;&emsp;&emsp;=5/14*（-2/5*log(2/5)-3/5*log(3/5))+4/14*(-1*log1)+5/14*(-3/5*log(3/5)-2/5*log(2/5))=0.20878

所以对于outlook的信息熵增益为G(S,outlook)=H(S)-H(S,A)=0.07427
用类似的方法，算出，A取其他三个特征时候的信息熵增益分别为：
G(S,temperature)=0.00879
G(S,humidity)=0.04567
G(S,windy)=0.01448
显然，用outlook作为特征A时候，熵增益最大，因此作为第一个节点，即为根节点。这样，S就换分为3个子集，sunny，overcast，rainy，其中overcast熵为0，已经分好类，直接作为叶子节点，而sunny，rainy熵都大于0，采取类似于上述的过程继续选择特征，最后可得决策树为：
<div align=center>
	<img src="http://i.imgur.com/ewGSoY2.png" width="400" height="270" alt="最终决策树">
</div>

### ID3算法的python代码实现
    import pandas as pd
	import math
	
	def Tree_building(dataSet):
	    tree = []
	    if(Calculate_Entropy(dataSet) == 0): #熵为0说明分类已经到达叶子节点
	        if(dataSet['play'].sum()==0):  #根据play的值到达0或者1叶子节点
	            tree.append(0)
	        else:
	            tree.append(1)
	        return tree
	    numSamples=len(dataSet) #样例数
	    Feature_Entropy={} #记录按特征A分类后的熵值的字典
	    for i in range(1,len(dataSet.columns)-1):
	        Set=dict(list(dataSet.groupby(dataSet.columns[i]))) #取出不同的特征
	        Entropy=0.0
	        for key,subSet in Set.items():
	            Entropy+=(len(subSet)/numSamples)*Calculate_Entropy(subSet) #计算熵
	        Feature_Entropy[dataSet.columns[i]]=Entropy
	    
	    #选最小熵值的特征分类点，这样熵值增益最大    
	    Feature = min(zip(Feature_Entropy.values(),Feature_Entropy.keys()))[1] 
	    Set=dict(list(dataSet.groupby(Feature)))
	    for key,value in Set.items():
	        subTree=[]
	        subTree.append(Feature)
	        subTree.append(key)
	        subTree.append(Tree_building(value)) #树枝扩展函数的迭代
	        tree.append(subTree)
	        
	    return tree
	    
	def Calculate_Entropy(data):
	    numSamples=len(data)  #样本总数
	    P=data.sum()['play']  #正例数量
	    N=numSamples-P   #反例数量
	    if((N==0)or(P==0)):  
	        Entropy=0
	        return Entropy
	    Entropy = -P/numSamples*math.log(P/numSamples)-N/numSamples*math.log(N/numSamples)
	    return Entropy
	
	if __name__ == '__main__':
	    data=pd.read_csv('tennis.csv')
	    tree=Tree_building(data)
	    print(tree)

具体代码源文件可见我的[github](https://github.com/lkj1114889770/Machine-Leanring-Algorithm/tree/master/Decision%20Tree)。
## C4.5 和CART算法

### C4.5算法
ID3算法有一个弊端，因为是选择信息增益最大的特征来分裂，所以更偏向于具有大量属性的特征进行分裂，这样做其实有时候是没有意义的，针对此，有了C4.5算法，对ID3进行了改进。C4.5采用信息增益率来选择分裂特征即：
gr(S,A)=gain(S,A)/H(S,A),
其中，gain(S,A)为ID3算法的熵的增益，H(S,A)为取特征为A进行分类的的信息熵
取A为outlook即为H（S,outlook），H(S,A)=H(S,outlook)= -(5/14)*log(5/14) - (5/14)*log(5/14) - (4/14)*log(4/14)

C4.5算法采用的熵信息增益率，因为分母采用了基于属性A分裂后的信息熵，从而抵消了如果信息A属性取值数目过大带来的影响。

C4.5算法还可以应用于处理连续性的属性则按属性A的取值递增排序，将每对相邻值的中点看作可能的分裂点，对每个可能的分裂点，计算：
![](http://i.imgur.com/DWuCTu8.png)

SL和SR分别对应A属性分类点划分出的两个子集，取使得划分后信息熵最小的取值作为A属性的最佳分裂点，参与后面的运算，不够感觉这样计算量有点多的orz

### CART算法
CART算法的划分基于递归建立二叉树，对于一个变量属性来说，它的划分点是一对连续变量属性值的中点。假设m个样本的集合一个属性有m个连续的值，那么则会有m-1个分裂点，每个分裂点为相邻两个连续值的均值。每个属性的划分按照能减少的杂质的量来进行排序，而杂质的减少量定义为划分前的杂质减去划分后的每个节点的杂质量划分所占比率之和。而杂质度量方法常用Gini指标，假设一个样本共有C类，那么一个节点的Gini不纯度可定义为

![](http://i.imgur.com/s2AqUO0.png)

那么，按属性A的某个属性值t分裂最后的Gini值为：

![](https://i.imgur.com/x3qv9ea.png)

分别计算属性A不同属性值的Gini值，取最小的作为A的最佳分类点，然后对于S集此时所有的属性进行上述运算之后，取具有最小的Gini作为分裂属性，其最小的Gini值的属性值作为分裂点。

CART还可以用于作为回归树，但是此时Gini值的算法就不一样，采用的是总方差：

![](http://i.imgur.com/FS7qiGE.png)

回归树的叶节点所含样本中，其输出变量的平均值就是预测结果。



