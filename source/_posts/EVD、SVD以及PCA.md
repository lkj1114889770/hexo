---
title: EVD、SVD以及PCA整理
date: 2017-11-01 14:40:14
tags:
	- 机器学习
---
最近的机器学习算法学习到了主成分分析（PCA），在人脸识别中对样本数据进行了降维，借此对特征值分解（EVD）、奇异值分解（SVD）进行了梳理整理。

## 特征值分解（EVD)

矩阵是一种线性变换，比如矩阵Ax=y中，矩阵A将向量x线性变换到另一个矩阵y，这个过程中包含3类效应：旋转、缩放以及投影。
<!-- more -->

对角矩阵对应缩放，比如![](https://i.imgur.com/s5t13lw.png)

其对应的线性变换如下：

<img alt="" src="http://img.blog.csdn.net/20140118132422328">

对与正交矩阵来说，对应的是向量的旋转，比如将向量OA从正交基e1e2中,到另一组正交基为e1'e2'中，

![](https://i.imgur.com/GMLZwPD.png)
<img alt="" src="http://img.blog.csdn.net/20150123124108372?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhvbmdrZWppbmd3YW5n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast">

当矩阵A与x维度不一样时，得到的y的维度也与x不一样，即存在投影变换。

考虑一种特殊矩阵，对称阵的特征值分解，其实在机器学习中也经常是对XX'求特征向量，也就是对称阵。

![](https://i.imgur.com/LLiUV34.png)

其中：

![](https://i.imgur.com/2SkuNLs.png)

![](https://i.imgur.com/5abYT6K.png)

这个时候用到对称阵的特性，U为正交矩阵，其逆矩阵等于转置。

![](https://i.imgur.com/Jstukbo.png)

即矩阵A将向量X转移到了U这组基的空间上，再进行缩放，而后又通过U正交基进行旋转，所以只有缩放，没有旋转和投影。

## 奇异值分解（SVD)

奇异值分解其实类似于特征值分解，不过奇异值分解适用于更一般的矩阵，而不是方阵。

<div align=center>
	<img src="https://i.imgur.com/Gp5sZub.png" width="250" height="50">
</div>

U、V都是一组正交基，表示一个向量从V这组正交基旋转到U这组正交基，同时也在每个方向进行缩放。

奇异值和特征值对应为：

![](https://i.imgur.com/FbOWs0v.png)

v即为式子中的右奇异向量，同时也可以得到：

![](https://i.imgur.com/TUH2piG.png)

在奇异值按从小到大排序的情况下，很多情况下，前面部分的奇异值就占所有奇异值和的99%以上，所以我们可以取前r个奇异值来近似描述矩阵，可以用来数据降维。
<div align=center>
	<img src="https://i.imgur.com/kckZiod.png" width="250" height="50">
</div>
这样可以还原出A矩阵，减少数据存储。
下面看如何利用SVD降维：
<div align=center>
	<img src="https://i.imgur.com/fKT2IuH.png" width="300" height="200">
</div>
从而将A'从n * m降到n * r

SVD常用于推荐系统，有基于用户的协同过滤（User CF)和基于物品的协同过滤（Item CF)。这里给出一个Item CF实现。

网上找的一个实现代码，找不到出处了。。。

	#coding=utf-8
	from numpy import *
	from numpy import linalg as la
	
	'''加载测试数据集'''
	def loadExData():
	    return mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
	           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
	           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
	           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
	           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
	           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
	           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
	           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
	           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
	           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
	           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
	
	'''以下是三种计算相似度的算法，分别是欧式距离、皮尔逊相关系数和余弦相似度,
	注意三种计算方式的参数inA和inB都是列向量'''
	def ecludSim(inA,inB):
	    return 1.0/(1.0+la.norm(inA-inB))  #范数的计算方法linalg.norm()，这里的1/(1+距离)表示将相似度的范围放在0与1之间
	
	def pearsSim(inA,inB):
	    if len(inA)<3: return 1.0
	    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]  #皮尔逊相关系数的计算方法corrcoef()，参数rowvar=0表示对列求相似度，这里的0.5+0.5*corrcoef()是为了将范围归一化放到0和1之间
	
	def cosSim(inA,inB):
	    num=float(inA.T*inB)
	    denom=la.norm(inA)*la.norm(inB)
	    return 0.5+0.5*(num/denom) #将相似度归一到0与1之间
	
	'''按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
	后续计算SVD时需要将原始矩阵转换到k维空间'''
	def sigmaPct(sigma,percentage):
	    sigma2=sigma**2 #对sigma求平方
	    sumsgm2=sum(sigma2) #求所有奇异值sigma的平方和
	    sumsgm3=0 #sumsgm3是前k个奇异值的平方和
	    k=0
	    for i in sigma:
	        sumsgm3+=i**2
	        k+=1
	        if sumsgm3>=sumsgm2*percentage:
	            return k
	
	'''函数svdEst()的参数包含：数据矩阵、用户编号、物品编号和奇异值占比的阈值，
	数据矩阵的行对应用户，列对应物品，函数的作用是基于item的相似性对用户未评过分的物品进行预测评分'''
	def svdEst(dataMat,user,simMeas,item,percentage):
	    n=shape(dataMat)[1]
	    simTotal=0.0;ratSimTotal=0.0
	    u,sigma,vt=la.svd(dataMat)
	    k=sigmaPct(sigma,percentage) #确定了k的值
	    sigmaK=mat(eye(k)*sigma[:k])  #构建对角矩阵
	    xformedItems=dataMat.T*u[:,:k]*sigmaK.I  #根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
	    for j in range(n):
	        userRating=dataMat[user,j]
	        if userRating==0 or j==item:continue
	        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T) #计算物品item与物品j之间的相似度
	        simTotal+=similarity #对所有相似度求和
	        ratSimTotal+=similarity*userRating #用"物品item和物品j的相似度"乘以"用户对物品j的评分"，并求和
	    if simTotal==0:return 0
	    else:return ratSimTotal/simTotal #得到对物品item的预测评分
	
	'''函数recommend()产生预测评分最高的N个推荐结果，默认返回5个；
	参数包括：数据矩阵、用户编号、相似度衡量的方法、预测评分的方法、以及奇异值占比的阈值；
	数据矩阵的行对应用户，列对应物品，函数的作用是基于item的相似性对用户未评过分的物品进行预测评分；
	相似度衡量的方法默认用余弦相似度'''
	def recommend(dataMat,user,N=5,simMeas=cosSim,estMethod=svdEst,percentage=0.9):
	    unratedItems=nonzero(dataMat[user,:].A==0)[1]  #建立一个用户未评分item的列表
	    print(unratedItems)
	    if len(unratedItems)==0:return 'you rated everything' #如果都已经评过分，则退出
	    itemScores=[]
	    for item in unratedItems:  #对于每个未评分的item，都计算其预测评分
	        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
	        itemScores.append((item,estimatedScore))
	    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)#按照item的得分进行从大到小排序
	    return itemScores[:N]  #返回前N大评分值的item名，及其预测评分值
	
	testdata=loadExData()
	print(recommend(testdata,1,N=3,percentage=0.8))#对编号为1的用户推荐评分较高的3件商品

## 主成分分析（PCA）
PCA也常用于数据降维，特别是在人脸识别对于图像数据的处理中，得到了广泛的运用。数据降维的原则是使得数据维度减小，即行向量方差尽可能大，但是同时信息保留最多，就要求行向量之间相关性尽量小，即行向量之间协方差为0.
对于数据：

![](https://i.imgur.com/t8KFr7N.png)

标准化处理先：

![](https://i.imgur.com/dRX80IG.png)

那么协方差矩阵：

![](https://i.imgur.com/KFs2TqU.png)

希望降维后协方差矩阵对角元素极可能大，非对角元素尽可能为0，即成为对角矩阵，则可对X进行线性变换，Y=QX,那么：

![](https://i.imgur.com/MsufQmF.png)

所以，Q为CX的特征向量，其方差为特征值，进行降维一般取前r个特征值对应的特征向量，转换结果为：

![](https://i.imgur.com/N9dKJmz.png)

另一种证明方式可见：[http://blog.csdn.net/zhongkejingwang/article/details/42264479](http://blog.csdn.net/zhongkejingwang/article/details/42264479)




