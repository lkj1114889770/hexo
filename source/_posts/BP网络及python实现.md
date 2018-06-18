---
title: BP网络及python实现
date: 2017-10-23 16:48:07
tags:
	- 机器学习
	- 监督学习
---
BP神经网络改变了感知器的结构，引入了新的隐含层以及误差反向传播，基本上能够解决非线性分类问题，也是神经网络的基础网络结构，在此对BP神经网络算法进行总结，并用python对其进行了实现。
<!-- more -->

BP神经网络的典型结构如下图所示：

![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1508759278442&di=35b034d166ee7a0c6e09c7154c096d3f&imgtype=0&src=http%3A%2F%2Fimgsrc.baidu.com%2Fbaike%2Fpic%2Fitem%2F9922720e0cf3d7ca65c52b8ef01fbe096b63a912.jpg)

隐含层通常为一层，也可以是多层，在BP网络中一般不超过2层。

## 正向传播
正向传播的过程与感知器类似，都是输入与权重的点积，隐含层和输出层都包含一个激活函数，BP网络常用sigmod函数。

![](https://i.imgur.com/5FpsaS2.png)

但是现在好像不常用了，更多地是Tanh或者是ReLU，好像最近又出了一个全新的激活函数，后续还得去了解。
BP神经网络的误差函数是全局误差，将所有样本的误差都进行计算求和，所以在算法过程学习的时候，进行的是批量学习，等所有数据都进行批次计算之后，才进行权重调整。

![](https://i.imgur.com/vW9ndOr.png)

## 反向传播过程
这个可以说是BP网络比较精髓的部分了，也是BP网络能够从数据中学习的关键，误差的反向传播过程就是两种情况，要么输出层神经元，要么是隐含层神经元。

![](https://i.imgur.com/1luEa3W.png)

对于输出神经元，权重的梯度修正法则为：

![](https://i.imgur.com/vc9TjCN.png)

即权重增量等于学习率、局域梯度、输出层输出结果的乘积，对于局域梯度，其计算如下：

![](https://i.imgur.com/xvHCU1P.png)

即为误差信号乘于激活函数的导数，其中n表示第n次迭代。
对于sigmod函数来说，其导数为：

![](https://i.imgur.com/nQVO1tw.png)

对于隐藏层来说，情况更加复杂一点，需要经过上一层的误差传递。

![](https://i.imgur.com/Ep6t5KR.png)

隐藏层的局域梯度为：

![](https://i.imgur.com/pC56jAc.png)

上面式子的第一项，说明隐含层神经元j局域梯度的计算仅以来神经元j的激活函数的导数，但是第二项求和，是上一层神经元的局域梯度通过权重w进行了传递。

总的来说，反向传播算法中，权重的调整值规则为：

（权值调整）=（学习率参数） X （局域梯度） X（神经元j的输入信号）

BP算法中还有一个动量因子（mc），主要是网络调优，防止网络发生震荡或者收敛过慢，其基本思想就是在t时刻权重更新的时候考虑t-1时刻的梯度值。
	
	self.out_wb = self.out_wb + (1-self.mc)*self.eta*dout_wb + self.mc*self.eta*dout_wbold
	self.hi_wb =self.hi_wb + (1-self.mc)*self.eta*dhi_wb + self.mc*self.eta*dhi_wbold

## BP网络分类算法
首先构造的一个BP类

	# -*- coding: utf-8 -*-
	"""
	Created on Mon Oct 23 09:12:46 2017
	
	@author: lkj
	"""
	
	import numpy as np
	from numpy import *
	import matplotlib.pyplot as plt
	
	class BpNet(object):
	    def __init__(self):
	        # 以下参数需要手动设置  
	        self.eb=0.01              # 误差容限，当误差小于这个值时，算法收敛，程序停止
	        self.eta=0.1             # 学习率
	        self.mc=0.3               # 动量因子：引入的一个调优参数，是主要的调优参数 
	        self.maxiter=2000         # 最大迭代次数
	        self.errlist=[]           # 误差列表
	        self.dataMat=0            # 训练集
	        self.classLabels=0        # 分类标签集
	        self.nSampNum=0             # 样本集行数
	        self.nSampDim=0             # 样本列数
	        self.nHidden=4           # 隐含层神经元 
	        self.nOut=1              # 输出层个数
	        self.iterator=0            # 算法收敛时的迭代次数
	     
	    #激活函数
	    def logistic(self,net):
	        return 1.0/(1.0+exp(-net))
	    
	    #反向传播激活函数的导数
	    def dlogistic(self,y):
	        return (y*(1-y))
	    
	    #全局误差函数
	    def errorfuc(self,x):
	        return sum(x*x)*0.5
	    
	    #加载数据集
	    def loadDataSet(self,FileName):
	        data=np.loadtxt(FileName)
	        m,n=shape(data)
	        self.dataMat = np.ones((m,n))
	        self.dataMat[:,:-1] = data[:,:-1] #除数据外一列全为1的数据，与权重矩阵中的b相乘
	        self.nSampNum = m  #样本数量
	        self.nSampDim = n-1  #样本维度
	        self.classLabels =data[:,-1]    
	    
	    #数据集归一化，使得数据尽量处在同一量纲，这里采用了标准归一化
	    #数据归一化应该针对的是属性，而不是针对每条数据
	    def normalize(self,data):
	        [m,n]=shape(data)
	        for i in range(n-1):
	            data[:,i]=(data[:,i]-mean(data[:,i]))/(std(data[:,i])+1.0e-10)
	        return data
	    
	    #隐含层、输出层神经元权重初始化
	    def init_WB(self):
	        #隐含层
	        self.hi_w = 2.0*(random.rand(self.nSampDim,self.nHidden)-0.5)
	        self.hi_b = 2.0*(random.rand(1,self.nHidden)-0.5)
	        self.hi_wb = vstack((self.hi_w,self.hi_b))
	        
	        #输出层
	        self.out_w = 2.0*(random.rand(self.nHidden,self.nOut)-0.5)
	        self.out_b = 2.0*(random.rand(1,self.nOut)-0.5)
	        self.out_wb = vstack((self.out_w,self.out_b))
	        
	    def BpTrain(self):
	        SampIn = self.dataMat
	        expected = self.classLabels
	        dout_wbold = 0.0
	        dhi_wbold = 0.0 #记录隐含层和输出层前一次的权重值，初始化为0
	        self.init_WB()
	        
	        for i in range(self.maxiter):
	            #信号正向传播
	            #输入层到隐含层
	            hi_input = np.dot(SampIn,self.hi_wb)
	            hi_output = self.logistic(hi_input)
	            hi2out = np.hstack((hi_output,np.ones((self.nSampNum,1))))
	            
	            #隐含层到输出层
	            out_input=np.dot(hi2out,self.out_wb)
	            out_output = self.logistic(out_input)
	            #计算误差
	            error = expected.reshape(shape(out_output)) - out_output
	            sse = self.errorfuc(error)
	            self.errlist.append(sse)
	            if sse<=self.eb:
	                self.iterator = i+1
	                break
	            
	            #误差反向传播
	            
	            #DELTA输出层梯度
	            DELTA = error*self.dlogistic(out_output)
	            #delta隐含层梯度
	            delta =  self.dlogistic(hi_output)*np.dot(DELTA,self.out_wb[:-1,:].T)
	            dout_wb = np.dot(hi2out.T,DELTA)
	            dhi_wb = np.dot(SampIn.T,delta)
	            
	            #更新输出层和隐含层权值
	            if i==0:
	                self.out_wb = self.out_wb + self.eta*dout_wb
	                self.hi_wb = self.hi_wb + self.eta*dhi_wb
	            else:
	                #加入动量因子
	               self.out_wb = self.out_wb + (1-self.mc)*self.eta*dout_wb + self.mc*self.eta*dout_wbold
	               self.hi_wb =self.hi_wb + (1-self.mc)*self.eta*dhi_wb + self.mc*self.eta*dhi_wbold
	            dout_wbold = dout_wb
	            dhi_wbold = dhi_wb
	    
	    ##输入测试点，输出分类结果      
	    def BpClassfier(self,start,end,steps=30):
	        x=linspace(start,end,steps)
	        xx=np.ones((steps,steps))
	        xx[:,0:steps] = x
	        yy = xx.T
	        z = np.ones((steps,steps))
	        for i in  range(steps):
	            for j in range(steps):
	                xi=array([xx[i,j],yy[i,j],1])
	                hi_input = np.dot(xi,self.hi_wb)
	                hi_out = self.logistic(hi_input)
	                hi_out = mat(hi_out)
	                m,n=shape(hi_out)
	                hi_b = ones((m,n+1))
	                hi_b[:,:n] = hi_out
	                out_input = np.dot(hi_b,self.out_wb)
	                out = self.logistic(out_input)
	                z[i,j] = out
	        return x,z
	                
	    def classfyLine(self,plt,x,z):
	        #画出分类分隔曲线，用等高线画出
	        plt.contour(x,x,z,1,colors='black')
	        
	    def errorLine(self,plt,color='r'):
	        x=linspace(0,self.maxiter,self.maxiter)
	        y=log2(self.errlist)
	        #y=y.reshape(())
	        #print(shape(x),shape(y))
	        plt.plot(x,y,color)
	        
	   # 绘制数据散点图
	    def drawDataScatter(self,plt):
	        i=0
	        for data in self.dataMat:
	            if(self.classLabels[i]==0):
	                plt.scatter(data[0],data[1],c='blue',marker='o')
	            else:
	                plt.scatter(data[0],data[1],c='red',marker='s')
	            i=i+1

利用分类器执行分类：

	from BpNet import *
	import matplotlib.pyplot as plt 
	
	# 数据集
	bpnet = BpNet() 
	bpnet.loadDataSet("testSet2.txt")
	bpnet.dataMat = bpnet.normalize(bpnet.dataMat)
	
	# 绘制数据集散点图
	bpnet.drawDataScatter(plt)
	
	# BP神经网络进行数据分类
	bpnet.BpTrain()
	
	print(bpnet.out_wb)
	print(bpnet.hi_wb)
	
	# 计算和绘制分类线
	x,z = bpnet.BpClassfier(-3.0,3.0)
	bpnet.classfyLine(plt,x,z)
	plt.show()
	# 绘制误差曲线
	bpnet.errorLine(plt)
	plt.show()            
            
输出结果为：

![](https://i.imgur.com/GYv2q53.png)
            
误差输出结果：

![](https://i.imgur.com/xAXcgfX.png)
            
可以看到在1000次左右迭代就已经出现了比较好的结果了。  
具体代码可见个人github仓库[https://github.com/lkj1114889770/Machine-Leanring-Algorithm/tree/master/BpNet](https://github.com/lkj1114889770/Machine-Leanring-Algorithm/tree/master/BpNet)        
            
除了分类，BP神经网络也常用在函数逼近，这时候输出层神经元激活函数一般就不会再采用sigmod函数了，通常采用线性函数。


**【参考文献】**
《神经网络与机器学习》（第3版） （加） Simon Haykin 著；
《机器学习算法原理与编程实践》 郑捷著；
            
            
            
            
            
        
