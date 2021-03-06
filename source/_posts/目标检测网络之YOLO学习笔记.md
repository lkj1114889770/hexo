---
title: '目标检测网络之YOLO学习笔记'
date: 2018-06-15 14:59:34
tags:
	- 深度学习
	- 目标检测
---
YOLO是一种全新的，与R-CNN思想截然不同的目标检测的方法。R-CNN系列网络是通过proposal region产生可能包含目标物体的bounding box，再通分类器判断是否包含物品以及物品类别，用regression对bounding的坐标、大小进行修正。YOLO则是一种end to end的方式，用一个神经网络，实现了预测出bounding box 的坐标、box中包含物体的置信度和物体的probabilities，因此检测速度更快，训练相对更加简单，当然相对来说也带来一些其他缺点。
<!-- more -->

YOLO项目主页[地址](https://pjreddie.com/yolo/)
YOLO1 [论文](https://arxiv.org/abs/1506.02640)
YOLO2 [论文](https://arxiv.org/abs/1612.08242)
YOLO3 [论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

## YOLO v1

YOLO使用来自整张图片的feature map来预测bounding box和class，因此可以保持较高的精度。YOLO将整张图片分成S×S的网格，如果一个目标的中心落入到网格单元中，那么这个网格单元负责这个目标的检测。
<div align=center>
	<img src="https://i.imgur.com/KVhyOJE.png" width="300" height="300" >
</div>
每个网格单元预测B个bounding box和confidence score，confidence score反应了box包含目标的可信度，论文中将可confidence score定义为：
<div align=center>
	<img src="https://i.imgur.com/6vox4OO.png" height="30">
</div>
，因此，如果没有目标存在confidence score为0，否则应该为IOU(intersection over union)，即真实框和预测框的交集部分。所以每个bounding box的预测值包括(x,y,w,h.confidence score). (x,y)表示预测的box中心相对于网格单元的的位置，(w,h)是用整个图片大小进行归一化的宽度和高度，另外，针对C个类别，每个类别需要预测一个条件概率，即：
<div align=center>
	<img src="https://i.imgur.com/VgNV9Rc.png" height="30">
</div>
最终得到box中包含某个特定物品的概率为：
<div align=center>
	<img src="https://i.imgur.com/sehftW1.png" height="40">
</div>
整个过程如下图所示。
<div align=center>
	<img src="https://i.imgur.com/u90BPbt.png" width="700" height="400" >
</div>

总结来说，YOLO网络将检测问题转换成regression，首先将整张图片转换成S×S的网格，并且每个网格单元预测
B个边界框，这些边界框的(x,y,w,h,confidence score)以及C个类别概率,这些预测被编码为S×S×(B\*5+C)
的张量。

### Network Design
YOLO1的网络结构设计借鉴了GoodLeNet模型，包含了24个卷积层和2个全连接层，YOLO未使用inception module，而是使用1x1卷积层和）3x3卷积层简单替代，交替出现的1x1卷积层实现了跨通道信息融合以及通道数目降低。
<div align=center>
	<img src="https://i.imgur.com/BmDEK3e.png" >
</div>

### Training
1. 使用 ImageNet 1000 类数据训练YOLO网络的前20个卷积层+1个average池化层+1个全连接层。
2. 用上面得到的前20个卷积层网络参数来初始化YOLO模型前20个卷积层的网络参数，加入后面的4层卷积层以及2层全连接层进行detection的训练，detection通常需要有细密纹理的视觉信息,所以为提高图像精度，在训练检测模型时，将输入图像分辨率从224 × 224 resize到448x448。
3. 最后一层预测类概率和边界框坐标。我们通过图像宽度和高度来规范(w,h)，使它们落在0和1之间。我们将边界框(x,y)坐标参数化为特定网格单元位置的偏移量，所以它们边界也在0和1之间。

### Loss
YOLO1的误差计算对于分类误差和定位误差用了不同的权重，对包含与不包含物品的box的误差权重也进行了区分。具体来说，论文中增加了边界框坐标预测损失，并减少了不包含目标边界框的置信度预测损失，使用两个参数λcoord和λnoobj来完成这个工作，论文中设置了λcoord=5和λnoobj=0.5。
另一个问题是平方和误差权重在大框和小框中进行了区分。相同的误差下，小框误差的重要性肯定更好，论文中用了一个很巧妙的方法，**直接预测边界框宽度和高度的平方根，而不是宽度和高度**。根据y=x^1/2的函数就可以知道，函数斜率是随着x的增大而减小的，这样就可以提高小框的误差权重，真的巧妙。
YOLO每个网格单元预测多个box。在训练时，每个目标我们只需要一个box来负责，选定的原则是与真实框具有最大的IOU。
<div align=center>
	<img src="https://i.imgur.com/CsmNs9B.png" >
</div>

### Shortcoming
YOLO对边界框预测强加空间约束，因为每个网格单元只预测两个盒子，只能有一个类别。这个空间约束限制了我们的模型可以预测的邻近目标的数量，因此在小物品的检测上比较局限。

## YOLO v2 ##

为提高物体定位精准性和召回率，YOLO2对网络结构的设计进行了改进，输出层使用卷积层替代YOLO的全连接层，联合使用coco物体检测标注数据和imagenet物体分类标注数据训练物体检测模型。相比YOLO，YOLO9000在识别种类、精度、速度、和定位准确性等方面都有大大提升。

### Better

#### Batch Normalization
YOLO2取消了dropout，在所有的卷积层中加入Batch Normalization
。
#### High Resolution Classifier
YOLO2将ImageNet以448×448 的分辨率微调最初的分类网络，迭代10 epochs。

#### Convolutional With Anchor Boxes
借鉴faster R-CNN的思想，引入anchor box，取消全连接层来进行预测，改用卷积层作为预测层对anchor box的offset和confidence进行预测。去除了一个池化层，使得输出特征具有更高的分辨率，将图片输入尺寸resize为416而非448，使得特征图大小为奇数，所以有一个中心单元格。目标，特别是大目标，倾向于占据图像的中心，所以在中心有一个单一的位置可以很好的预测这些目标，而不是四个位置都在中心附近。YOLO的卷积层将图像下采样32倍，所以通过使用输入图像416，我们得到13×13的输出特征图。
同时，使用anchor box进行预测的时候，解耦空间位置预测与类别预测，对每个anchor box都预测object和class，仍然沿用YOLO1，目标检测仍然是预测proposed box和ground truth的IOU，类别预测（class predictions）仍然是存在object下的条件概率。

#### Dimension Clusters
YOLO2不再采用手动挑选的box尺寸，而是对训练集的box尺寸进行k-means聚类，因为聚类的目的是想要更好的IOU，所以聚类的距离使用下列公式：
<div align=center>
	<img src="https://i.imgur.com/F8AsPNZ.png" height="40">
</div>
对不同的k值采用k-means聚类算法，即对数据集的ground truth聚类，在VOC和COCO数据集上的bounding box得到的结果如下图：
<div align=center>
	<img src="https://i.imgur.com/EkQOmSz.png" >
</div>
根据上图，k=5的时候，模型的复杂度和IOU能够得到一个不错的trade off。

#### Direct location prediction
对于位置坐标，YOLO2没有采用R-CNN的预测偏移，而是仍然类似于YOLO1中的，他预测相对于网格单元的位置坐标，将ground truth也限制在0-1之间，使用logistic activation 来实现。网络为每个边界框预测tx，ty，th，tw和to这5个坐标。如果网格单元从图像的左上角偏移（Cx，Cy），给定的anchor的宽度，高度分比为Pw，Ph那么预测结果为：
<div align=center>
	<img src="https://i.imgur.com/XShUyy1.png" >
</div>
<div align=center>
	<img src="https://i.imgur.com/21a8YXR.png" >
</div>

#### Fine-Grained Features
在13×13特征图上检测可以很容易检测到大目标，从更小粒度的特征图中可以更好地检测小物体，YOLO2添加一个passthrough layer从前一层26×26的特征图进行融合。传递层通过将相邻特征堆叠到不同的通道而不是堆叠到空间位置，将较高分辨率特征与低分辨率特征相连，类似于ResNet中的标识映射。这将26×26×512特征映射转换为13×13×2048特征映射，其可以与原始特征连接。

#### Multi-Scale Training
添加anchor box后，YOLO2将分辨率更改为416×416。然而，由于模型只使用卷积层和池化层，它可以在运行中调整大小。为了使YOLOv2能够在不同大小的图像上运行，相比于固定输入图像大小，YOLO2每隔几次迭代更改网络。每迭代10个batch网络随机选择一个新的图像尺寸大小。因为模型以32的因子下采样，YOLO2从以下32的倍数中抽取：{320,352，…，608}。因此，最小的选项是320×320，最大的是608×608.调整网络的大小，并继续训练。
这种训练方法迫使网络学习在各种输入维度上很好地预测。这意味着相同的网络可以预测不同分辨率的检测。网络在更小的尺寸下运行更快，因此YOLO2在速度和精度之间提供了一个简单的折衷。

### Faster

#### Darknet-19
YOLO2大多数3×3的过滤器，并在每个池化步骤后将通道数量加倍，使用全局平均池进行预测，使用1×1滤波器以压缩3×3卷积之间的特征，最终模型，称为Darknet-19，有19卷积层和5个最大池化层，详见下图。
<div align=center>
	<img src="https://i.imgur.com/LCahDqp.png" >
</div>

#### Training for classification
使用Darknet19在标准ImageNet 1000类分类数据集上训练，在训练期间，使用数据增强技巧。

#### Training for detection
为了训练检测器，修改上面的网络，移除最后的卷积层，添加3个3×3卷积层，最后增加1×1卷积层，其输出为我们需要的检测维度，如对于VOC数据集，预测5个box，每个具有5个坐标，每个box20个类，因此125个过滤器。还添加了从最后的3×3×512层到第二到最后的卷积层的传递层passthrough layer，使得模型可以使用细粒度特征。

### Stronger
构建了一种分层分类模型（WordTree），提出了一种关于分类和检测数据的联合训练机制。
<div align=center>
	<img src="https://i.imgur.com/zdvKkpA.png" >
</div>
<div align=center>
	<img src="https://i.imgur.com/TnEqUtA.png" >
</div>

ImageNet数据量更大，用于训练分类，COCO和VOC用于训练检测，ImageN对应分类有9000多种，COCO只有80找那个目标检测，通过wordTree来combine，来自分类的图片只计算分类的loss，来自检测集的图片计算完整的loss。

## YOLO v3

YOLO3 对于YOLO2有了一些改进，总的来说有几点：加深了网络，用了上采样，残差网络，多尺度预测，下面详细说明。

### Bounding Box Prediction
坐标预测仍然沿用YOLO2的，yolov3对每个bounding box预测四个坐标值(tx, ty, tw, th)，对于预测的cell（一幅图划分成S×S个网格cell）根据图像左上角的偏移(cx, cy)，以及之前得到bounding box的宽和高pw, ph可以对bounding box按如下的方式进行预测：
<div align=center>
	<img src="https://i.imgur.com/XShUyy1.png" >
</div>

训练的时候，loss的计算采用sum of squared error loss（平方和距离误差损失），yolov3对每个bounding box通过逻辑回归预测一个物体的得分，如果预测的这个bounding box与真实的边框值大部分重合且比其他所有预测的要好，那么这个值就为1.如果overlap没有达到一个阈值（yolov3中这里设定的阈值是0.5），那么这个预测的bounding box将会被忽略。YOLO3论文中使用的阈值是0.5.每个object智慧分配一个bounding box，所以对应没有分配有ground truth object的box，其坐标损失和预测损失不需要计入，只需考虑objectness loss。If a bounding box prior is not assigned to a ground truth object it incurs no loss for coordinate or class predictions, only objectness.

### Class Prediction
每个框预测分类，bounding box使用多标签分类（multi-label classification）。论文中说没有使用softmax分类，只是使用了简单的逻辑回归进行分类，采用的二值交叉熵损失（binary cross-entropy loss）。
Each box predicts the classes the bounding box may contain using multilabel classification. We do not use a softmax as we have found it is unnecessary for good performance, instead we simply use independent logistic classifiers. During training we use binary cross-entropy loss for the class predictions.
This formulation helps when we move to more complex domains like the Open Images Dataset. In this dataset there are many overlapping labels (i.e. Woman and Person). Using a softmax imposes the assumption that each box has exactly one class which is often not the case. A multilabel approach better models the data.

### Predictions Across Scales
YOLO3在三种不同吃的来预测box，应用一个类似于特征金字塔网络（feature pyramid network）上提前约特征，如下图：
<div align=center>
	<img src="https://i.imgur.com/AYQngjv.png" >
</div>

对于第一个scale的预测，即base feature extractor，最后预测得到一个3-d tensor，包含bounding box,objectness,class prediction.比如在COCO数据集中有80类物品，每一个scale预测3个box，所以tensor得到为（N×N×[3\*(4+1+80)]）。
next scale，从上一步2 layer previous的feature map中进行上采样，然后从特征提取网络中的取earlier feature 与上采样后的进行合并，得到更多信息的语义，以及从earlier feature map可以得到更细粒度的特征。最后的scale采用前述类似的方法进行。可能实际代码更能体现这个过程，如下：
三种跨尺度预测

	predict boxes at 3 different scales
	'''
	def build(self, feat_ex, res18, res10):
	    self.conv52 = self.conv_layer(feat_ex, 1, 1, 1024, 512, True, 'conv_head_52')  		# 13x512
	    self.conv53 = self.conv_layer(self.conv52, 3, 1, 512, 1024, True, 'conv_head_53')   # 13x1024
	    self.conv54 = self.conv_layer(self.conv53, 1, 1, 1024, 512, True, 'conv_head_54')   # 13x512
	    self.conv55 = self.conv_layer(self.conv54, 3, 1, 512, 1024, True, 'conv_head_55')   # 13x1024
	    self.conv56 = self.conv_layer(self.conv55, 1, 1, 1024, 512, True, 'conv_head_56')   # 13x512
	    self.conv57 = self.conv_layer(self.conv56, 3, 1, 512, 1024, True, 'conv_head_57')   # 13x1024
	    self.conv58 = self.conv_layer(self.conv57, 1, 1, 1024, 75, False, 'conv_head_58')   # 13x75
	    # follow yolo layer mask = 6,7,8
	    self.conv59 = self.conv_layer(self.conv56, 1, 1, 512, 256, True, 'conv_head_59')    # 13x256
	    size = tf.shape(self.conv59)[1]
	    self.upsample0 = tf.image.resize_nearest_neighbor(self.conv59, [2*size, 2*size],    # 上采样
	                                                      name='upsample_0')                # 26x256
	    self.route0 = tf.concat([self.upsample0, res18], axis=-1, name='route_0')           # 26x768
	    self.conv60 = self.conv_layer(self.route0, 1, 1, 768, 256, True, 'conv_head_60')    # 26x256
	    self.conv61 = self.conv_layer(self.conv60, 3, 1, 256, 512, True, 'conv_head_61')    # 26x512
	    self.conv62 = self.conv_layer(self.conv61, 1, 1, 512, 256, True, 'conv_head_62')    # 26x256
	    self.conv63 = self.conv_layer(self.conv62, 3, 1, 256, 512, True, 'conv_head_63')    # 26x512
	    self.conv64 = self.conv_layer(self.conv63, 1, 1, 512, 256, True, 'conv_head_64')    # 26x256
	    self.conv65 = self.conv_layer(self.conv64, 3, 1, 256, 512, True, 'conv_head_65')    # 26x512
	    self.conv66 = self.conv_layer(self.conv65, 1, 1, 512, 75, False, 'conv_head_66')    # 26x75
	    # follow yolo layer mask = 3,4,5
	    self.conv67 = self.conv_layer(self.conv64, 1, 1, 256, 128, True, 'conv_head_67')    # 26x128
	    size = tf.shape(self.conv67)[1]
	    self.upsample1 = tf.image.resize_nearest_neighbor(self.conv67, [2 * size, 2 * size],
	                                                      name='upsample_1')                # 52x128
	    self.route1 = tf.concat([self.upsample1, res10], axis=-1, name='route_1')           # 52x384
	    self.conv68 = self.conv_layer(self.route1, 1, 1, 384, 128, True, 'conv_head_68')    # 52x128
	    self.conv69 = self.conv_layer(self.conv68, 3, 1, 128, 256, True, 'conv_head_69')    # 52x256
	    self.conv70 = self.conv_layer(self.conv69, 1, 1, 256, 128, True, 'conv_head_70')    # 52x128
	    self.conv71 = self.conv_layer(self.conv70, 3, 1, 128, 256, True, 'conv_head_71')    # 52x256
	    self.conv72 = self.conv_layer(self.conv71, 1, 1, 256, 128, True, 'conv_head_72')    # 52x128
	    self.conv73 = self.conv_layer(self.conv72, 3, 1, 128, 256, True, 'conv_head_73')    # 52x256
	    self.conv74 = self.conv_layer(self.conv73, 1, 1, 256, 75, False, 'conv_head_74')    # 52x75
	    # follow yolo layer mask = 0,1,2
	
	    return self.conv74, self.conv66, self.conv58

上面是最后的预测部分，需要输入的三个特征从Darknet-53网络中得到的，输出地方做了注释，Darknet-53网络结构如下：

 	def build(self, img, istraining, decay_bn=0.99):
        self.phase_train = istraining
        self.decay_bn = decay_bn
        self.conv0 = self.conv_layer(bottom=img, size=3, stride=1, in_channels=3,   # 416x3
                                     out_channels=32, name='conv_0')                # 416x32
        self.conv1 = self.conv_layer(bottom=self.conv0, size=3, stride=2, in_channels=32,
                                     out_channels=64, name='conv_1')                # 208x64
        self.conv2 = self.conv_layer(bottom=self.conv1, size=1, stride=1, in_channels=64,
                                     out_channels=32, name='conv_2')                # 208x32
        self.conv3 = self.conv_layer(bottom=self.conv2, size=3, stride=1, in_channels=32,
                                     out_channels=64, name='conv_3')                # 208x64
        self.res0 = self.conv3 + self.conv1                                         # 208x64
        self.conv4 = self.conv_layer(bottom=self.res0, size=3, stride=2, in_channels=64,
                                     out_channels=128, name='conv_4')               # 104x128
        self.conv5 = self.conv_layer(bottom=self.conv4, size=1, stride=1, in_channels=128,
                                     out_channels=64, name='conv_5')                # 104x64
        self.conv6 = self.conv_layer(bottom=self.conv5, size=3, stride=1, in_channels=64,
                                     out_channels=128, name='conv_6')               # 104x128
        self.res1 = self.conv6 + self.conv4     # 128                               # 104x128
        self.conv7 = self.conv_layer(bottom=self.res1, size=1, stride=1, in_channels=128,
                                     out_channels=64, name='conv_7')                # 104x64
        self.conv8 = self.conv_layer(bottom=self.conv7, size=3, stride=1, in_channels=64,
                                     out_channels=128, name='conv_8')               # 104x128
        self.res2 = self.conv8 + self.res1      # 128                               # 104x128
        self.conv9 = self.conv_layer(bottom=self.res2, size=3, stride=2, in_channels=128,
                                     out_channels=256, name='conv_9')               # 52x256
        self.conv10 = self.conv_layer(bottom=self.conv9, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_10')             # 52x128
        self.conv11 = self.conv_layer(bottom=self.conv10, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_11')             # 52x256
        self.res3 = self.conv11 + self.conv9                                        # 52x256
        self.conv12 = self.conv_layer(bottom=self.res3, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_12')             # 52x128
        self.conv13 = self.conv_layer(bottom=self.conv12, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_13')             # 52x256
        self.res4 = self.conv13 + self.res3                                         # 52x256
        self.conv14 = self.conv_layer(bottom=self.res4, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_14')             # 52x128
        self.conv15 = self.conv_layer(bottom=self.conv14, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_15')             # 52x256
        self.res5 = self.conv15 + self.res4                                         # 52x256
        self.conv16 = self.conv_layer(bottom=self.res5, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_16')             # 52x128
        self.conv17 = self.conv_layer(bottom=self.conv16, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_17')             # 52x256
        self.res6 = self.conv17 + self.res5                                         # 52x256
        self.conv18 = self.conv_layer(bottom=self.res6, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_18')             # 52x128
        self.conv19 = self.conv_layer(bottom=self.conv18, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_19')             # 52x256
        self.res7 = self.conv19 + self.res6                                         # 52x256
        self.conv20 = self.conv_layer(bottom=self.res7, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_20')             # 52x128
        self.conv21 = self.conv_layer(bottom=self.conv20, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_21')             # 52x256
        self.res8 = self.conv21 + self.res7                                         # 52x256
        self.conv22 = self.conv_layer(bottom=self.res8, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_22')             # 52x128
        self.conv23 = self.conv_layer(bottom=self.conv22, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_23')             # 52x256
        self.res9 = self.conv23 + self.res8                                         # 52x256
        self.conv24 = self.conv_layer(bottom=self.res9, size=1, stride=1, in_channels=256,
                                      out_channels=128, name='conv_24')             # 52x128
        self.conv25 = self.conv_layer(bottom=self.conv24, size=3, stride=1, in_channels=128,
                                      out_channels=256, name='conv_25')             # 52x256
        self.res10 = self.conv25 + self.res9                                        # 52x256 一个输出的特征尺度
        self.conv26 = self.conv_layer(bottom=self.res10, size=3, stride=2, in_channels=256,
                                      out_channels=512, name='conv_26')             # 26x512
        self.conv27 = self.conv_layer(bottom=self.conv26, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_27')             # 26x256
        self.conv28 = self.conv_layer(bottom=self.conv27, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_28')             # 26x512
        self.res11 = self.conv28 + self.conv26                                      # 26x512
        self.conv29 = self.conv_layer(bottom=self.res11, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_29')             # 26x256
        self.conv30 = self.conv_layer(bottom=self.conv29, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_30')             # 26x512
        self.res12 = self.conv30 + self.res11                                       # 26x512
        self.conv31 = self.conv_layer(bottom=self.res12, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_31')             # 26x256
        self.conv32 = self.conv_layer(bottom=self.conv31, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_32')             # 26x512
        self.res13 = self.conv32 + self.res12                                       # 26x512
        self.conv33 = self.conv_layer(bottom=self.res13, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_33')             # 26x256
        self.conv34 = self.conv_layer(bottom=self.conv33, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_34')             # 26x512
        self.res14 = self.conv34 + self.res13                                       # 26x512
        self.conv35 = self.conv_layer(bottom=self.res14, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_35')             # 26x256
        self.conv36 = self.conv_layer(bottom=self.conv35, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_36')             # 26x512
        self.res15 = self.conv36 + self.res14                                       # 26x512
        self.conv37 = self.conv_layer(bottom=self.res15, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_37')             # 26x256
        self.conv38 = self.conv_layer(bottom=self.conv37, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_38')             # 26x512
        self.res16 = self.conv38 + self.res15                                       # 26x512
        self.conv39 = self.conv_layer(bottom=self.res16, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_39')             # 26x256
        self.conv40 = self.conv_layer(bottom=self.conv39, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_40')             # 26x512
        self.res17 = self.conv40 + self.res16                                       # 26x512
        self.conv41 = self.conv_layer(bottom=self.res17, size=1, stride=1, in_channels=512,
                                      out_channels=256, name='conv_41')             # 26x256
        self.conv42 = self.conv_layer(bottom=self.conv41, size=3, stride=1, in_channels=256,
                                      out_channels=512, name='conv_42')             # 26x512
        self.res18 = self.conv42 + self.res17                                       # 26x512，一个输出的特征尺度
        self.conv43 = self.conv_layer(bottom=self.res18, size=3, stride=2, in_channels=512,
                                      out_channels=1024, name='conv_43')            # 13x1024
        self.conv44 = self.conv_layer(bottom=self.conv43, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_44')             # 13x512
        self.conv45 = self.conv_layer(bottom=self.conv44, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_45')            # 13x1024
        self.res19 = self.conv45 + self.conv43                                      # 13x1024
        self.conv46 = self.conv_layer(bottom=self.res19, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_46')             # 13x512
        self.conv47 = self.conv_layer(bottom=self.conv44, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_47')            # 13x1024
        self.res20 = self.conv47 + self.res19                                       # 13x1024
        self.conv48 = self.conv_layer(bottom=self.res20, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_48')             # 13x512
        self.conv49 = self.conv_layer(bottom=self.conv48, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_49')            # 13x1024
        self.res21 = self.conv49 + self.res20                                       # 13x1024
        self.conv50 = self.conv_layer(bottom=self.res21, size=1, stride=1, in_channels=1024,
                                      out_channels=512, name='conv_50')             # 13x512
        self.conv51 = self.conv_layer(bottom=self.conv50, size=3, stride=1, in_channels=512,
                                      out_channels=1024, name='conv_51')            # 13x1024
        self.res23 = self.conv51 + self.res21                                       # 13x1024
        return self.res23  # 最后输出特征

同样采用k-means聚类的到anchor box的尺寸。选取了9种，3中不同的scale：(10×13); (16×30); (33×23); (30×61); (62×45); (59×119); (116 × 90); (156 × 198); (373 × 326).

### Feature Extractor
YOLO3的新的更深的网络，Darknet-53，实现细节可参见上面的代码
<div align=center>
	<img src="https://i.imgur.com/ZxUzfO8.png" >
</div>
