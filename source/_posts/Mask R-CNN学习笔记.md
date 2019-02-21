---
title: 'Mask R-CNN论文+源码阅读笔记'
date: 2018-09-20 15:54:57
tags:
	- 实例分割
	- 深度学习
	- CV
---
论文链接：[https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)
源码链接：[https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
之前一直在做目标检测这块，最近了解了一下实例分割，其实是目标检测更细致的任务，在图像中做到像素级的分割，包括目标检测和准确的像素分割，所以说是结合了之前目标检测的任务（classification and localization），以及语义分割（将像素点分类到特定的所属类别），首先拜读的就是17年何凯明大神的论文[Mask R-CNN](https://arxiv.org/abs/1703.06870)，并且阅读了keras版本的实现[代码](https://github.com/matterport/Mask_RCNN)，在此做一个学习笔记，
<!-- more -->

## Introduction
Mask R-CNN是在faster R-CNN的classification branch和bounding box regression branch基础上，增加了一个segmentation mask branch，以像素到像素的方法来预测分割掩码（segmentation mask），如下图所示。

![](/img/Mask_R-CNN/figure_1.png)

Faster R-CNN由于RoI pooling，没有办法做到输入和输出之间的像素到像素的对齐(pixel-to-pixel)，为了解决这个问题，Mask R-CNN提出了一个RoiAlign层，可以极大地提高掩码的准确率；同时解耦掩码预测和分类预测，为每个类都独立地预测二进制掩码，这样不会跨类别竞争。最终运行速度可达5 FPS左右。

## Mask R-CNN
Mask R-CNN和faster R-CNN类似，具有两个阶段，第一阶段是RPN，第二阶段出了预测类别和检测框的偏移外，还能够为每个RoI输出二进制掩码，这三个输出都是并行输出的。此外对每一个RoI提出了multi task：
L=L*cls*+L*box*+L*mask*
Lcls和Lbox和之前faster RCNN中定义的一样，掩码分支对于每个RoI的输出维度为Km2，即K个分辨率为m×m的二进制掩码，每个类别一个，K为类别数量。对每个像素应用sigmod，Lmask为平均二进制交叉熵损失，对于真实类别为k的RoI，仅在第k个掩码上计算Lmask，其他掩码输出不计入损失。

![](/img/Mask_R-CNN/figure_2.png)

### Mask Representation
掩码用来表述目标在图片中的像素位置，在mask R-CNN中通过卷积的方法，提供了像素到像素的对应来解决。具体来说，使用FCN的方法为每一个RoI预测一个m x m的掩码，与使用FC层预测掩码的方式不同，全卷积的方法需要更少的参数，预测也会更加准确。这种像素到像素的行为需要RoI特征，为了更好地与原图进行对齐，来准确地对应原图的像素关系，这里就提出了一个很关键的模块，RoIAlign层。

### RoIAlign
这个网络层主要是为了更好地与原图像素进行对齐，对之前faster R-CNN使用RoI Pooling操作中两次量化造成的区域不匹配(mis-aligment)问题进行了改进，所以在这里就不得不提一下RoI的局限性，借鉴了[一篇博客](http://blog.leanote.com/post/afanti/b5f4f526490b)的详细介绍。
#### RoI pooling的局限性
**Faster R-CNN的网络框架**

![](/img/Mask_R-CNN/figure_3.png)

由上图可以看到，RoI pooling位于RPN、Feature map和classification and regression之间，针对RPN输出的RoI，将其resize到统一的大小，首先将RoI映射到feature map对应的位置，将映射的区域划分为k x k个单元，对每个单元进行maxpooling，这样就得到统一大小k x k的输出了,期间就存在两次量化的过程
1. 将候选框量化为整数点坐标值
2. 将量化后的边界区域分割成k x k个单元（bin),对每一个单元的边界进行量化。

RoIPooling 采用的是 INTER_NEAREST（即最近邻插值） ，即在resize时，对于 缩放后坐标不能刚好为整数 的情况，采用了 粗暴的四舍五入，相当于选取离目标点最近的点。，经过这样两次量化就出现了边界不匹配的问题了(misaligment),候选框和最开始回归出来的已经有很大偏差了。

#### RoIAlign
Mask R-CNN将最邻近插值换成了双线性插值，这样就有了RoIAlign，主要流程为：
1. 遍历每一个候选区域，保持浮点数边界不做量化。
2. 将候选区域分割成k x k个单元，每个单元的边界也不做量化。
3. 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

第三步的操作，论文也说的很清楚，这个固定位置指的是每一个单元(bin)中按照固定规则确定的位置，比如，如果采样点数是1，那么就是这个单元的中心点。如果采样点数是4，那么就是把这个单元平均分割成四个小方块以后它们分别的中心点。显然这些采样点的坐标通常是浮点数，所以需要使用插值的方法得到它的像素值。下图的例子中，虚线为特征图，实线为RoI，这里假设RoI分割成2 x 2的单元，在每个单元换分为4个小方块后，每个小方块的中心作为采样点，即图中的点，但是这些点的坐标一般来说是浮点数，采用双线性插值的方法来获得像素值，这样就不存在量化过程，很好地解决了misAligment问题。

![](/img/Mask_R-CNN/figure_4.png)

### Network Architecture
Mask R-CNN使用”网络-深度-特征输出层”的方式命名底下层卷积网络。我们评估了深度为50或101层的ResNet25和ResNeXt26网络。使用ResNet的Faster R-CNN从第四阶段的最终卷积层提取特征，我们称之为C4。例如，使用ResNet-50的下层网络由ResNet-50-C4表示。
Mask R-CNN扩展了 ResNet和FPN中提出的Faster R-CNN的上层网络。详细信息如下图所示：（上层网络架构：我们扩展了两种现有的Faster R-CNN上层网络架构，分别添加了一个掩码分支。图中数字表示分辨率和通道数，箭头表示卷积、反卷积或全连接层（可以通过上下文推断，卷积减小维度，反卷积增加维度。）所有的卷积都是3×3的，除了输出层，是1×1的。反卷积是2×2的，步进为2，,隐藏层使用ReLU。左中，“res5”表示ResNet的第五阶段，简单起见，我们修改了第一个卷积操作，使用7×7，步长为1的RoI代替14×14，步长为2的RoI25。右图中的“×4”表示堆叠的4个连续的卷积。）

![](/img/Mask_R-CNN/figure_5.png)

## 代码阅读
主要参考来自于[csdn一篇博客](https://blog.csdn.net/horizonheart/article/details/81188161)，借用他的图。

![](/img/Mask_R-CNN/flow_diagram.png)

### backbone network
使用resnet101作为特征提取网络，生成金字塔网络，并在每层提取特征。


    # Build the shared convolutional layers.
    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    if callable(config.BACKBONE):
        _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                            train_bn=config.TRAIN_BN)
    else:
        _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                         stage5=True, train_bn=config.TRAIN_BN)
    # Top-down Layers
    # TODO: add assert to varify feature map sizes match what's in config
    # FPN：把底层的特征和高层的特征进行融合，便于细致检测。
    # 这里P5=C5，然后P4=P5+C4,P2 P3类似，最终得到rpn_feature_maps，注意这里多了个P6,其仅是由P5下采样获得。
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]

### Anchor
在前面金字塔特征图的基础上，生成anchor

	# Anchors
    # 如果是训练的情况，就在金字塔特征图上生成Anchor
    if mode == "training":
        # 在金字塔特征图上以每个像素为中心，以配置文件的anchor大小为宽高生成anchor
        # 根据特征图相应原图缩小的比例，还原到原始的输入图片上，即得到的是anchor在原始图片上的坐标
        # 获得结果为(N,[y1,x1,y2,x2])
        anchors = self.get_anchors(config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    else:
        # 如果是Inference就是输入的Anchor
        anchors = input_anchors

### RPN Model
将金字塔特征图输入到RPN中，得到网络的分类（前景和背景两类）和bbox的回归值。

	# RPN Model
    # RPN主要实现2个功能：
    # 1 > box的前景色和背景色的分类
    # 2 > box框体的回归修正
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]
    # rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits(before softmax)
    # rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    # rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    rpn_class_logits, rpn_class, rpn_bbox = outputs

其中RPN网络也用keras做了实现，Builds a Keras model of the Region Proposal Network.It wraps the RPN graph so it can be used multiple times with shared weights.

 	# Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

代码主要实现是，在特征图上，用kernel_size为所需输出个数（如对分类，为2 * anchors_per_location），stride为1的卷积在特征图上进行卷积，得到RPN的输出。

### Generate proposals
这部分网络主要是用来对anchor进行筛选，所谓proposal，主要步骤为：

	# Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, :, 1]
    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
    # Anchors
    anchors = inputs[2]

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
    ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                     name="top_anchors").indices
    scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                               self.config.IMAGES_PER_GPU)
    deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                               self.config.IMAGES_PER_GPU)
    pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                self.config.IMAGES_PER_GPU,
                                names=["pre_nms_anchors"])

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = utils.batch_slice([pre_nms_anchors, deltas],
                              lambda x, y: apply_box_deltas_graph(x, y),
                              self.config.IMAGES_PER_GPU,
                              names=["refined_anchors"])

    # Clip to image boundaries. Since we're in normalized coordinates,
    # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = utils.batch_slice(boxes,
                              lambda x: clip_boxes_graph(x, window),
                              self.config.IMAGES_PER_GPU,
                              names=["refined_anchors_clipped"])

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    def nms(boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(boxes, indices)
        # Pad if needed
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
    proposals = utils.batch_slice([boxes, scores], nms,
                                  self.config.IMAGES_PER_GPU)
    return proposals

1. 按score得分排序，取前6000个
2. 将rpn的输出应用到anchors进行修正
3. 舍弃修正后边框超过归一化的0-1区间内的
4. 用非极大值抑制的方法获取最后的anchor

### Generate detection target
训练的时候计算loss需要有target，这一步就是对剩下的anchor产生detection target，以便后续计算loss。主要计算的步骤为：
1. 计算proposal和gt_box之间的iou值，大于0.5则被认为是正样本，小于0.5，并且和crow box相交不大的为负样本
2. 对负样本进行采样，保证正样本占有33%的比例，保证正负样本平衡
3. 根据正样本和那个gt_box的iou最大来给正样本分配gt_box和gt_max,以便计算偏差

### fpn classifier graph &fpn mask graph
这部分为分类网络，当然还有一个并行的mask分支，分类使用的是mrcnn_feature_map，即前面的P2、P3、P4、P5。基本思路是先经过ROIAlign层（取代了RoIPooling），再经过两层卷积后连接两个全连接层分别输出class和box。fpn_mask_graph也是类似，针对mask部分，只不过不同的是，前者经过PyramidROIAlign得到的特征图是7x7大小的，二，而后者经过PyramidROIAlign得到的特征图大小是14x14.

### Loss
loss包括5个部分组成，分别是rpn网络的两个损失：rpn_class_loss，计算前景和背景分类损失；rpn_bbox_loss，计算rpn\_box损失,以及输出的class，box和mask的损失计算。


    # Losses
    rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])

### PyramidROIAlign
PyramidROIAlign输入时金字塔特征图，所以首先需要确认来自于哪一层，作者的计算方法采用如下公式

![](/img/Mask_R-CNN/PyramidROIAlign.png)

这里k0=4，从对应特征图中去除坐标对应区域，利用双线性插值进行pooling，这里作者依据论文中的，虽然没有采用论文中的4个点采样的方法，但是采用了论文中提到的也非常有效的1个点采样的方法，而tf.crop_and_resize这个函数crops and resizes an image and handles the bilinear interpolation，所以用这个进行了实现。

		# Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
	boxes = inputs[0]
	
	# Image meta
	# Holds details about the image. See compose_image_meta()
	image_meta = inputs[1]
	
	# Feature Maps. List of feature maps from different level of the
	# feature pyramid. Each is [batch, height, width, channels]
	feature_maps = inputs[2:]
	
	# Assign each ROI to a level in the pyramid based on the ROI area.
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
	h = y2 - y1
	w = x2 - x1
	# Use shape of first image. Images in a batch must have the same size.
	image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
	# Equation 1 in the Feature Pyramid Networks paper. Account for
	# the fact that our coordinates are normalized here.
	# e.g. a 224x224 ROI (in pixels) maps to P4
	image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
	roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
	roi_level = tf.minimum(5, tf.maximum(
	    2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
	roi_level = tf.squeeze(roi_level, 2)
	
	# Loop through levels and apply ROI pooling to each. P2 to P5.
	pooled = []
	box_to_level = []
	for i, level in enumerate(range(2, 6)):
	    ix = tf.where(tf.equal(roi_level, level))
	    level_boxes = tf.gather_nd(boxes, ix)
	
	    # Box indices for crop_and_resize.
	    box_indices = tf.cast(ix[:, 0], tf.int32)
	
	    # Keep track of which box is mapped to which level
	    box_to_level.append(ix)
	
	    # Stop gradient propogation to ROI proposals
	    level_boxes = tf.stop_gradient(level_boxes)
	    box_indices = tf.stop_gradient(box_indices)
	
	    # Crop and Resize
	    # From Mask R-CNN paper: "We sample four regular locations, so
	    # that we can evaluate either max or average pooling. In fact,
	    # interpolating only a single value at each bin center (without
	    # pooling) is nearly as effective."
	    #
	    # Here we use the simplified approach of a single value per bin,
	    # which is how it's done in tf.crop_and_resize()
	    # Result: [batch * num_boxes, pool_height, pool_width, channels]
	    pooled.append(tf.image.crop_and_resize(
	        feature_maps[i], level_boxes, box_indices, self.pool_shape,
	        method="bilinear"))
	
	# Pack pooled features into one tensor
	pooled = tf.concat(pooled, axis=0)
	
	# Pack box_to_level mapping into one array and add another
	# column representing the order of pooled boxes
	box_to_level = tf.concat(box_to_level, axis=0)
	box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
	box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
	                         axis=1)
	
	# Rearrange pooled features to match the order of the original boxes
	# Sort box_to_level by batch then box index
	# TF doesn't have a way to sort by two columns, so merge them and sort.
	sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
	ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
	    box_to_level)[0]).indices[::-1]
	ix = tf.gather(box_to_level[:, 2], ix)
	pooled = tf.gather(pooled, ix)
	
	# Re-add the batch dimension
	pooled = tf.expand_dims(pooled, 0)
	return pooled

### build_rpn_targets
从loss可以看到，训练的时候，rpn的loss输入需要有target，代码中为rpn_match和rpn_box,计算方法主要也是根据金字塔的给定anchors和gt_box的iou，此处阈值为0.7，来确定postive和negative，并分配对应的gt_box来计算delta。

	 """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

### train
可以看到,计算RPN的时候的RPN loss和分类loss其实原始的输入都需要gt_box，只不过训练好之后，分类loss是在RPN的基础上。作者代码将RPN和后面的faster RCNN部分以及增加的mask 分支一起训练，所以在训练代码中加了下面一段：

	# Stop gradient propogation to ROI proposals
	level_boxes = tf.stop_gradient(level_boxes)
	box_indices = tf.stop_gradient(box_indices)

引用官方的解释，主要是为了不让两部分互相影响。
If we don't stop the gradients, TensorFlow will try to compute the gradients all the way back to the code that generates the anchor box refinement. But we already handle learning the anchor refinement in the RPN, so we don't want to influence that with additional gradients from stage 2. So, the sooner we stop it, the more unnecessary computation we avoid. Further, it's not clear (at least I haven't looked into it) how the gradients calculation back through crop_and_resize works.