---
title: 'TensorFlow输入数据处理框架'
date: 2018-07-25 21:55:36
tags:
	- 深度学习
	- TensorFlow
---
Tensoflow提供了一种统一的数据格式来存储数据，这个格式就是TFrecords，基于TFrecords我们构建一个完整的TensorFlow输入数据处理框架，以COCO数据集为例，介绍了COCO数据集的TFrecords文件制作，以及读取解析的过程，以此来介绍一个构建文件处理框架的过程。

<!-- more -->

## TFrecords格式介绍
TFrecords是一种二进制文件，通过tf.train.Example Protocol Buffer的格式存储数据，以下的代码给出了tf.train.Example的定义。

	message Example {
	    Features features = 1;
	};
	message Features {
	    map<string, Feature> feature = 1;
	};
	message Feature {
	    oneof kind {
	    BytesList bytes_list = 1;
	    FloatList float_list = 2;
	    Int64List int64_list = 3;
	}
	};

tf.train.Example包含了一个从属性名称到取值的字典，其中属性名称为一个字符串，属性取值可以是字符串(BytesList)，实数列表(FloatList)或者整数列表(Int64List），比如将解码前的图像存为一个字符串，将lable存为整数列表，或者将bounding box存为实数列表。

## COCO数据集的TFrecords文件制作
COCO数据集是微软做的一个比较大的数据集，可以用来做图像的recognition、segmentation、captioning，我用来做物体检测识别。官方也提供了API操作数据集（[https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi "https://github.com/cocodataset/cocoapi")）。根据链接介绍下载安装python的API后，就可以开始Tfrecords的文件制作了。

	from pycocotools.coco import COCO
	import tensorflow as tf
	import numpy as np
	from PIL import Image
	from time import time
	import os
	
	dataDir='/home/zju/lkj/data/COCO Dataset'
	dataType='train2017'
	annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
	
	classes = ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
	            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
	            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
	            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
	            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
	
	# initialize COCO api for instance annotations
	coco = COCO(annFile)
	classesId = coco.getCatIds(classes)
	imgIds = coco.getImgIds()
	img_filters=[]
	for imgId in imgIds:
	    Anns = coco.loadAnns(coco.getAnnIds(imgIds=imgId))
	    annIds = list(map(lambda x:x['category_id'],Anns))
	    for annId in annIds:
	        if annId in classesId:
	            img_filters.append(imgId)
	img_filters = set(img_filters)
	
	
	# 归一化
	# size: 图片大小
	# box：[x,y,w,h]
	# return 归一化结果
	def convert(size,box):
	    dw = 1./size[0]
	    dh = 1./size[1]
	    x = box[0]+box[2]/2.0
	    y = box[1]+box[3]/2.0
	    x = x*dw
	    w = box[2]*dw
	    y = y*dh
	    h = box[3]*dh
	    return [x,y,w,h]
	
	
	def convert_img(img_id):
	    img_id_str = str(img_id).zfill(12)
	    img_path = '{}/{}/{}.jpg'.format(dataDir, dataType, img_id_str)
	    image = Image.open(img_path)
	    resized_image = image.resize((416, 416), Image.BICUBIC)
	    image_data = np.array(resized_image, dtype='float32') / 255
	    if image_data.size != 519168: # 不为3通道
	        return False
	    img_raw = image_data.tobytes()
	    return img_raw
	
	def convert_annotation(image_id):
	    img_info = coco.loadImgs(image_id)[0]  # 读入的是照片的详细信息，而非图像信息, 返回的是list，只有1个id输入时，取0
	    w = int(img_info['width'])
	    h = int(img_info['height'])
	    bboxes = []
	    Anns = coco.loadAnns(ids=coco.getAnnIds(imgIds=image_id))
	    i = 0
	    for Ann in Anns:
	        if i>29:
	            break
	        iscrowd = Ann['iscrowd']
	        if iscrowd == 1:
	            continue
	        if Ann['category_id'] not in classesId:
	            continue
	        cls_id = classesId.index(Ann['category_id'])  # 取新的编号
	        bbox = Ann['bbox']
	        bb = convert((w, h), bbox) + [cls_id]
	        bboxes.extend(bb)
	        i = i + 1
	
	    if len(bboxes) < 30*5:
	        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))
	    return np.array(bboxes, dtype=np.float32).flatten().tolist()
	
	filename = os.path.join('train2017'+'.tfrecords')
	writer = tf.python_io.TFRecordWriter(filename)
	i=0
	start = time()
	for imgId in img_filters:
	    xywhc = convert_annotation(imgId)
	    img_raw = convert_img(imgId)
	    if img_raw:
	        example = tf.train.Example(features=tf.train.Features(feature={
	            'xywhc':
	                    tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
	            'img':
	                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
	            }))
	        writer.write(example.SerializeToString())
			# 显示制作进度，剩余时间
	        if i%100==99:
	            t = time()-start
	            print(i,'t={:0.4f}s/100 step'.format(t),'  left time={:0.4f}s'.format((len(img_filters)-i)*t/100))
	            start = time()
	        i = i+1
	print('Done!')
	writer.close()

下面分段对代码进行介绍，这个数据制作是应用于物品检查与分割，并且只有部分物品，所以在程序开头有classes列举（总共45种，完整的COCO数据集包含91种）。COCO数据集中混有灰度图，所以在reshape的时候会一直报错，刚开始还一直想不清楚为什么，后来遍历原始数据集才发现有灰度图的存在，所以reshape成416\*416\*3会报错,所以程序有一个判断是否为3通道：

	image = Image.open(img_path)
    resized_image = image.resize((416, 416), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32') / 255
    if image_data.size != 519168: # 不为3通道
        return False

图像读取后转换成字符串(BytesList):

	img_raw = image_data.tobytes()

bounding box转换成实数列表(FloatList):

	return np.array(bboxes, dtype=np.float32).flatten().tolist()

基于此，核心的构建部分为：

	example = tf.train.Example(features=tf.train.Features(feature={
	    'xywhc':
	            tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
	    'img':
	            tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
	    }))
	writer.write(example.SerializeToString())

## TFrecords文件读取解析
对应构建时候的数据格式，进行解析，可以加一些程序对于读取后的图像文件的一些进一步处理，比如图像增强

	def parser(example):
	    features = {
	                'xywhc': tf.FixedLenFeature([150], tf.float32),
	                'img': tf.FixedLenFeature((), tf.string)}
	    feats = tf.parse_single_example(example, features)
	    coord = feats['xywhc']
	    coord = tf.reshape(coord, [30, 5])
	
	    img = tf.decode_raw(feats['img'], tf.float32)
	    img = tf.reshape(img, [416, 416, 3])
	    img = tf.image.resize_images(img, [cfg.train.image_resized, cfg.train.image_resized])
	    rnd = tf.less(tf.random_uniform(shape=[], minval=0, maxval=2), 1)
		# 添加对于读取后的图像文件的一些进一步处理，图像增强
		def flip_img_coord(_img, _coord):
	        zeros = tf.constant([[0, 0, 0, 0, 0]]*30, tf.float32)
	        img_flipped = tf.image.flip_left_right(_img)
	        idx_invalid = tf.reduce_all(tf.equal(coord, 0), axis=-1)
	        coord_temp = tf.concat([tf.minimum(tf.maximum(1 - _coord[:, :1], 0), 1),
	                               _coord[:, 1:]], axis=-1)
	        coord_flipped = tf.where(idx_invalid, zeros, coord_temp)
	        return img_flipped, coord_flipped

	    img, coord = tf.cond(rnd, lambda: (tf.identity(img), tf.identity(coord)), lambda: flip_img_coord(img, coord))
	
	    img = tf.image.random_hue(img, max_delta=0.1)
	    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
	    img = tf.image.random_brightness(img, max_delta=0.1)
	    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
	    img = tf.minimum(img, 1.0)
	    img = tf.maximum(img, 0.0)
	    return img, coord

然后构建一个data_pipeline来作为训练数据的输入框架：
	
	def data_pipeline(file_tfrecords, batch_size):
	    dt = tf.data.TFRecordDataset(file_tfrecords)
	    dt = dt.map(parser, num_parallel_calls=4)
	    dt = dt.prefetch(batch_size)
	    dt = dt.shuffle(buffer_size=20*batch_size)
	    dt = dt.repeat()
	    dt = dt.batch(batch_size)
	    iterator = dt.make_one_shot_iterator()
	    imgs, true_boxes = iterator.get_next()
	
	    return imgs, true_boxes

测试一下整个数据输入模块：

	file_path = 'train2007.tfrecords'
    imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)
    sess = tf.Session()
    imgs_, true_boxes_ = sess.run([imgs, true_boxes])
    print(imgs_.shape, true_boxes_.shape)
    for imgs_i, boxes_ in zip(imgs_, true_boxes_):
        valid = (np.sum(boxes_, axis=-1) > 0).tolist()
        print([cfg.names[int(idx)] for idx in boxes_[:, 4][valid].tolist()])
        plt.figure()
        plt.imshow(imgs_i)
    plt.show()



