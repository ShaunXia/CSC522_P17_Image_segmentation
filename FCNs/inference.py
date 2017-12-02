#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:54:05 2017

@author: ck
"""

#%matplotlib inline

from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

sys.path.append("/home/ck/tf-image-segmentation/")
sys.path.append("/home/ck/models/slim/")

fcn_16s_checkpoint_path = \
 '/home/ck/tf_projects/segmentation/model_fcn32s_final.ckpt'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

slim = tf.contrib.slim


from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from scipy.misc import imread, imsave, imresize

number_of_classes = 21

inputpath = "/media/ck/New Volume1/Course/17 Fall/Project/BSDS300/images/test/"
#inputpath = "/home/ck/tf_projects/segmentation/input/"
outputpath = "/home/ck/tf_projects/segmentation/output/"
filenames = sorted(os.listdir(inputpath))

for i in range(0,1):
    from tf_image_segmentation.models.fcn_32s import FCN_32s
    tf.reset_default_graph()
    image_filename = '%s%s' % (inputpath,filenames[i] )
    imagefile = imread(image_filename)
    size1 = imagefile.shape[0]
    size2 = imagefile.shape[1]

    #image_filename = 'small_cat.jpg'

    image_filename_placeholder = tf.placeholder(tf.string)

    feed_dict_to_use = {image_filename_placeholder: image_filename}

    image_tensor = tf.read_file(image_filename_placeholder)

    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

    # Fake batch for image and annotation by adding
    # leading empty axis.
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

    # Be careful: after adaptation, network returns final labels
    # and not logits
    FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)


    pred, fcn_16s_variables_mapping = FCN_32s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

    # The op for initializing the variables.
    initializer = tf.local_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        sess.run(initializer)
        
        saver.restore(sess,
                      "/home/ck/tf_projects/segmentation/model_fcn8s_final.ckpt")
        
        image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)
        
        c=np.zeros((size1,size2,3))
        c[:,:,0]=pred_np.squeeze()*20
        d=np.concatenate([image_np,c],axis=1)
        imsave('%s%s.jpg' %(outputpath, filenames[i]), d)
        sess.close