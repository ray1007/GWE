import os

import numpy as np
import tensorflow as tf

def unpool(updates, ksize=[1, 2, 2, 1]):
    original_shape = updates.get_shape()
    original_shape = tuple([i.__int__() for i in original_shape])
    new_size = tf.shape(updates)[1:3]
    new_size *= tf.constant(np.array([ksize[1], ksize[2]]).astype('int32'))
    #input_shape = updates.get_shape().as_list()
    #new_size = tf.to_int32(tf.pack([input_shape[1]*ksize[1], input_shape[2]*ksize[2]]))

    ret = tf.image.resize_nearest_neighbor(updates, new_size)
    ret.set_shape((original_shape[0], 
                   original_shape[1] * ksize[1] if original_shape[1] is not None else None,
                   original_shape[2] * ksize[2] if original_shape[2] is not None else None,
                   original_shape[3]))
    
    return ret

BATCH_SIZE = 20

class Model:
  def __init__(self, 
               log_dir,
               session,
               layerwise=False):
    self._log_dir = log_dir
    self._session = session
    if layerwise:
      self.build_layer_wise_graph()
    else:
      self.build_graph()

  def conv_n_kernel(self, x, kernel_shape, scope_name, strides=[1,1,1,1], padding='VALID', act=tf.nn.relu6):
    with tf.variable_scope(scope_name) as scope:
      kernel = tf.get_variable("kernel", 
                               kernel_shape,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
      biases = tf.get_variable("biases",[kernel_shape[-1]],initializer=tf.constant_initializer(0.0))
      conv = tf.nn.conv2d(x, kernel, strides, padding=padding)
      bias = tf.nn.bias_add(conv, biases)
      x = act(bias, name=scope.name)
    return x, kernel

  def conv_transpose(self, x, kernel, scope_name, strides=[1,1,1,1], padding='VALID', act=tf.nn.relu6):
    x_shape = x.get_shape().as_list()
    W_shape = kernel.get_shape().as_list()
    if padding == 'SAME':
      out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
    elif padding == 'VALID':
      out_shape = tf.pack([x_shape[0],
                           (x_shape[1] - 1)*strides[1]+W_shape[0],
                           (x_shape[2] - 1)*strides[2]+W_shape[1],
                           W_shape[2]])
    with tf.variable_scope(scope_name) as scope:
      biases = tf.get_variable("biases",[W_shape[2]],initializer=tf.constant_initializer(0.0))

      deconv = tf.nn.conv2d_transpose(x,
                                      kernel,
                                      output_shape=out_shape,
                                      strides=strides,
                                      padding=padding)
      bias = tf.nn.bias_add(deconv, biases)
      x = act(bias, name=scope.name)
    return x

  def forward(self, images):
    #x = 60
    # conv1
    x, kernel1 = self.conv_n_kernel(images, [5,5,1,16], "conv1", strides=[1,1,1,1])
    feat_map_l1_norm = tf.reduce_sum(tf.abs(x))
    
    #x = 56
    # conv2
    x, kernel2 = self.conv_n_kernel(x, [4,4,16,64], 'conv2', strides=[1,2,2,1])
    feat_map_l1_norm += tf.reduce_sum(tf.abs(x))

    # x = 27
    # conv3
    x, kernel3 = self.conv_n_kernel(x, [5,5,64,64], 'conv3', strides=[1,2,2,1])
    feat_map_l1_norm += tf.reduce_sum(tf.abs(x))

    # x = 12
    # conv4
    x, kernel4 = self.conv_n_kernel(x, [4,4,64,128], 'conv4', strides=[1,2,2,1])
    feat_map_l1_norm += tf.reduce_sum(tf.abs(x))
   
    # x = 5
    # conv5
    x, kernel5 = self.conv_n_kernel(x, [5,5,128,512], 'conv5', strides=[1,1,1,1])
    feat_map_l1_norm += tf.reduce_sum(tf.abs(x))

    # x = 1
    self._embs = x
    #deconv5
    x = self.conv_transpose(x, kernel5, 'deconv5', strides=[1,1,1,1])

    # x = 5
    #deconv4
    #x, _ = self.conv_n_kernel(x, [5,5,128,64], "deconv4")
    x = self.conv_transpose(x, kernel4, 'deconv4', strides=[1,2,2,1])

    # x = 12
    #deconv3
    #x, _ = self.conv_n_kernel(x, [5,5,256,32], "deconv3")
    x = self.conv_transpose(x, kernel3, 'deconv3', strides=[1,2,2,1])

    # x = 27
    #deconv2
    #x, kernel2 = self.conv_n_kernel(x, [5,5,32,16], "deconv2")
    x = self.conv_transpose(x, kernel2, 'deconv2', strides=[1,2,2,1])

    # x = 56
    #deconv1
    #images_hat, _ = self.conv_n_kernel(x, [5,5,16,1], "deconv1", act=tf.nn.sigmoid)
    images_hat = self.conv_transpose(x, kernel1, 'deconv1', strides=[1,1,1,1], act=tf.nn.sigmoid)
    # x = 60

    return images_hat, feat_map_l1_norm

  def layer_wise_forward(self, images):
    # conv1
    x, kernel1 = self.conv_n_kernel(images, [5,5,1,16], "conv1", strides=[1,1,1,1])
    l1_norm = tf.reduce_sum(tf.abs(x))
    self._conv1 = x

    #l1_deconv1
    l1_img_hat = self.conv_transpose(x, kernel1, 'l1_deconv1', strides=[1,1,1,1])

    # conv2
    x, kernel2 = self.conv_n_kernel(x, [4,4,16,16], 'conv2', strides=[1,2,2,1])
    l2_norm = tf.reduce_sum(tf.abs(x))
    self._conv2 = x

    #l2_deconv2
    l2_x = self.conv_transpose(x, kernel2, 'l2_deconv2', strides=[1,2,2,1])
    #l2_deconv1
    l2_img_hat = self.conv_transpose(l2_x, kernel1, 'l2_deconv1', strides=[1,1,1,1])

    # conv3
    x, kernel3 = self.conv_n_kernel(x, [5,5,16,256], 'conv3', strides=[1,2,2,1])
    l3_norm = tf.reduce_sum(tf.abs(x))
    self._conv3 = x

    #l3_deconv3
    l3_x = self.conv_transpose(x, kernel3, 'l3_deconv3', strides=[1,2,2,1])
    #l3_deconv2
    l3_x = self.conv_transpose(l3_x, kernel2, 'l3_deconv2', strides=[1,2,2,1])
    #l3_deconv1
    l3_img_hat = self.conv_transpose(l3_x, kernel1, 'l3_deconv1', strides=[1,1,1,1])

    # conv4
    x, kernel4 = self.conv_n_kernel(x, [4,4,256,256], 'conv4', strides=[1,2,2,1])
    l4_norm = tf.reduce_sum(tf.abs(x))
    self._conv4 = x

    #l4_deconv4
    l4_x = self.conv_transpose(x, kernel4, 'l4_deconv4', strides=[1,2,2,1])
    #l4_deconv3
    l4_x = self.conv_transpose(l4_x, kernel3, 'l4_deconv3', strides=[1,2,2,1])
    #l4_deconv2
    l4_x = self.conv_transpose(l4_x, kernel2, 'l4_deconv2', strides=[1,2,2,1])
    #l4_deconv1
    l4_img_hat = self.conv_transpose(l4_x, kernel1, 'l4_deconv1', strides=[1,1,1,1])
   
    # conv5
    x, kernel5 = self.conv_n_kernel(x, [5,5,256,512], 'conv5', strides=[1,1,1,1])
    l5_norm = tf.reduce_sum(tf.abs(x))
    self._conv5 = x

    #l5_deconv5
    l5_x = self.conv_transpose(x, kernel5, 'l5_deconv5', strides=[1,1,1,1])
    #l5_deconv4
    l5_x = self.conv_transpose(l5_x, kernel4, 'l5_deconv4', strides=[1,2,2,1])
    #l5_deconv3
    l5_x = self.conv_transpose(l5_x, kernel3, 'l5_deconv3', strides=[1,2,2,1])
    #l5_deconv2
    l5_x = self.conv_transpose(l5_x, kernel2, 'l5_deconv2', strides=[1,2,2,1])
    #l5_deconv1
    l5_img_hat = self.conv_transpose(l5_x, kernel1, 'l5_deconv1', strides=[1,1,1,1])
   
    return l1_img_hat, l1_norm, l2_img_hat, l2_norm, \
           l3_img_hat, l3_norm, l4_img_hat, l4_norm, l5_img_hat, l5_norm

  def loss(self, images, images_hat):
    loss = tf.reduce_sum( tf.square(images - images_hat) )
    #loss = tf.reduce_sum( images_hat * tf.log(images) )
    return loss

  def optimize(self, loss, var_list=None):
    optimizer = tf.train.AdagradOptimizer(0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
    #                                   beta1=0.9, 
    #                                   beta2=0.999, 
    #                                   epsilon=1e-08)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if var_list:
      train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
    else:
      train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

  def build_graph(self):
    print "conv_ae_char_nopool.py :: build_graph()"
    images = tf.placeholder(tf.float32, (BATCH_SIZE, 60, 60, 1))
    keep_prob = tf.placeholder(tf.float32)
    self._images = images
    self._keep_prob = keep_prob

    images_drop = tf.nn.dropout(images, keep_prob)
    images_hat, l1_loss = self.forward(images_drop)
    self._images_hat = images_hat

    loss = self.loss(images, images_hat)
    self._loss = loss
    self._l1_loss = l1_loss
    self._train_op = self.optimize(loss+0.002*l1_loss)

    tf.global_variables_initializer().run()

    if self._log_dir != '':
      self._summary_writer = tf.summary.FileWriter(self._log_dir,
                                                   self._session.graph)
      self._summary_writer.flush()

    self._saver = tf.train.Saver()

  def build_layer_wise_graph(self):
    print "conv_ae_char_nopool.py :: build_layer_wise_graph()"
    images = tf.placeholder(tf.float32, (BATCH_SIZE, 60, 60, 1))
    self._images = images

    self._l1_img_hat, self._l1_norm, self._l2_img_hat, self._l2_norm, \
    self._l3_img_hat, self._l3_norm, self._l4_img_hat, self._l4_norm, \
    self._l5_img_hat, self._l5_norm = self.layer_wise_forward(images)

    self._l1_loss = self.loss(images, self._l1_img_hat)
    self._l2_loss = self.loss(images, self._l2_img_hat)
    self._l3_loss = self.loss(images, self._l3_img_hat)
    self._l4_loss = self.loss(images, self._l4_img_hat)
    self._l5_loss = self.loss(images, self._l5_img_hat)

    l1_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')
    l1_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l1_deconv1')

    l2_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2')
    l2_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l2_deconv2')
    l2_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l2_deconv1')

    l3_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv3')
    l3_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l3_deconv3')
    l3_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l3_deconv2')
    l3_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l3_deconv1')

    l4_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv4')
    l4_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l4_deconv4')
    l4_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l4_deconv3')
    l4_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l4_deconv2')
    l4_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l4_deconv1')

    l5_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv5')
    l5_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l5_deconv5')
    l5_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l5_deconv4')
    l5_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l5_deconv3')
    l5_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l5_deconv2')
    l5_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l5_deconv1')

    self._train_l1 = self.optimize(self._l1_loss + 0.002*self._l1_norm, var_list=l1_var_list)
    self._train_l2 = self.optimize(self._l2_loss + 0.001*self._l2_norm, var_list=l2_var_list)
    self._train_l3 = self.optimize(self._l3_loss + 0.01*self._l3_norm, var_list=l3_var_list)
    self._train_l4 = self.optimize(self._l4_loss + 0.1*self._l4_norm, var_list=l4_var_list)
    self._train_l5 = self.optimize(self._l5_loss + 0.1*self._l5_norm, var_list=l5_var_list)

    tf.global_variables_initializer().run()

    if self._log_dir != '':
      self._summary_writer = tf.summary.FileWriter(self._log_dir,
                                                   self._session.graph)
      self._summary_writer.flush()

    self._saver = tf.train.Saver()

  def add_summary(self, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self._summary_writer.add_summary(summary, step)
    self._summary_writer.flush()

  def save_model(self, step):
    self._saver.save(self._session,
                     os.path.join(self._log_dir, 'conv_ae_char_nopool.ckpt'),
                     global_step=step)

  def load_model(self, model_path):
    self._saver.restore(self._session, model_path)

  def train_layer(self, layer_idx, images):
    if layer_idx == 0:
      _, loss_val, norm = self._session.run([self._train_l1, self._l1_loss, self._l1_norm], {
        self._images: images})
    elif layer_idx == 1:
      _, loss_val, norm = self._session.run([self._train_l2, self._l2_loss, self._l2_norm], {
        self._images: images})
    elif layer_idx == 2:
      _, loss_val, norm = self._session.run([self._train_l3, self._l3_loss, self._l3_norm], {
        self._images: images})
    elif layer_idx == 3:
      _, loss_val, norm = self._session.run([self._train_l4, self._l4_loss, self._l4_norm], {
        self._images: images})
    elif layer_idx == 4:
      _, loss_val, norm = self._session.run([self._train_l5, self._l5_loss, self._l5_norm], {
        self._images: images})

    return loss_val, norm

  def train_batch(self, images, keep_prob):
    _, loss_val, l1_norm = self._session.run([self._train_op, self._loss, self._l1_loss], {
      self._images: images,
      self._keep_prob: keep_prob})

    return loss_val, l1_norm

  def test_layer(self, layer_idx, images):
    if layer_idx == 0:
      images_hat = self._session.run(self._l1_img_hat, {self._images: images})
    elif layer_idx == 1:
      images_hat = self._session.run(self._l2_img_hat, {self._images: images})
    elif layer_idx == 2:
      images_hat = self._session.run(self._l3_img_hat, {self._images: images})
    elif layer_idx == 3:
      images_hat = self._session.run(self._l4_img_hat, {self._images: images})
    elif layer_idx == 4:
      images_hat = self._session.run(self._l5_img_hat, {self._images: images})

    return images_hat

  def test_batch(self, images):
    images_hat = self._session.run(self._images_hat, {
      self._images: images,
      self._keep_prob: 1})

    return images_hat

  def get_embs(self, images):
    embs_val = self._session.run(self._embs, {
      self._images: images,
      self._keep_prob: 1})
    return embs_val

  def get_layers_n_args(self, images):
    l1, l2, l3, l4, l5 = self._session.run(
      [self._conv1, self._conv2, self._conv3, self._conv4, self._conv5], 
      {self._images: images})
    
    return l1, l2, l3, l4, l5
        
  def reconstruct_from_layer(self, layer_idx, act):
    if layer_idx == 0:
      images_hat = self._session.run(self._l1_img_hat, {self._conv1: act})
    elif layer_idx == 1:
      images_hat = self._session.run(self._l2_img_hat, {self._conv2: act})
    elif layer_idx == 2:
      images_hat = self._session.run(self._l3_img_hat, {self._conv3: act})
    elif layer_idx == 3:
      images_hat = self._session.run(self._l4_img_hat, {self._conv4: act})
    elif layer_idx == 4:
      images_hat = self._session.run(self._l5_img_hat, {self._conv5: act})

    return images_hat

  def reconstruct_from_embs(self, embs):
    images_hat = self._session.run(self._images_hat, {
      #self._images: images,
      self._embs: embs})
    return images_hat

