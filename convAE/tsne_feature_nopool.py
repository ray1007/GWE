# -*- coding: utf-8 -*-

import cPickle as pkl
import pdb
import sys

import numpy as np
#from PIL import Image
import tensorflow as tf
from tsne import bh_sne

sys.path.insert(0, './models/')
from conv_ae_char_nopool import Model

def save_collection_img(img_filename, n_row, n_col, img_size, offset, imgs):
  image=Image.new("RGB", (n_col*img_size + (n_col+1)*offset,
                          n_row*img_size + (n_row+1)*offset), 'black')
  pixels = image.load()
  offset_h = offset
  offset_w = offset
  for n_h in xrange(n_row):
    offset_w = offset
    for n_w in xrange(n_col):
      feat_idx = n_col * n_h + n_w
  
      bitmap = imgs[feat_idx]
      for p_h in xrange(img_size):
        for p_w in xrange(img_size):
          mag = bitmap[p_h, p_w] * 255
          pixels[offset_w+p_w, offset_h+p_h] = (mag,mag,mag)
      offset_w += offset + img_size
    offset_h += offset + img_size
  
  image.save(img_filename)


#checkpoint_filename = './checkpoints/conv_ae_char_switch_var.ckpt-100'
#checkpoint_filename = './log/conv_ae_char_switch_var.ckpt-100'
checkpoint_filename = './checkpoints/conv_ae_char_nopool.ckpt-100'
char_dict_pkl_filename = '../data/char_dict.pkl'

# load char bitmap data
with open(char_dict_pkl_filename, 'rb') as f:
  char_bitmap_dict = pkl.load(f)

feats = []

bitmaps = []
chars = []
for k in char_bitmap_dict:
  chars.append(k)
  bitmaps.append(char_bitmap_dict[k])

BATCH_SIZE = 20

with tf.Graph().as_default(), tf.Session() as sess:
  model = Model('', sess, True)
  model.load_model(checkpoint_filename)

  for batch_num in xrange(0, int(len(chars)/BATCH_SIZE)):
    start = batch_num * BATCH_SIZE    
    end   = (batch_num+1) * BATCH_SIZE
    if end > len(chars):
      start = len(chars)-BATCH_SIZE
      end = len(chars)
    
    bitmap_batch_list = [ bitmaps[idx] / 255.0 for idx in xrange(start,end) ]
    X = np.stack(bitmap_batch_list, axis=0)

    conv1, conv2, conv3, conv4, conv5 = model.get_layers_n_args(X)
    for idx in xrange(BATCH_SIZE):
      #char_feat_dict[chars[start+idx]] = embs[idx,0,0,:]
      feats.append(conv5[idx,0,0,:])
      #feats.append(np.reshape(conv4[idx], (-1,)))

    #embs = model.get_embs(X)


vis_data = bh_sne(np.asarray(feats, dtype='float64'))
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

print vis_data.shape
#vis_x *= 3
#vis_y *= 3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

img_filename = 'tsne_conv_ae_feat_nopool_layerwise_conv5.png'
fig = plt.gcf()
fig.clf()
fig.set_size_inches(40, 40)
fig.set_dpi(80)
ax = plt.subplot(111)
ax.set_xlim([np.min(vis_x) - 5, np.max(vis_x) + 5])
ax.set_ylim([np.min(vis_y) - 5, np.max(vis_y) + 5])

for idx in xrange(8780):
  bitmap = np.tile(bitmaps[idx], (1,1,3))

  imagebox = OffsetImage(bitmap, zoom=0.5)
  xy = [vis_x[idx], vis_y[idx]]
  #pdb.set_trace()
  #xy = vis_data[idx]

  ab = AnnotationBbox(imagebox, xy,
                      #xybox=(10., -10.),
                      xycoords='data',
                      boxcoords="offset points",
                      pad=0)                                  
  ax.add_artist(ab)

  if idx % 100 == 0:
    print idx
  #if idx == 10:
  #  print idx
  #  break

ax.grid(True)
plt.draw()
plt.savefig(img_filename)
#plt.show()

'''
img_size = 60
image=Image.new("RGB", (int(np.max(vis_x) - np.min(vis_x)) + 2*img_size,
                        int(np.max(vis_y) - np.min(vis_y)) + 2*img_size), 'white')
pixels = image.load()
for idx in xrange(4000):
  bitmap = bitmaps[idx]
  offset_w, offset_h = int(vis_x[idx] - img_size), int(vis_y[idx] - img_size)
  for p_h in xrange(img_size):
    for p_w in xrange(img_size):
      mag = 255 - bitmap[p_h, p_w] * 255
      pixels[offset_w+p_w, offset_h+p_h] = (mag,mag,mag)

image.save(img_filename)
'''

