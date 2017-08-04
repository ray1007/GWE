import cPickle as pkl
import pdb
import random
import sys
import time

import numpy as np
from PIL import Image
import tensorflow as tf

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

  
log_dir = './log'
dict_pkl_filename = '../data/char_word_dicts.pkl'
char_dict_pkl_filename = '../data/char_dict.pkl'

# load char bitmap data
with open(char_dict_pkl_filename, 'rb') as f:
  char_bitmap_dict = pkl.load(f)

#char_bitmap_dict.pop(u'UNK')
chars = []
bitmaps = []
for k in char_bitmap_dict:
  chars.append(k)
  bitmaps.append(char_bitmap_dict[k])

pickList = range(0, len(bitmaps))
random.shuffle(pickList)

MAX_EPOCH = 100
BATCH_SIZE = 20

def mask_activation(act, n):
  mask = np.zeros_like(act)
  for b in xrange(BATCH_SIZE):
    values = np.reshape(np.abs(act[b]), (-1,))
    values = np.sort(values)[::-1]
    threshold = values[n]
    indices = np.where(np.abs(act[b]) > threshold)
    mask[b,indices[0],indices[1],indices[2]] = 1
    
  return act * mask

with tf.Graph().as_default(), tf.Session() as sess:
  # initialize model.
  model = Model(log_dir, sess, True)

  n_activations = [75, 50, 25, 5, 1]
  for layer_idx in xrange(5):
    for epoch in xrange(MAX_EPOCH):
      tStart = time.time()
      print "Epoch:",epoch+1
      cost = 0
      l1_loss = 0
      random.shuffle(pickList)
      for batch_num in xrange(0, int(len(pickList)/BATCH_SIZE)):
        tStart_batch = time.time()
        start = batch_num * BATCH_SIZE    
        end   = (batch_num+1) * BATCH_SIZE
        if end > len(pickList):
          start = len(pickList)-BATCH_SIZE
          end = len(pickList)
        
        bitmap_batch_list = [ bitmaps[idx] / 255.0 for idx in pickList[start:end] ]
        X = np.stack(bitmap_batch_list, axis=0)

        batch_cost, batch_l1_loss = model.train_layer(layer_idx, X)

        batch_time = time.time() - tStart_batch
        cost += batch_cost
        l1_loss += batch_l1_loss
        sys.stdout.write( 
          ('\rbatch #{0}, '
           'loss_val: {1}, '
           'l1_loss: {2}, '
           'total_batch_loss: {3}, '
           'epoch_time: {4} ').format(
          batch_num+1, batch_cost, batch_l1_loss, cost, batch_time))
        sys.stdout.flush()

      average_cost = cost/float(int(len(pickList)/BATCH_SIZE))
      print "Total cost =",cost," ,Average cost =",average_cost
      print "Total l1_loss = ",l1_loss
      tEnd = time.time()
      print "Time used:", tEnd-tStart

    print "Finished training layer {0}!".format(layer_idx+1)

    ''' 
    The following code is used to generate reconstruction of character glyphs.

    test_bitmap_batch = [ bitmaps[idx] / 255.0 for idx in pickList[:BATCH_SIZE] ]
    X = np.stack(test_bitmap_batch, axis=0)
    X_hat = model.test_layer(layer_idx, X)

    selected_imgs = np.zeros((20,60,60,1))
    selected_imgs[0:20:2, :,:,:] = X[:10,:,:,:]
    selected_imgs[1:20:2, :,:,:] = X_hat[:10,:,:,:]

    save_collection_img("conv_ae_char_nopool_layerwise_l{0}.png".format(layer_idx+1),
                        n_row=2, n_col=10,
                        img_size=60, offset=20,
                        imgs=selected_imgs)

    l1, l2, l3, l4, l5 = model.get_layers_n_args(X)
    acts = [l1, l2, l3, l4, l5]

    ma = mask_activation(acts[layer_idx], n_activations[layer_idx])
    images_hat = model.reconstruct_from_layer(layer_idx, ma)

    img_filename = "analyze_conv_ae_nopool_layerwise_l{0}_{1}.png".format(layer_idx+1, n_activations[layer_idx])
    save_collection_img(img_filename,
                        n_row=2, n_col=10,
                        img_size=60, offset=20,
                        imgs=images_hat)

    print "Finished analyzing layer {0}!".format(layer_idx+1)
    '''

  model.save_model(100)
  '''
  embs = model.get_embs(X)
  val = np.mean(embs)
  
  embs = np.zeros((512,1,1,512))
  for i in xrange(512):
    embs[i,0,0,i] = val
  reconstructed_imgs = np.zeros((512,60,60,1))
  
  imgs = model.reconstruct_from_embs(embs[:100])
  reconstructed_imgs[:100,:,:,:] = imgs
  imgs = model.reconstruct_from_embs(embs[100:200])
  reconstructed_imgs[100:200,:,:,:] = imgs
  imgs = model.reconstruct_from_embs(embs[200:300])
  reconstructed_imgs[200:300,:,:,:] = imgs
  imgs = model.reconstruct_from_embs(embs[300:400])
  reconstructed_imgs[300:400,:,:,:] = imgs
  imgs = model.reconstruct_from_embs(embs[400:500])
  reconstructed_imgs[400:500,:,:,:] = imgs
  imgs = model.reconstruct_from_embs(embs[412:])
  reconstructed_imgs[412:,:,:,:] = imgs
  
  save_collection_img("conv_ae_nopool_emb.png",
                      n_row=16, n_col=32,
                      img_size=60, offset=15,
                      imgs=reconstructed_imgs)
  '''

