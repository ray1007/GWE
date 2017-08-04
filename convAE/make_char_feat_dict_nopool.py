import cPickle as pkl
import pdb
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, './models/')
from conv_ae_char_nopool import Model

checkpoint_filename = './checkpoints/conv_ae_char_nopool.ckpt-100'
char_bitmap_dict_pkl_filename = '../data/char_dict.pkl'
char_feat_dict_filename = '../data/char_feat_dict.pkl'

char_feat_dict = {}

# load char bitmap data
with open(char_bitmap_dict_pkl_filename, 'rb') as f:
  char_bitmap_dict = pkl.load(f)

bitmaps = []
chars = []
for k in char_bitmap_dict:
  chars.append(k)
  bitmaps.append(char_bitmap_dict[k])

BATCH_SIZE = 20

with tf.Graph().as_default(), tf.Session() as sess:
  model = Model('', sess, layerwise=True)
  model.load_model(checkpoint_filename)

  for batch_num in xrange(0, int(len(chars)/BATCH_SIZE)):
    start = batch_num * BATCH_SIZE    
    end   = (batch_num+1) * BATCH_SIZE
    if end > len(chars):
      start = len(chars)-BATCH_SIZE
      end = len(chars)
    
    bitmap_batch_list = [ bitmaps[idx] / 255.0 for idx in xrange(start,end) ]
    X = np.stack(bitmap_batch_list, axis=0)
    
    #embs = model.get_embs(X)
    _, _, _, _, embs = model.get_layers_n_args(X)
    for idx in xrange(BATCH_SIZE):
      char_feat_dict[chars[start+idx]] = embs[idx,0,0,:]

    sys.stdout.write('\rbatch #{0} '.format(batch_num))
    sys.stdout.flush()

#pdb.set_trace()
with open(char_feat_dict_filename, 'wb') as f:
    pkl.dump(char_feat_dict, f, pkl.HIGHEST_PROTOCOL)

