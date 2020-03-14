from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from transformer_support_functions import *
from Optimiser import CustomSchedule
from Transformer import Transformer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_val_split = ['train[:85%]', 'train[85%:]']

reddit_train, reddit_validation = tfds.load(
    name='reddit_tifu/long', split=train_val_split, as_supervised=True)

tokeniser_post = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (post.numpy() for post, tldr in reddit_train), target_vocab_size=2**13)

tokenizer_tldr = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (tldr.numpy() for post, tldr in reddit_train), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokeniser_post.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokeniser_post.decode(tokenized_string)
print('The original string: {}'.format(original_string))


def encode(post, tldr):
    post = [tokeniser_post.vocab_size] + tokeniser_post.encode(
        post.numpy()) + [tokeniser_post.vocab_size+1]
    tldr = [tokenizer_tldr.vocab_size] + tokenizer_tldr.encode(
        tldr.numpy()) + [tokenizer_tldr.vocab_size+1]
    return post, tldr


def tf_encode(posts, tldrs):
    results_posts, results_tldrs = tf.py_function(
        encode, [posts, tldrs], [tf.int64, tf.int64])
    results_posts.set_shape([None])
    results_tldrs.set_shape([None])
    return results_posts, results_tldrs


train_dataset = reddit_train.map(tf_encode)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(
    BUFFER_SIZE).padded_batch(BATCH_SIZE, ([None], [None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

train_dataset.as_numpy_iterator()

val_dataset = reddit_validation.map(tf_encode)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset.as_numpy_iterator()

pt_batch, en_batch = next(iter(val_dataset))
print(pt_batch)
print(en_batch)


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokeniser_post.vocab_size + 2
target_vocab_size = tokenizer_tldr.vocab_size + 2
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

EPOCHS = 20

for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> post, tar -> tldr
  for (batch, (inp, tar)) in enumerate(train_dataset):
    train_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
