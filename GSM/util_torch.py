# python3.6                                
# encoding    : utf-8 -*-                            
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com                                    
# @file       : util_torch.py
# @Time       : 2021/8/17 22:15
import math

import random
import torch


def data_set(data_url):
  """process data input."""
  data = []
  word_count = []
  fin = open(data_url)
  while True:
    line = fin.readline()
    if not line:
      break
    id_freqs = line.split()
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
    # for id_freq in id_freqs:
      items = id_freq.split(':')
      # python starts from 0
      doc[int(items[0])-1] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
  fin.close()
  return data, word_count


# 将文档集以id的形式分为batch
def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))  # 给文档编号
    if shuffle:
        random.shuffle(ids)
    batch_count = math.floor(data_size / batch_size)
    for i in range(batch_count):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
    return batches


def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = torch.zeros((batch_size, vocab_size))
    count_batch = []
    mask = torch.zeros(batch_size)
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():  # data[doc_id]是一个字典
                data_batch[i, word_id] = freq  # bag-of-word representation of document
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


def create_vocab(vocab_url):
    vocab = []
    with open(vocab_url, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            vocab.append(line.strip("\n").split(" ")[0])
    return vocab
