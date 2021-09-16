# python3.6                                
# encoding    : utf-8 -*-                            
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com                                    
# @file       : GSMPytorch.py
# @Time       : 2021/8/19 15:49
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import util_torch as utils

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 这句话表示只有第1块gpu可见，其他gpu不可用，此时要注意第1块gpu已经变成第0块
device = torch.device("cuda:0")
print("--->torch cuda use: ", torch.cuda.current_device(), torch.cuda.device_count())

torch.manual_seed(2002)  # reproducability,保证每次随机化初始参数都是一样的


# 初始化topic_vec(K*L) 和 word_vec(V*L) 两个矩阵
def vec_init(dim1, dim2, constant=1):
    low = -constant*np.sqrt(6.0/(dim1 + dim2))
    high = constant*np.sqrt(6.0/(dim1 + dim2))
    return torch.Tensor(dim1, dim2).uniform_(low, high)
    # return torch.random_uniform((dim1, dim2),
    #                          minval=low, maxval=high,
    #                           dtype=torch.float32)


# Model
class GSM(nn.Module):
    def __init__(self, vocab_size, n_hidden, n_topic):
        super(GSM, self).__init__()
        self.f_pi = nn.Linear(vocab_size, n_hidden)
        self.f_pi_Drop = nn.Dropout(0.8)
        self.f_mean = nn.Linear(n_hidden, n_topic)
        self.f_log_sigma = nn.Linear(n_hidden, n_topic)
        self.f_theta = nn.Linear(n_topic, n_topic)
        # 初始化topic vector and word vector
        # self.topic_vec = nn.Parameter(vec_init(n_topic, n_hidden))
        # self.word_vec = nn.Parameter(vec_init(vocab_size, n_hidden))
        # 0-1均匀分布来初始化两个矩阵
        self.topic_vec = nn.Parameter(torch.rand(size=(n_topic, n_hidden)))
        self.word_vec = nn.Parameter(torch.rand(size=(vocab_size, n_hidden)))

    def encoder(self, x_in):
        doc_pi = F.leaky_relu(self.f_pi(x_in))  # BOW representation经过MLP
        doc_pi = self.f_pi_Drop(doc_pi)
        mean = self.f_mean(doc_pi)
        log_sigma = self.f_log_sigma(doc_pi)  # 标准差的对数
        return mean, log_sigma

    # 近似后验中采样（推断出）隐变量
    def sample(self, mean, log_sigma):
        sd = torch.exp(log_sigma)  # 标准差
        if torch.cuda.is_available():
            epsilon = Variable(torch.randn(sd.size()), requires_grad=False).cuda()  # Sample from standard normal
        else:
            epsilon = Variable(torch.randn(sd.size()), requires_grad=False)
        w = mean + torch.multiply(epsilon, sd)
        theta = torch.softmax(self.f_theta(w), dim=1)
        return theta

    def decoder(self, theta):
        # 注意这里实际上是p(d|theta) = log(theta * beta)，需要输入theta来生成d，表达式就是d = log(theta * beta)
        # 输出就是重构样本d
        self.beta = torch.softmax(torch.matmul(self.topic_vec, torch.transpose(self.word_vec, 0, 1)), dim=1)
        x_out = torch.log(torch.matmul(theta, self.beta))  # x_out的每一维就是log(p(wi|theta))
        return x_out

    def forward(self, x_in, sampleTime=1):
        # 第一步，inference network得到近似后验分布（变分分布）
        mean, log_sigma= self.encoder(x_in)
        # 第二步，从近似后验中采样推断出隐变量,并由推断出的隐变量重构X
        if sampleTime > 1:  # 多次采样
            x_out_sum = torch.zeros_like(input=x_in)
            for i in range(sampleTime):
                theta = self.sample(mean=mean, log_sigma=log_sigma)
                x_out_sum += self.decoder(theta)
            x_out = torch.divide(x_out_sum, sampleTime)
        else:  # 只采样一次
            theta = self.sample(mean=mean, log_sigma=log_sigma)
            x_out = self.decoder(theta)
        return x_out, mean, log_sigma


# Loss function
def criterion(x_out, x_in, mean, log_sigma, mask):
    # 多元高斯KL散度, KL(Norm(miu, sigma^2)||Norm(0, 1)) (set as standard Gaussian distribution, miu = 0.0, sigma = 1.0)
    # 求的是先验与近似后验的KL divergence
    kld_loss = -0.5 * torch.sum(1 + 2 * log_sigma - torch.square(mean) - torch.exp(2 * log_sigma), 1)
    # kld_loss = mask * kld_loss  # mask document paddings in batch
    kld_loss = torch.multiply(mask, kld_loss)
    # 重构误差
    recons_loss = -torch.sum(torch.multiply(x_out, x_in), dim=1)   # 求和
    # 重构误差 + KL Divergence
    loss = recons_loss + kld_loss
    loss = torch.sum(loss) / torch.sum(mask)
    recons_loss = torch.sum(recons_loss)
    batch_kld_loss = torch.sum(kld_loss) / torch.sum(mask)
    return recons_loss, batch_kld_loss, loss


# Training
def train(epochs=500):
    for epoch in range(epochs):
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)  # 将文档集以id的形式分为多个batch
        recons_loss_sum = 0.0  # 所有文档重构误差之和，计算perplexity
        word_count = 0  # 文档集总词数
        kld_sum = 0.0  # 一次epoch所有batch的kld的和
        for idx_batch in train_batches:  # 一次取一个batch的id
            # 根据id找到batch里面的文档对应的dict并形成BOW的形式（batch_size*vocab_size）
            data_batch, count_batch, mask = utils.fetch_data(train_set, train_count, idx_batch, vocab_size)
            if torch.cuda.is_available():
                x_in = data_batch.cuda()
                mask = mask.cuda()
            else:
                x_in = data_batch
            word_count += np.sum(count_batch)
            x_out, mean, log_sigma = model(x_in, sampleTime=10)
            recons_loss, batch_kld_loss, loss = criterion(x_out, x_in, mean, log_sigma, mask)
            recons_loss_sum += recons_loss.cpu().detach().numpy()
            kld_sum += batch_kld_loss.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print('Epoch : {}/{}'.format(epoch, epochs))
            print('--->Train: ', end="\t")
            epoch_ppx = np.exp(recons_loss_sum / word_count)  # 当前epoch完成后的困惑度
            epoch_kld = kld_sum / len(train_batches)
            print('| Corpus perplexity: {:.5f}'.format(epoch_ppx),  # perplexity for all docs
                  '| KLD: {:.5}'.format(epoch_kld),
                  )
            print('--->Validation: ', end="\t")
            valid_batches = utils.create_batches(len(valid_set), batch_size, shuffle=False)
            recons_loss_sum = 0.0  # 所有文档重构误差之和，计算perplexity
            word_count = 0  # 文档集总词数
            kld_sum = 0.0  # 一次epoch所有batch的kld的和
            for idx_batch in valid_batches:  # 一次取一个batch的id
                # 根据id找到batch里面的文档对应的dict并形成BOW的形式（batch_size*vocab_size）
                data_batch, count_batch, mask = utils.fetch_data(valid_set, valid_count, idx_batch, vocab_size)
                if torch.cuda.is_available():
                    x_in = data_batch.cuda()
                    mask = mask.cuda()
                else:
                    x_in = data_batch
                word_count += np.sum(count_batch)
                x_out, mean, log_sigma = model(x_in)
                recons_loss, batch_kld_loss, loss = criterion(x_out, x_in, mean, log_sigma, mask)
                recons_loss_sum += recons_loss.cpu().detach().numpy()
                kld_sum += batch_kld_loss.cpu().detach().numpy()
            epoch_ppx = np.exp(recons_loss_sum / word_count)  # 当前epoch完成后的困惑度
            epoch_kld = kld_sum / len(train_batches)
            print('| Corpus perplexity: {:.5f}'.format(epoch_ppx),  # perplexity for all docs
                  '| KLD: {:.5}'.format(epoch_kld), "\n",
                  "=" * 50
                  )

# testing
def test():
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)  # 将文档集以id的形式分为多个batch
    recons_loss_sum = 0.0  # 所有文档重构误差之和，计算perplexity
    word_count = 0  # 文档集总词数
    kld_sum = 0.0
    for idx_batch in test_batches:  # 一次取一个batch的id
        # 根据id找到batch里面的文档对应的dict并形成BOW的形式（batch_size*vocab_size）
        data_batch, count_batch, mask = utils.fetch_data(test_set, test_count, idx_batch, vocab_size)
        if torch.cuda.is_available():
            x_in = data_batch.cuda()
            mask = mask.cuda()
        else:
            x_in = data_batch
        word_count += np.sum(count_batch)
        x_out, mean, log_sigma = model(x_in)
        recons_loss, batch_kld_loss, loss = criterion(x_out, x_in, mean, log_sigma, mask)
        recons_loss_sum += recons_loss.cpu().detach().numpy()
        kld_sum += batch_kld_loss.cpu().detach().numpy()
    epoch_ppx = np.exp(recons_loss_sum / word_count)
    epoch_kld = kld_sum / len(test_batches)
    print('| Corpus perplexity: {:.5f}'.format(epoch_ppx),  # perplexity for all docs
          '| KLD: {:.5}'.format(epoch_kld))


if __name__ == '__main__':
    data_dir = "../data/20news"
    vocab_size = 2000
    n_hidden = 500
    n_topic = 50
    batch_size = 64
    learning_rate = 5e-5

    model = GSM(vocab_size, n_hidden, n_topic)
    if torch.cuda.is_available():
        model.cuda()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 网络中可以训练的参数（变量）
    print("trainable_variables:", model.state_dict().keys())
    # 先把数据给读进内存
    train_url = os.path.join(data_dir, 'train.feat')
    test_url = os.path.join(data_dir, 'test.feat')
    vocab_url = os.path.join(data_dir, 'vocab.new')
    # 每个文档以dict的形式,key是word_id, value是word_count, 然后所有文档存放在list中
    # train_count存放每个文档中有多少个词
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)
    vocab = utils.create_vocab(vocab_url)
    # 验证集
    valid_set = test_set[:1000]
    valid_count = test_count[:1000]
    print("==========train model==========")
    train()
    print("==========test model==========")
    test()
    # 结果写入输出文件中
    output_dir = data_dir + "/result_" + str(n_topic)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 输出所有主题
    print("=====top 15 words in each topic=====")
    topic_url = os.path.join(output_dir, 'topic_word.txt')
    beta = model.beta.cpu().detach().numpy()
    # print("topic word distribution:", beta)
    with open(topic_url, "a", encoding="utf-8") as output_topic:
        for topic_id, topic_dist in enumerate(beta, start=0):
            top_word_ids = topic_dist.argsort()[:-16:-1]
            topic = "Topic:" + str(topic_id) + "\n"
            for word_id in top_word_ids:
                topic += (vocab[word_id] + "\n")
            print(topic)
            output_topic.write(topic + "\n")
    # 输出所有文档的主题分布
    doc_url = os.path.join(output_dir, 'doc_topic.txt')
    with open(doc_url, "a", encoding="utf-8") as output_doc:
        for test_doc in test_set:
            # 得到文档的词袋表示
            doc_rep = torch.zeros(1, vocab_size)
            for word_id, freq in test_doc.items():  # data[doc_id]是一个字典
                doc_rep[0, word_id] = freq  # bag-of-word representation of document
            doc_rep = doc_rep.cuda()
            # 通过推理网络得到主题分布
            mean, log_sigma = model.encoder(doc_rep)
            theta = model.sample(mean=mean, log_sigma=log_sigma)
            theta = theta.cpu().detach().numpy()
            theta = theta[0]
            doc = ""
            for p_k in theta:
                doc += (str(format(p_k, ".5f")) + "\t")
            output_doc.write(doc + "\n")
    # 输出所有主题的embedding
    topic_vec_url = os.path.join(output_dir, 'topic_vec.txt')
    topic_vec = model.topic_vec.data.cpu().detach().numpy()
    with open(topic_vec_url, "a", encoding="utf-8") as output_topic_embedding:
        for topic_v in topic_vec:
            topic_embedding = ""
            for l in topic_v:
                topic_embedding += (str(l) + "\t")
            output_topic_embedding.write(topic_embedding + "\n")
    # 输出所有词的embedding
    word_vec_url = os.path.join(output_dir, 'word_vec.txt')
    word_vec = model.word_vec.data.cpu().detach().numpy()
    with open(word_vec_url, "a", encoding="utf-8") as output_word_embedding:
        for word_v in word_vec:
            word_embedding = ""
            for l in word_v:
                word_embedding += (str(l) + "\t")
            output_word_embedding.write(word_embedding + "\n")
