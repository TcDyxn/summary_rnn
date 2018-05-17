# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:51:54 2018

@author: TcD-PC
"""

print('---开始构建模型---')

# coding: utf-8

# In[1]:

import jieba
import numpy as np
import pickle
import gc
import time
import re
import tensorflow as tf

# cache
cache_path = 'C:/data/data/news_sohusite_xml.full/'
# 文本路径
path_sogou = 'C:/data/data/news_sohusite_xml.full/news_sohusite_xml.html'#'C:/Users/TcD-PC/Desktop/SogouCS.html'
path_sogou = 'C:/Users/TcD-PC/Desktop/zhaiyao/SogouCS.WWW08.txt'
# words embeddings路径
path_word_embe = 'C:/Users/TcD-PC/Desktop/ae_output.txt'
path_word_model = 'C:/data/data/word2vec/news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim128.bin'

# 停用词路径
path_stop_words = 'C:/Users/TcD-PC/Desktop/zhaiyao/StopwordsCN.txt'

# 保存
def save(filename, contents): 
    fh = open(filename, 'w',encoding='utf-8',errors='ignore') 
    fh.write(contents) 
    fh.close()

# 中文字符+数字提取
def Chinese_word_extraction(content_raw):
        chinese_pattern = u"([\u4e00-\u9fa5]+|[\d]+|[a-zA-Z]+|,|，|、|。|;|；)"
        eng_replace = u"([a-zA-Z]+)"
        time_replace = u"(\d{4})年(\d{1,2})月|(\d{4})年|(\d{4})年(\d{1,2})月(\d{1,2})日|(\d{1,2})月(\d{1,2})日|(\d{1,2})日|(\d{1,2})月|(\d{1,4})点"
        d_replace = u"(\d+)"
        rank_replace = u"第(\d{1,4})"
        chi_pattern = re.compile(chinese_pattern)
        re_data = chi_pattern.findall(content_raw)
        content_clean  = ''.join(re_data)
        content_clean = re.sub(eng_replace,'english',content_clean)
        content_clean = re.sub(time_replace,'time',content_clean)
        content_clean = re.sub(rank_replace,'rank',content_clean)
        content_clean = re.sub(d_replace,'number',content_clean)
        return content_clean
# 停用词
f=open(path_stop_words,encoding='utf-8')
#加载停用词
stop_words=f.read().strip().split('\n')
f.close()
#stop_words=[]
#TODO 读取训练集
def load_preprocess():
    with open(cache_path+'summary_data.p', mode='rb') as in_file:
        return pickle.load(in_file)

(int_texts, int_summaries), (sorted_texts, sorted_summaries), (vocab_to_int, int_to_vocab), word_embedding_matrix = load_preprocess()



# In[2]:


# model的输入数据 placeholder
def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # new
    weights = tf.placeholder(tf.float32, [None, None], name='weights')

    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')
    
    return input_data, targets, weights, lr, keep_prob, summary_length, max_summary_length, text_length


# In[3]:


# 去掉batch最后一个单词，连接<GO>到头部
def process_decoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data,[0,0],[batch_size,-1],[1,1])
    dec_input = tf.concat([tf.fill([batch_size,1],vocab_to_int['<GO>']),ending],1)
    return dec_input


# In[4]:


# RNN: LSTM + dropout
# encoder 层
def encoding_layer(rnn_size,sequence_length, num_layers, rnn_inputs, keep_prob):
    # RNN cell
#    def make_cell(rnn_size):
#        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
#                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
#        return enc_cell
#
#    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
#    
#    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, rnn_inputs, sequence_length=sequence_length, dtype=tf.float32)
#    
#    return enc_output, enc_state

    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            # 前向
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=111))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,input_keep_prob=keep_prob)
            # 反向
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)
            # 双向rnn
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                   cell_bw,
                                                                   rnn_inputs,
                                                                   sequence_length,
                                                                   dtype=tf.float32)
            # ????
            enc_output = tf.concat(enc_output,2)
            # 当前层的输出作为下一层的输入
            rnn_inputs = enc_output
    
    # 返回最后一层
    return enc_output, enc_state


# In[8]:


# train\inference 阶段
def training_decoding_layer(dec_embed_input, summary_length, dec_cell, output_layer, vocab_size, max_summary_length, batch_size):
    # sequence_length 当前batch中每个序列的长度
    # input 对应embedded_input 其shape为[batch_size, sequence_length, embedding_size]
    trainint_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                       sequence_length=summary_length,
                                                       time_major=False)
    # cell 为多层LSTM实例
    # 与encoder final_state的state同类型，这里直接将encoder final_state作为输入
    # 
    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                   helper=trainint_helper,
                                                   initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                   output_layer=output_layer)
    
    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                       output_time_major=False,
                                                       impute_finished=True,
                                                       maximum_iterations=max_summary_length)
    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, output_layer, max_summary_length, batch_size):
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32),[batch_size],name='start_tokens')
    # endtoken 序列终止时的token_id
    # start_tokens 每个序列开始时的token_id
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                               start_tokens,
                                                               end_token)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       inference_helper,
                                                       dec_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
                                                       output_layer)
    inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                        output_time_major=False,
                                                        impute_finished=True,
                                                        maximum_iterations=max_summary_length)
    return inference_logits


# In[9]:


# lstm加上dropout
def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

# 加入Dense
from tensorflow.python.layers.core import Dense

# decoder 层
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                  max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layer):
    dec_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size,keep_prob) for _ in range(num_layer)])
    
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
    # attention
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                    enc_output,
                                                    text_length,
                                                    normalize=False,
                                                    name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell, attn_mech, rnn_size)
    
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  summary_length,
                                                  dec_cell,
                                                  output_layer,
                                                  vocab_size,
                                                  max_summary_length,
                                                  batch_size)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                   vocab_to_int['<GO>'],
                                                   vocab_to_int['<EOS>'],
                                                   dec_cell,
                                                   output_layer,
                                                   max_summary_length,
                                                   batch_size)
    return training_logits, inference_logits
    
    


# In[10]:


# seq2seq模型------合并encoder与decoder
def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                 vocab_size, rnn_size, num_layer, vocab_to_int, batch_size):
    #cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(num_layers)])
    #print(type(input_data),type(target_data))
#    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
#    dec_input = process_decoding_input(target_data, vocab_to_int, batch_size)
#    result, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
#                                                    encoder_inputs=input_data,
#                                                    decoder_inputs=dec_input,
#                                                    cell=cell,
#                                                    num_encoder_symbols=vocab_size,
#                                                    num_decoder_symbols=vocab_size,
#                                                    embedding_size=64,
#                                                    output_projection=None,
#                                                    feed_previous=False)
    

    embeddings = word_embedding_matrix
    # encoder 输入数据
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    # encoder
    enc_output, enc_state = encoding_layer(rnn_size,
                                          text_length,
                                          num_layer,
                                          enc_embed_input,
                                          keep_prob)
    # decoder 输入数据
    dec_input = process_decoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    # decoder
    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                      embeddings,
                                                      enc_output,
                                                      enc_state,
                                                      vocab_size,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      rnn_size,
                                                      vocab_to_int,
                                                      keep_prob,
                                                      batch_size,
                                                      num_layer)
    
    return training_logits, inference_logits
    """
    return result
    """
    
    
    

# 参数
# 训练轮数
epochs = 100
# 每次训练的数据量
batch_size = 16
# 每轮训练的次数
epoch_size = 32
# RNN的大小
rnn_size = 128
# 网络层的数量
num_layers = 2
# 学习率
learning_rate = 0.001
# dropout 参数
keep_probability = 0.95

# In[11]:


# 补充序列尾部， 使所有序列长度为最长序列的长度， 补充字符为'<PAD>'
def pad_batch(article_batch):
    max_article = max([len(a) for a in article_batch])
    return [a+[vocab_to_int['<PAD>']]*(max_article-len(a)) for a in article_batch]

# 随机得到batch_size数量大小的数据作为输入训练
# 可以考虑使用蓄水池方法
# 这里只是顺序读取
def get_batches(summaries, texts, batch_size):
    for batch_i in range(0, epoch_size):
        summaries_batch = []
        texts_batch = []
        for i in range(batch_size):
            start_i = np.random.randint(len(texts)-1)
            summaries_batch.append(summaries[start_i])
            texts_batch.append(texts[start_i])
        pad_summaries_batch = np.array(pad_batch(summaries_batch))
        pad_texts_batch = np.array(pad_batch(texts_batch))
        
        # 需要每个数据长度作为参数
        pad_summaries_lengths = []
        
        weights_batch = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
            weights = np.ones(len(summary),dtype=np.float32)
            for i in range(len(summary)):
                if summary[i]==vocab_to_int['<PAD>']:
                    weights[i]=0.0
            weights_batch.append(weights)
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths, weights_batch


# 建立 graph, 也即整个网络
train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, weights, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                     targets,
                                                     keep_prob,
                                                     text_length,
                                                     summary_length,
                                                     max_summary_length,
                                                     len(vocab_to_int)+1,
                                                     rnn_size,
                                                     num_layers,
                                                     vocab_to_int,
                                                     batch_size)
    # 创建tensors 
    training_logits = tf.identity(training_logits[0].rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')
    #targets = tf.nn.embedding_lookup(word_embedding_matrix, targets)
    #training_logits = tf.nn.embedding_lookup(word_embedding_matrix, inference_logits)
#    
#    masks = tf.sequence_mask(summary_length,max_summary_length, dtype=tf.float32, name='masks')
#    result = seq2seq_model(tf.reverse(input_data, [-1]),
#                                                     targets,
#                                                     keep_prob,
#                                                     text_length,
#                                                     summary_length,
#                                                     max_summary_length,
#                                                     len(vocab_to_int)+1,
#                                                     rnn_size,
#                                                     num_layers,
#                                                     vocab_to_int,
#                                                     batch_size)
#    logits=tf.stack(result,axis=0)
    
#    w_t = tf.get_variable("proj_w", [len(vocab_to_int)+1, rnn_size], dtype=tf.float32)
#    w = tf.transpose(w_t)
#    b = tf.get_variable("proj_b", [len(vocab_to_int)+1], dtype=tf.float32)
#    output_projection = (w, b)
#    
#    def sampled_loss(labels, logits):
#        labels = tf.reshape(labels, [-1, 1])
#        # We need to compute the sampled_softmax_loss using 32bit floats to
#        # avoid numerical instabilities.
#        local_w_t = tf.cast(w_t, tf.float32)
#        local_b = tf.cast(b, tf.float32)
#        local_inputs = tf.cast(logits, tf.float32)
#        return tf.cast(
#                tf.nn.sampled_softmax_loss(
#                        weights=local_w_t,
#                        biases=local_b,
#                        labels=labels,
#                        inputs=local_inputs,
#                        num_sampled=4096,
#                        num_classes=len(vocab_to_int)),
#                        dtype=tf.float32)
#    softmax_loss_function = sampled_loss
    with tf.name_scope("optimization"):
        # Loss函数  损失函数
        # 代价函数
        
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, weights)#,softmax_loss_function=softmax_loss_function)
        # cost = tf.nn.softmax_cross_entropy_with_logits(labels=training_logits, logits=targets)
        # 优化函数---adam
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # 计算梯度下降
        gradients = optimizer.compute_gradients(cost)
        # 范围缩放，将梯度值缩小在（-5，5）
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.),var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


graph_location = "./graph"
print(graph_location)

train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(train_graph)

# 开始train
#训练参数设置
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 1
stop_early = 0
#如果loss 3次还没有减少，就stop
stop = 10
per_epoch = 30
update_check = 3#(len(sorted_texts)//batch_size//per_epoch)-1

update_loss = 0
batch_loss = 0
summary_update_loss = []
# 缓存路径
checkpoint = "./best_model.ckpt"
# 训练+++ early stop


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths, weights_batch)\
        in enumerate(get_batches(sorted_summaries, sorted_texts, batch_size)):
            # print(summaries_batch)
            # 记录时间
            start_time = time.time()
            # 得到loss
            _, loss = sess.run(
                [train_op, cost],
                {input_data:texts_batch,
                targets:summaries_batch,
                weights:weights_batch, 
                lr:learning_rate,
                 summary_length:summaries_lengths,
                 text_length:texts_lengths,
                 keep_prob:keep_probability}
                )
            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time
            # 输出
            #if batch_i % display_step == 0 and batch_i > 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              epoch_size,#len(sorted_texts) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
            batch_loss = 0
            # 每一轮的check
            if (batch_i+1) % update_check == 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)
                
                # 如果得到的loss是最小值 则保存模型
                if update_loss <= min(summary_update_loss):
                    stop_early = 0
                    saver = tf.train.Saver() 
                    print("---模型保存---")
                    saver.save(sess, checkpoint)
                # 如果不是则stop_early+1, 当到达stop的阈值时跳出循环
                else:
                    stop_early += 1
                    if stop_early == stop:
                        print("Stopping Training...")
                        break
                update_loss = 0
            
                    
        # 减小学习率，但最小不小于min_learning_rate
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training...")
            break
