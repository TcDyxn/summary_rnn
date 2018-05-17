# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:54:17 2018

@author: TcD-PC
"""
import tensorflow as tf
import jieba
import pickle
import re

# cache
cache_path = 'C:/data/data/news_sohusite_xml.full/'
# 文本路径
path_sogou = 'C:/data/data/news_sohusite_xml.full/news_sohusite_xml.html'#'C:/Users/TcD-PC/Desktop/SogouCS.html'
path_sogou = 'C:/Users/TcD-PC/Desktop/SogouCS.WWW08.txt'
# words embeddings路径
path_word_embe = 'C:/Users/TcD-PC/Desktop/ae_output.txt'
path_word_model = 'C:/data/data/word2vec/news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim128.bin'

# 停用词路径
path_stop_words = 'C:/Users/TcD-PC/Desktop/StopwordsCN.txt'

# 参数
# 训练轮数
epochs = 1
# 每次训练的数据量
batch_size = 16
# RNN的大小
rnn_size = 128
# 网络层的数量
num_layers = 4
# 学习率
learning_rate = 0.001
# dropout 参数
keep_probability = 0.7
# 停用词
f=open(path_stop_words,encoding='utf-8')
#加载停用词
stop_words=f.read().strip().split('\n')
f.close()
stop_words=[]

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

#TODO 读取训练集
def load_preprocess():
    with open(cache_path+'summary_data.p', mode='rb') as in_file:
        return pickle.load(in_file)

(int_texts, int_summaries), (sorted_texts, sorted_summaries), (vocab_to_int, int_to_vocab), word_embedding_matrix = load_preprocess()

################################################################
# 预测
# 文本转换为ints
def text_to_seq(text):
    text = ' '.join(jieba.cut(Chinese_word_extraction(text)))
    return [vocab_to_int.get(x,vocab_to_int['<UNK>']) for x in text.split() if x not in stop_words]

# 输入文本
input_sentences = ["荷甲联赛第33轮全面开打,领头羊埃因霍温在客场1-1被实力有限的乌特勒支逼平。与此同时,阿贾克斯以5-2的大比分击败了鹿特丹斯巴达,阿尔克马尔也以3-1战胜了海伦芬,这样3支球队同积72分,阿尔克马尔净胜球最多排在第一,阿贾克斯次之,埃因霍温被挤到了第三。"]
texts = [text_to_seq(article) for article in input_sentences]

# 给出需要得到的摘要的长度
# 这里的list长度要和输入文本的个数一样！***
generate_summary_length_list = [10,10]

# 模型位置
checkpoint = "./best_model.ckpt"

# graph
loaded_graph = tf.Graph()
# 结果
result = []
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #预测 
    for i, text in enumerate(texts):
        # 得到当前文本需要提取的摘要长度
        generate_summary_length = generate_summary_length_list[i]
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                          summary_length: [generate_summary_length], #summary_length: [np.random.randint(5,8)], 
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0] 
        # 移除pad
        pad = vocab_to_int["<PAD>"] 
        print('- Article:\n\r {}'.format(input_sentences[i]))
        print('- Abstract:\n\r {}\n\r\n\r'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
