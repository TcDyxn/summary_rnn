# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:35:51 2018

@author: TcD-PC
"""
import jieba
#解析html
from bs4 import BeautifulSoup
#进度条
from tqdm import tqdm
from tqdm import trange

import numpy as np
import gensim
import gc
import pickle
import re

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

## 中文字符+数字提取
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
#TODO 读取文本，提取摘要和内容
# 摘要和内容
abc=[]
content=[]
with open(path_sogou,'r',encoding='ansi')as a:
    pattern1 = BeautifulSoup(a.read(),'lxml').find_all(["contenttitle","content"])
    for index,item in tqdm(enumerate(pattern1)):
        if(index%2==0):
            abc.append(Chinese_word_extraction(item.string))
        else:
            content.append(Chinese_word_extraction(item.string))
print('摘要长度',len(abc),'内容长度',len(content))
        
# jieba分词后" "连接起来
n = [' '.join([a for a in jieba.cut(x) if a not in stop_words]) for x in abc]
m = [' '.join([a for a in jieba.cut(x) if a not in stop_words]) for x in content]
#del abc
#del content

# 合并为一个text
source_text = (' <END> ').join([a for a in m])
target_text = (' <END> ').join([a for a in n])

# 保存分词结果
save(cache_path+'content.txt',source_text)
save(cache_path+'abstract.txt',target_text)
print("---分词结果已保存---")

#del m
#del n
gc.collect()

arti = source_text.split(' <END> ')
abst = target_text.split(' <END> ')
print('---分词结束---')

##分词结束##
#################################################################################3
#TODO 1
#读取word2vec得到的中文词向量，word embedding
#embeddings_index = {}
#with open(path_word_embe, encoding='utf-8') as f:
#    for line in f:
#        values = line.split(' ')
#        word = values[0]
#        embedding = np.asarray(values[1:], dtype='float32')
#        embeddings_index[word] = embedding

#读取word2vec得到的中文词向量，word embedding
embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        path_word_model,binary=True)

#all_words = list(embeddings_index.vocab.keys())
#TODO 计算每个词出现的次数

words_count = {}

def count_words(count_dict, text):
#    with open(path_stop_words,encoding='utf-8')as f:
#        #加载停用词
#        stop_words=f.read().strip().split('\n')
    for word in text.split():
        #去掉特殊字符
#        if(word in stop_words)or(word.isdigit()):
#            continue
        if word not in count_dict:
            count_dict[word] = 1
        else:
            count_dict[word] += 1
    return count_dict

# 得到每个词的出现次数
words_count = count_words(words_count, source_text)
words_count = count_words(words_count, target_text)

# 词---索引
vocab_to_int = {}
value=0;
# 最少出现的次数
th=2
# 建立 词---索引
for word, count in words_count.items():
    if(count>=th or word in embeddings_index):
        vocab_to_int[word]=value
        value+=1    


# 词---索引
#all_vocab_to_int = {}
#value=0;
#for word in all_words:
#    all_vocab_to_int[word]=value
#    value+=1
#del all_words

# 特殊符号
codes = ["english","number","time","rank","<UNK>", "<PAD>", "<EOS>", "<GO>"]   
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)


#  建立 索引---词
int_to_vocab = {}
for word,v in vocab_to_int.items():
    int_to_vocab[v]=word

print('---词索引建立完毕---')
#总共会用到多少词
print('总共会用到多少词',round(len(vocab_to_int) / len(words_count),4))

# 建立词嵌入
# 这里词向量的维度为64维
word_dim = 64
words_count_all = len(vocab_to_int)
#TODO 创建词嵌入
unk_emb = {}
word_embedding_matrix = np.zeros((words_count_all, word_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    elif word in unk_emb:
        word_embedding_matrix[i] = unk_emb[word]
    else:
        # 如果这个词在vec中不存在，就随机一个新的vec。
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, word_dim))
        unk_emb[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

#del embeddings_index
gc.collect()
print('---词嵌入建立完毕---')
#############################################################################3
#TODO 将文本中的词转换为索引数字
def convert_to_ints(text,word_count,unk_count,eos=False):
    ints = []
    for seq in text:
        seqq = []
        for word in seq.split(' '):
                word_count+=1;
                if(word in vocab_to_int):
                    seqq.append(vocab_to_int[word])
                else:
                    seqq.append(vocab_to_int['<UNK>'])
                    unk_count+=1
        if eos:
            seqq.append(vocab_to_int['<EOS>'])
        ints.append(seqq)
    return ints, word_count, unk_count

word_count = 0
unk_count = 0

#TODO 将中文转换为索引
int_summaries, word_count, unk_count = convert_to_ints(abst, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(arti, word_count, unk_count, eos=True)

# 计算没有标记的词的个数
def unk_counter(sentence):
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
        if word in [vocab_to_int[x] for x in ["图文","组图","numberenglishranknumber","timeenglish","englishnumber"]]:
            unk_count += 2
    return unk_count

# 过滤样本，设置各项门槛
sorted_summaries = []
sorted_texts = []
max_summary_length = 14
max_text_length = 150
min_length = 5
# 过滤不存在的词  摘要中数量2以下的 或者 文章中数量20以下的样本
unk_summary_limit = 1
unk_text_limit = 10

for count, words in (enumerate(int_summaries)):
    if (len(int_summaries[count]) >= min_length and
        len(int_summaries[count]) <= max_summary_length and
        len(int_texts[count]) >= min_length and
        unk_counter(int_summaries[count]) <= unk_summary_limit and
        unk_counter(int_texts[count]) <= unk_text_limit
       ):
        #print('合格')
        sorted_summaries.append(int_summaries[count])
        sorted_texts.append(int_texts[count])
print("训练集大小为：",len(sorted_texts))
print('---词向量处理完毕---')


#保存得到的数据集
with open(cache_path+'summary_data.p', 'wb') as out_file:
    pickle.dump((
        (int_texts, int_summaries),
        (sorted_texts, sorted_summaries),
        (vocab_to_int, int_to_vocab), 
        word_embedding_matrix), out_file)




