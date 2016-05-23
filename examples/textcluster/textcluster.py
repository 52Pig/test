# -*- coding:utf-8 -*-


# 这里基于电影的文本文件做了一些分析和聚类的工作，主要包括下面这些内容:
# 英文句子的分词和stemming
# 对得到的词做tf-idf编码到向量空间
# 用余弦/欧式距离计算2个文本之间的距离
# 用K-means做聚类
# 用multidimensional scaling对结果数据降维（方便可视化）
# 画出聚类的结果
# 做层级聚类
# 画出层次聚类结果


# 主要内容
# 停用词, stemming, 分词
# Tf-idf与文本相似度计算
# K-means聚类
# Multidimensional scaling降维
# 可视化聚类
# 文本层次聚类
# 首先引入需要的包

import numpy as np
import pandas as pd
import nltk  # 自然语言处理
from bs4 import BeautifulSoup  # 网页解析
import re #正则表达式
import os
import codecs
from sklearn import feature_extraction # 特征抽取

# 停用词, stemming, 分词
#那个，读入一下数据
titles = open('title_list.txt').read().split('\n')
#只看了前100的数据
titles = titles[:100]

links = open('link_list_imdb.txt').read().split('\n')
links = links[:100]

# 读入网页数据并解析
synopses_wiki = open('synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_wiki.append(text)

synopses_wiki = synopses_clean_wiki


genres = open('genres_list.txt').read().split('\n')
genres = genres[:100]

print(str(len(titles)) + ' titles')
print(str(len(links)) + ' links')
print(str(len(synopses_wiki)) + ' synopses')
print(str(len(genres)) + ' genres')

##########################################

synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    # 格式化
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

######################################
synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)

# 为每一个词编码下标
ranks = []

for i in range(0, len(titles)):
    ranks.append(i)

#####################################
# 这个部分会用NLTK做一些自然语言处理的工作，包括去掉英语中 "a", "the", "in"这种没有太重要含义的“停用词”。

# 加载停用词表
stopwords = nltk.corpus.stopwords.words('english')
# 做stemming的操作，实际就是把长得很像的英文单词关联在一起
# l调用现成的nltk SnowballStemmer库
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# 定义2个不同的函数:
# tokenize_and_stem: 同时完成分词格式化和stemming工作
# tokenize_only: 只分词格式化
def tokenize_and_stem(text):
    # 先拿到一个tokenized过后的词表
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 去掉无字母的词(比如数字，标点)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # 先拿到一个tokenized过后的词表
    filtered_tokens = []
    # 去掉无字母的词(比如数字，标点)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# 分别使用这2个函数处理数据，拿到2组不同的次序列结果
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

# %time
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

# print(tfidf_matrix.shape)
# 用pandas建一个可查询的stemm词表，这样run就可以和ran，running这样的词关联上了
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
####################################
# Tf-idf与文本相似度计算
# 查看图片293-2402-1-PB     刚才拿到的都是一个个的词，咱们现在得把它编码成特征向量了，为了简单，咱们直接调用sikit-kearn里面的TfidfVectorizer来完成这个事情
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

############################
# K-means聚类
# 现在的文本(每部电影)都特征向量编码完了，就可以跑聚类了。咱们先试试K-means吧
# 别忘了，K-means聚类需要指定聚类编号，咱们指定为5好了，你们可以试试别的聚类数。
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

# %time
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

import pandas as pd

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }

frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

frame['cluster'].value_counts()

grouped = frame['rank'].groupby(frame['cluster'])

grouped.mean()

from __future__ import print_function

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()

# Top terms per cluster:
#
# Cluster 0 words: family, home, mother, war, house, dies,
#
# Cluster 0 titles: Schindler's List, One Flew Over the Cuckoo's Nest, Gone with the Wind, The Wizard of Oz, Titanic, Forrest Gump, E.T. the Extra-Terrestrial, The Silence of the Lambs, Gandhi, A Streetcar Named Desire, The Best Years of Our Lives, My Fair Lady, Ben-Hur, Doctor Zhivago, The Pianist, The Exorcist, Out of Africa, Good Will Hunting, Terms of Endearment, Giant, The Grapes of Wrath, Close Encounters of the Third Kind, The Graduate, Stagecoach, Wuthering Heights,
#
# Cluster 1 words: police, car, killed, murders, driving, house,
#
# Cluster 1 titles: Casablanca, Psycho, Sunset Blvd., Vertigo, Chinatown, Amadeus, High Noon, The French Connection, Fargo, Pulp Fiction, The Maltese Falcon, A Clockwork Orange, Double Indemnity, Rebel Without a Cause, The Third Man, North by Northwest,
#
# Cluster 2 words: father, new, york, new, brothers, apartments,
#
# Cluster 2 titles: The Godfather, Raging Bull, Citizen Kane, The Godfather: Part II, On the Waterfront, 12 Angry Men, Rocky, To Kill a Mockingbird, Braveheart, The Good, the Bad and the Ugly, The Apartment, Goodfellas, City Lights, It Happened One Night, Midnight Cowboy, Mr. Smith Goes to Washington, Rain Man, Annie Hall, Network, Taxi Driver, Rear Window,
#
# Cluster 3 words: george, dance, singing, john, love, perform,
#
# Cluster 3 titles: West Side Story, Singin' in the Rain, It's a Wonderful Life, Some Like It Hot, The Philadelphia Story, An American in Paris, The King's Speech, A Place in the Sun, Tootsie, Nashville, American Graffiti, Yankee Doodle Dandy,
#
# Cluster 4 words: killed, soldiers, captain, men, army, command,
#
# Cluster 4 titles: The Shawshank Redemption, Lawrence of Arabia, The Sound of Music, Star Wars, 2001: A Space Odyssey, The Bridge on the River Kwai, Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb, Apocalypse Now, The Lord of the Rings: The Return of the King, Gladiator, From Here to Eternity, Saving Private Ryan, Unforgiven, Raiders of the Lost Ark, Patton, Jaws, Butch Cassidy and the Sundance Kid, The Treasure of the Sierra Madre, Platoon, Dances with Wolves, The Deer Hunter, All Quiet on the Western Front, Shane, The Green Mile, The African Queen, Mutiny on the Bounty,
frame['Rank'] = frame['rank'] + 1
frame['Title'] = frame['title']

# 表格转成HTML
print(frame[['Rank', 'Title']].loc[frame['cluster'] == 1].to_html(index=False))
############################################
# Multidimensional scaling降维
import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# 降维的原因其实是，我们想可视化一下，但是维度太高的数据没法可视化
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

############################
# 可视化聚类
# 把每个类别的颜色设定好
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# 每个类别的主题词
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

# %matplotlib inline
# pandas格式化结果数据
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# 按cluster分组
groups = df.groupby('label')


# 设定绘图参数
fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  #show legend with only 1 point

#加上x，y标签
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)



plt.show() # 绘图展示

#如果要存储图片可以把下面的#去掉
#plt.savefig('clusters_small_noaxes.png', dpi=200)

plt.close()
# 肉眼看还凑合是吧，聚类主题相关的，很多都在附近。那咱们再用层次聚类试试好了。
#########################
# 文本层次聚类
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) # 定义linkage_matrix为ward型预算聚类

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params( \
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout()  # show plot with tight layout

# 如果要保存图片，把下面的#去掉
# plt.savefig('ward_clusters.png', dpi=200)


plt.close()






