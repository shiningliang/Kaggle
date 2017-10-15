import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


train = pd.read_csv(r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
test = pd.read_csv(r'E:\OpenSourceDatasetCode\Dataset\Bag of Words Meets Bags of Popcorn\testData.tsv', header=0, delimiter='\t', quoting=3)

label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(train_data[0], '\n')
print(test_data[0])