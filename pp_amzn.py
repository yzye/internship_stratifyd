
# coding: utf-8

# In[2]:


import re
import gzip
import random
import numpy as np
import pickle


# In[3]:


pos_path = "/tera/amzn/bal_text_pos.csv"


# In[4]:


neg_path = "/tera/amzn/bal_text_neg.csv"


# In[5]:


def parse(path):
    g = open(path, 'r')
    for l in g:
        yield l


# In[6]:


def get_reviews(path):
    
    reviews = []
    for review in parse(path):
        reviews.append(review)
    
    return reviews


# In[9]:


reviews_pos = get_reviews(pos_path)
reviews_neg = get_reviews(neg_path)


# In[10]:


def get_labels(reviews_pos,reviews_neg):
    
    labels = np.array([1]*len(reviews_pos)+[0]*len(reviews_neg))
    labels = labels.reshape([-1,1])
    labels = np.hstack((labels,1 - labels))
    
    return labels


# In[11]:


reviews = reviews_pos + reviews_neg
labels = get_labels(reviews_pos,reviews_neg)


# In[12]:


reviews[:10]


# In[13]:


labels[:10]


# In[14]:


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),\.!?]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"e\.g\.,", " ", string)
    string = re.sub(r"a\.k\.a\.", " ", string)
    string = re.sub(r"i\.e\.,", " ", string)
    string = re.sub(r"i\.e\.", " ", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"\'", "", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    #string = re.sub(r",", " , ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", "", string)
    #string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    #string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", "", string)
    #string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", "", string)
    #string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\.", "", string)
    #string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"u\.s\.", " us ", string)
    return string.strip().lower()


# In[15]:


sent=[]
vocab = {}


# In[16]:


for i in reviews:
    temp = clean_str(i).split()
    sent.append(temp)
    t = set(temp)
    for word in t:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

len(vocab)
vocab = dict((x, y) for x, y in vocab.iteritems() if y >= 50)
len(vocab)
# In[17]:


ixtoword = {}


# In[18]:


ixtoword[0] = 'END'
ixtoword[1] = 'UNK'
wordtoix = {}
wordtoix['END'] = 0
wordtoix['UNK'] = 1
ix = 2


# In[19]:


for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1


# In[20]:


def convert_word_to_ix(data):
    result = []
    for sent in data:
        temp = []
        for w in sent:
            if w in wordtoix:
                temp.append(wordtoix[w])
            else:
                temp.append(1)
        temp.append(0)
        result.append(temp)
    return result


# In[21]:


train_sent = convert_word_to_ix(sent)


# In[27]:


len(train_sent) == len(train_l)


# In[55]:


train_data = [i for i in zip(train_sent,labels)]


# In[56]:


train_data[:10]


# In[67]:


random.shuffle(train_data)


# In[68]:


train_sent = list(zip(*train_data)[0])


# In[82]:


train_lab = np.array(list(zip(*train_data)[1]))


# In[83]:


train_lab[:10]


# In[84]:


l=len(train_lab)


# In[85]:


s1=int(round(len(train_lab)*0.8))


# In[86]:


s2=int(round(len(train_lab)*0.9))


# In[87]:


train_x = train_sent[:s1]#[:5000]
train_y = train_lab[:s1]#[:5000]
val_x = train_sent[s1:s2]
val_y = train_lab[s1:s2]
test_x = train_sent[s2:]#[:1000]
test_y = train_lab[s2:]#[:1000]


# In[88]:


pickle.dump([train_x, val_x, test_x, train_y, val_y, test_y, wordtoix, ixtoword], open("/tera/yzye/amzn.p", "wb"))#, protocol = 2


# In[77]:


def load_data(path="./amzn.p"):
    with (open(path, "rb")) as openfile:
        while True:
            try:
                x = pickle.load(openfile)
            except EOFError:
                break
        
        train_x, train_y, val_x, val_y, test_x, test_y = x[0], x[3], x[1], x[4], x[2], x[5]
        wordtoix, ixtoword = x[6], x[7]
        #class_name = ['Good','Bad']
        
    return train_x, train_y, val_x, val_y, test_x, test_y, wordtoix, ixtoword


# In[78]:


train_x, train_y, val_x, val_y, test_x, test_y, wordtoix, ixtoword=load_data()


# In[81]:


train_y[:10]

