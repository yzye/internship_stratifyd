
# coding: utf-8

# In[1]:

import os
import torch
from torch import nn, optim, tensor
import pickle
import numpy as np
import random


# In[2]:


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F


# In[53]:


#CLASSDICT = {'0': 'Sports', '1': 'Society Culture', '2': 'Family Relationships', '3': 'Education Referenc',
#             '4': 'Business Finance', '5':'Health', '6': 'Computers Internet', '7': 'Politics Government',
#             '8': 'Science Mathematics', '9': 'Entertainment Music'}
CLASSDICT = {'0': 'Positive', '1': 'negative'}

# In[54]:

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


# In[55]:


USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


# In[5]:


def pad_to_batch(batch,max_len=80):
    x,y = zip(*batch)
    x_p = []
    for i in range(len(batch)):
        if x[i].size(0) < max_len:
            x_p.append(
            torch.cat([x[i],Variable(torch.LongTensor([0]*(max_len - x[i].size(0))))]))
        else:
            x_p.append(x[i][:max_len])
    return torch.cat(x_p), y


# In[6]:


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


# In[37]:


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
        
    return train_x, train_y, val_x, val_y, test_x, test_y, wordtoix, ixtoword#, class_name


# In[38]:


train_x, train_y, val_x, val_y, test_x, test_y, wordtoix, ixtoword=load_data()


# In[39]:

#train_data=[(Variable(torch.LongTensor((train_x[i]))),(np.argmax(train_y[i]))) for i in range(len(train_y))]
#train_data=[(Variable(torch.LongTensor((train_x[i]))),(train_y[i])) for i in range(len(train_y))]
train_data=[(Variable(torch.LongTensor((train_x[i]))),(train_y[i])) for i in range(10000)]

# In[40]:


#val_data=[(Variable(torch.LongTensor((val_x[i]))),(np.argmax(val_y[i]))) for i in range(len(val_y))]
#val_data=[(Variable(torch.LongTensor((val_x[i]))),(val_y[i])) for i in range(len(val_y))]
val_data=[(Variable(torch.LongTensor((val_x[i]))),(val_y[i])) for i in range(1000)]

# In[41]:


#test_data=[(Variable(torch.LongTensor((test_x[i]))),(np.argmax(test_y[i]))) for i in range(len(test_y))]
#test_data=[(Variable(torch.LongTensor((test_x[i]))),(test_y[i])) for i in range(len(test_y))]
test_data=[(Variable(torch.LongTensor((test_x[i]))),(test_y[i])) for i in range(1000)]

# In[56]:


batch_size = 32
learning_rate = 1e-2
num_epoches = 50
EPOCH = 50
embedding_dim = 300
hidden_dim = 100
ngram = 55
dropout = 0.5
valid_freq = 50
embpath = 'music_glove.p'

class_num = 2
vocab_size=len(wordtoix)


# In[43]:


def test_accuracy(batch_size, test_data, model):
    acc = []
    
    for i, data in enumerate(getBatch(batch_size, test_data), 1):
        
        inputs,targets = pad_to_batch(data)
        model.zero_grad()
        
        #print("inputs:",inputs)
        preds, _ = model(inputs.view(len(targets),-1))
        
        max_index = preds.max(dim = 1)[1]
        
        if len(targets)== batch_size:
            correct = (max_index == torch.LongTensor(targets).cuda()).sum()
            acc.append(float(correct)/batch_size)
        
    return np.mean(acc)


# In[44]:


def test_print(batch_size, test_data, model):
    acc = []
    
    with open("LEAM_STATS.html",'w') as f:
        for i, data in enumerate(getBatch(batch_size, test_data), 1):

            inputs,targets = pad_to_batch(data)        
            model.zero_grad()

            #print("inputs:",inputs)
            preds, beta = model(inputs.view(len(targets),-1))

            max_index = preds.max(dim = 1)[1]

            if len(targets)== batch_size:
                correct = (max_index == torch.LongTensor(targets).cuda()).sum()
                acc.append(float(correct)/batch_size)

            max_index = max_index.cpu().numpy()

            sents = [[ixtoword[ix] for ix in sent if ix not in [0]] for sent in inputs.view(len(targets),-1).numpy()]
            correct = [t==p for t,p in zip(targets, max_index)]

            #for c,t,p,i,j,_ in zip(correct,targets, max_index,sents, beta.squeeze(2).detach().cpu().numpy(),range(2)): 
            for c,t,p,i,j in zip(correct,targets, max_index,sents, beta.squeeze(2).detach().cpu().numpy()): 
                html_write(i,j,c,p,t,f)
        
    return np.mean(acc)


# In[45]:


def html_write(words, values, correct, pred, targets, html_file):
    alphas = (values - values.min()) / (values.max() - values.min())
    html_file.write(u'<p>\n')
    if not correct:
        html_file.write(u'<span style="background-color: rgba(0, 255, 0, 1)">%s</span>\n' % (correct))
    
        html_file.write(u'<span>preds: %s</span>\n' % (CLASSDICT[str(pred)]))
        html_file.write(u'<span>targets: %s</span>\n' % (CLASSDICT[str(targets)]))   
        for word, alpha in zip(words, alphas):
            html_file.write(u'<span style="background-color: rgba(255, 0, 0, %f)">%s</span>\n' % (alpha, word))
    
    html_file.write(u'</p>\n')


# In[46]:


class Classifier(nn.Module):
    
    def __init__(self, vocab_size, class_num, embedding_dim, hidden_dim, ngram, dropout, embpath):
        super(Classifier,self).__init__()
        
        self.class_num = class_num
        
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        try:
            embeddings = np.array(cPickle.load(open(embpath,'rb')),dtype = 'float32')
            self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        except:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.embedding_class = nn.Embedding(class_num, embedding_dim)
        
        self.conv = torch.nn.Conv1d(class_num, class_num, 2*ngram+1,padding=55)
        
        self.layer = nn.Linear(embedding_dim, class_num)
        #self.layer = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, class_num)
        
    def forward(self,inputs):
        emb = self.embedding(inputs.cuda()) # (B, L, e)
        #print("emb1:",emb)
        
        emb_norm = F.normalize(emb, p=2, dim=2, eps=1e-12)
        #print("emb2:",emb)
        #print("embsize:",emb.size()) 
        
        emb_c = self.embedding_class(torch.LongTensor([[i for i in range(self.class_num)] for j in range(inputs.size(0))]).cuda())

        emb_c_norm = F.normalize(emb_c, p=2, dim=2, eps=1e-12)
        
        emb_norm_t = emb_norm.permute(0, 2, 1) # (B, e, L)
        #print("embtsize:",embt.size())
        
        g = torch.bmm(emb_c_norm,emb_norm_t) #(B, C, L)
        #print("gsize:",g.size())
        
        g = F.relu(self.conv(g))
        
        beta = torch.max(g,1)[0].unsqueeze(2) #(B, L, 1)
        
        #print("betasize:",beta.size())
        beta = F.softmax(beta,1) #(B, L, 1)
        
        z = torch.mul(beta,emb) #(B, L, 1)*(B, L, e)
        #print("z1size:",z.size())
        
        z = z.sum(1) #(B, e)
        #print("z2size:",z.size())
        
        out = self.layer(z) #(B, C)
        #print("outsize:",out.size())
        
        #out = self.layer2(F.relu(out,1))
        
        logits = F.log_softmax(out,1) #(B, C)
        
        return logits, beta


# In[47]:


model = Classifier(vocab_size=len(wordtoix), class_num=class_num, 
                   embedding_dim=embedding_dim, hidden_dim=hidden_dim, 
                   ngram=ngram, dropout=dropout, embpath=embpath)


# In[48]:


if USE_CUDA: model=model.cuda()


# In[49]:


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[50]:


max_test_acc = 0.0


# In[ ]:


max_val_acc = 0.0
for epoch in range(EPOCH):
    print("--- epoch:",epoch,"---")
    losses = []
    accuracy = []
    for i, data in enumerate(getBatch(batch_size, train_data), 1):

        inputs,targets = pad_to_batch(data)
        model.zero_grad()

        preds, _ = model(inputs.view(len(targets),-1)) #(B, C)

        loss = loss_function(preds, torch.LongTensor(targets).cuda())
        losses.append(loss.data[0])
        
        max_index = preds.max(dim = 1)[1]
        correct = (max_index == torch.LongTensor(targets).cuda()).sum()
        acc = float(correct)/len(targets)
        accuracy.append(acc)

        loss.backward()
        optimizer.step()
        
        if i % valid_freq == 0:
            print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
            losses = []
            
            val_acc = test_accuracy(batch_size, val_data, model)
            print("val_accuracy : %0.4f" %val_acc)
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_test_acc = test_print(batch_size, test_data, model)
                print("\033[1m"+"max_test_acc : ", max_test_acc,"\033[0m")
                
                torch.save(model.state_dict(),"checkpoints/trained_model.pth")
    
    loss_epoch = np.mean(losses)
    print("loss_epoch : %0.4f" %loss_epoch)
    acc_epoch = np.mean(accuracy)
    print("acc_epoch : %0.4f" %acc_epoch)

#print("max_test_acc: %0.4f" %max_test_acc)    


# In[21]:


print("max_test_acc:",max_test_acc)    


# In[70]:


#trained_model = Classifier(vocab_size=len(wordtoix), class_num=class_num, embedding_dim=300, hidden_dim=100, ngram=55, dropout=0.5)
#trained_model.load_state_dict(torch.load("checkpoints/trained_model.pth"))


# In[69]:


#test_accuracy(batch_size, test_data, trained_model)

