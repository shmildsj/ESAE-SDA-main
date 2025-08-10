#!/usr/bin/env python
# coding: utf-8

# # DATA

# In[1]:
from torch import Tensor
from typing import Optional
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
import math
import torch
import os
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from libauc.optimizers import PESG
from libauc.losses import AUCMLoss
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from arg_parser import parse_args
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, GraphConv, APPNP
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
from BGA import BGA
from egnn_clean import EGNN
from BGA_layer import BGALayer
from torch_geometric.utils import subgraph


# In[2]:
args = parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)


def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)


def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)

# functional similarity
def S_fun1(DDsim, T0, T1):
    DDsim = np.array(DDsim)
    T0_T1 = []
    if len(T0) != 0 and len(T1) != 0:
        for ti in T0:
            m_ax = []
            for tj in T1:
                m_ax.append(DDsim[ti][tj])
            T0_T1.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T0_T1.append(0)
    T1_T0 = []
    if len(T0) != 0 and len(T1) != 0:
        for tj in T1:
            m_ax = []
            for ti in T0:
                m_ax.append(DDsim[tj][ti])
            T1_T0.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T1_T0.append(0)
    return T0_T1, T1_T0

# 计算Fs
def FS_fun1(T0_T1, T1_T0, T0, T1):
    a = len(T1)
    b = len(T0)
    S1 = sum(T0_T1)
    S2 = sum(T1_T0)
    FS = []
    if a != 0 and b != 0:
        Fsim = (S1+S2)/(a+b)
        FS.append(Fsim)
    if a == 0 or b == 0:
        FS.append(0)
    return FS


# In[3]:


# Gaussian interaction profile kernel similarity
def r_func(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    EUC_MD = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    EUC_MD = EUC_MD**2
    EUC_DL = EUC_DL**2
    sum_EUC_MD = np.sum(EUC_MD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1 / ((1 / m) * sum_EUC_MD)
    rt = 1 / ((1 / n) * sum_EUC_DL)
    return rl, rt


def Gau_sim(MD, rl, rt):
    MD = np.mat(MD)
    DL = MD.T
    m = MD.shape[0]
    n = MD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = MD[i] - MD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1**2
            b1 = math.exp(-rl * b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2**2
            b2 = math.exp(-rt * b2)
            d.append(b2)
    GMM = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)
    return GMM, GDD


# In[4]:


#cosine similarity
def cos_sim(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm!=0 and b_norm!=0:
                cos_ms = np.dot(a,b)/(a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)
            
    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm!=0 and b1_norm!=0:
                cos_ds = np.dot(a1,b1)/(a1_norm * b1_norm)
                cos_DS1.append(cos_ds)  
            else:
                cos_DS1.append(0)
        
    cos_MS1 = np.array(cos_MS1).reshape(m, m)
    cos_DS1 = np.array(cos_DS1).reshape(n, n)
    return cos_MS1,cos_DS1


# In[5]:


#sigmoid function kernel similarity
def sig_kr(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    sig_MS1 = []
    sig_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i,:]
            b = MD[j,:]
            z = (1/m)*(np.dot(a,b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)
            
    for i in range(n):
        for j in range(n):
            a1 = MD[:,i]
            b1 = MD[:,j]
            z1 = (1/n)*(np.dot(a1,b1))
            sig_ds = math.tanh(z1)
            sig_DS1.append(sig_ds)            
        
    sig_MS1 = np.array(sig_MS1).reshape(m, m)
    sig_DS1 = np.array(sig_DS1).reshape(n, n)
    return sig_MS1,sig_DS1    


# In[6]:


MD = pd.read_csv("D:/pycharm/d-vgae-main/data/RNADisease/association_matrix.csv",index_col=0)
MD


# In[7]:


DS = pd.read_csv("D:/pycharm/d-vgae-main/data/RNADisease/DD_matrix.csv",index_col=0)
DS


# In[8]:


m = MD.shape[0]
T = []
for i in range(m):
    T.append(np.where(MD.iloc[i] == 1))
Fs = []
for ti in range(m):
    for tj in range(m):
        Ti_Tj, Tj_Ti = S_fun1(DS, T[ti][0], T[tj][0])
        FS_i_j = FS_fun1(Ti_Tj, Tj_Ti, T[ti][0], T[tj][0])
        Fs.append(FS_i_j)
Fs = np.array(Fs).reshape(MD.shape[0], MD.shape[0])
Fs=pd.DataFrame(Fs)
for index,rows in Fs.iterrows():
    for col, rows in Fs.items():
        if index==col:
            Fs.loc[index,col]=1
Fs


# In[9]:


rm, rt = r_func(MD)
GaM, GaD = Gau_sim(MD, rm, rt)


# In[10]:


GaM = pd.DataFrame(GaM)
GaM


# In[11]:


GaD = pd.DataFrame(GaD)
GaD


# In[12]:


MD_c = MD.copy()
MD_c.columns=range(0,MD.shape[1])
MD_c.index=range(0,MD.shape[0])
MD_c=np.array(MD_c)
MD_c


# In[13]:


cos_MS, cos_DS=cos_sim(MD_c)


# In[14]:


cos_MS=pd.DataFrame(cos_MS)
cos_MS


# In[15]:


cos_DS=pd.DataFrame(cos_DS)
cos_DS


# In[16]:


sig_MS,sig_DS = sig_kr(MD_c)


# In[17]:


sig_MS = pd.DataFrame(sig_MS)
sig_MS


# In[18]:


sig_DS = pd.DataFrame(sig_DS)
sig_DS


# # Multi-source features fusion

# In[19]:

#
MM = (Fs+GaM+cos_MS+sig_MS)/4
MM


# In[20]:


DS_t=DS.copy()
DS_t.columns=np.arange(DS.shape[0])
DS_t.index=np.arange(DS.shape[0])
DD = (DS_t+GaD+cos_DS+sig_DS)/4

DD


# In[21]:


MM.max().max()


# In[22]:


DD.max().max()


# # feature

# In[23]:


MM = np.array(MM)
DD = np.array(DD)


# In[24]:


EIG = []# feature matrix of total sample
for i in range(DD.shape[0]):
    for j in range(MM.shape[0]):
        eig = np.hstack((DD[i],MM[j]))#feature vector length :DD.shape[0]+MM.shape[0]
        EIG.append(eig)
#  EIG[i][j] The eigenvector of the sample (d, m), and the corresponding label matrix is DM.
EIG_t = np.array(EIG).reshape(DD.shape[0],MM.shape[0],DD.shape[0]+MM.shape[0])
EIG_t


# In[25]:


#Define random number seed
def setup_seed(seed):
    torch.manual_seed(seed)# 
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)#
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False


# In[26]:


DM_lable = MD_c.T
DM_lable


# In[27]:


# 
DM_lable = np.array(DM_lable)
lable = DM_lable.reshape(DM_lable.size).copy()  # 
lable


# In[28]:


PS_sub = np.where(lable==1)[0]#
PS_sub


# # Selecting negative samples by k_means clustering

# In[29]:


PS_num = len(np.where(lable==1)[0])# Positive sample number
NS1 = np.where(lable==0)[0]# 
# NS_sub = np.array(random.sample(list(NS1),PS_num))#
# NS_sub
NS1


# In[30]:


NS1.shape


# In[31]:


#Take the labels and feature vectors corresponding to all negative samples.
N_SAMPLE_lable = []#Label  corresponding to the sample
N_CHA=[]#The eigenvector matrix of the sample
for i in NS1:
    N_SAMPLE_lable.append(lable[i])
    N_CHA.append(EIG[i])
N_CHA = np.array(N_CHA)
print("N_SAMPLE_lable",N_SAMPLE_lable)
print("N_CHA",N_CHA)


# In[32]:


np.array(N_CHA).shape


# In[33]:


kmeans=KMeans(n_clusters=23, random_state=36).fit(N_CHA)
kmeans


# In[34]:


center=kmeans.cluster_centers_
center


# In[35]:


labels=kmeans.labels_
labels


# In[36]:


center.shape


# In[37]:


labels.shape


# In[38]:


# center_x=[]
# center_y=[]
# for j in range(len(center)):
#     center_x.append(center[j][0])
#     center_y.append(center[j][1])


# In[39]:


# setup_seed(36)


# In[40]:


type1=[]
type2=[]
type3=[]
type4=[]
type5=[]
type6=[]
type7=[]
type8=[]
type9=[]
type10=[]
type11=[]
type12=[]
type13=[]
type14=[]
type15=[]
type16=[]
type17=[]
type18=[]
type19=[]
type20=[]
type21=[]
type22=[]
type23=[]
for i in range(len(labels)):
    if labels[i]==0:
        type1.append(NS1[i])
    if labels[i]==1:
        type2.append(NS1[i])
    if labels[i]==2:
        type3.append(NS1[i])
    if labels[i]==3:
        type4.append(NS1[i])
    if labels[i]==4:
        type5.append(NS1[i])
    if labels[i]==5:
        type6.append(NS1[i])
    if labels[i]==6:
        type7.append(NS1[i])       
    if labels[i]==7:
        type8.append(NS1[i])        
    if labels[i]==8:
        type9.append(NS1[i])      
    if labels[i]==9:
        type10.append(NS1[i])        
    if labels[i]==10:
        type11.append(NS1[i])        
    if labels[i]==11:
        type12.append(NS1[i])       
    if labels[i]==12:
        type13.append(NS1[i])      
    if labels[i]==13:
        type14.append(NS1[i])        
    if labels[i]==14:
        type15.append(NS1[i])        
    if labels[i]==15:
        type16.append(NS1[i])       
    if labels[i]==16:
        type17.append(NS1[i])       
    if labels[i]==17:
        type18.append(NS1[i])        
    if labels[i]==18:
        type19.append(NS1[i])      
    if labels[i]==19:
        type20.append(NS1[i])     
    if labels[i]==20:
        type21.append(NS1[i])        
    if labels[i]==21:
        type22.append(NS1[i])      
    if labels[i]==22:
        type23.append(NS1[i])


# In[41]:


type23


# In[42]:


len(type23)


# In[43]:


len(type2)


# In[44]:


type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23] 
type


# In[45]:


setup_seed(36)


# In[46]:


# Select a negative sample equal to the positive sample.
type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23]                                       
mtype=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]                                                                     
for k in range(23):
    mtype[k]=random.sample(type[k],196) 
mtype


# In[47]:


len(mtype)


# In[48]:


# Negative sample subscript
NS_sub=[]
for i in range(len(lable)):
    for z2 in range(23):
        if i in mtype[z2]: 
            NS_sub.append(i)
NS_sub


# In[49]:


len(NS_sub)


# # Determine the total sample

# In[50]:


#
SAMPLE_sub = np.hstack((PS_sub,NS_sub))#
SAMPLE_sub


# In[51]:


#The label and feature vector corresponding to this sample
SAMPLE_lable = []#Labels 0 and 1 corresponding to samples
CHA=[]#The eigenvector matrix of the sample
for i in SAMPLE_sub:
    SAMPLE_lable.append(lable[i])
    CHA.append(EIG[i])
CHA = np.array(CHA)
print("SAMPLE_lable",SAMPLE_lable)
print("CHA",CHA)


# # DSAE

# In[5]:


# Define some global constants
BETA = math.pow(10,-7)
N_INP = CHA.shape[1]#Input dimension
N_HIDDEN = 1152#Hide layer dimension
N_EPOCHS = 150# Epoch times
use_sparse = True #Sparse or not


# In[6]:


#DSAE
class DSAE(nn.Module):
        def __init__(self):
            super(DSAE,self).__init__()
            #encoder
            self.encoder = nn.Sequential(
                nn.Linear(in_features=N_INP,out_features=N_HIDDEN),
                nn.Sigmoid(),
                nn.Linear(N_HIDDEN, int(N_HIDDEN/2)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/2), int(N_HIDDEN/4)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/4), int(N_HIDDEN/8)),
                nn.Sigmoid()
            )
        # decoder
            self.decoder = nn.Sequential(
                nn.Linear(int(N_HIDDEN/8), int(N_HIDDEN/4)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/4), int(N_HIDDEN/2)),
                nn.Sigmoid(),
                nn.Linear(int(N_HIDDEN/2), N_HIDDEN),
                nn.Sigmoid(),
                nn.Linear(N_HIDDEN, N_INP),
                nn.Sigmoid(),      
            )
            
        def forward(self, x):
            x = x.view(x.size(0),-1)
            encode = self.encoder(x)
            decode = self.decoder(encode)
            return encode, decode
net = DSAE()
net


# In[54]:


# Transform feature vectors into tensors
# T = np.array(T)
CHA_t = CHA.copy()
CHA_t = torch.from_numpy(CHA_t)
# CHA_t = torch.as_tensor(CHA_t)
# CHA = CHA.float()
CHA_t = CHA_t.float()#将数据转化float32
CHA_t


# In[55]:


CHA_t.shape


# In[56]:


# Define and save network functions.
def save(net, path):
    torch.save(net.state_dict(), path)
#     torch.save(net, path)


# In[57]:


def KL_loss(encoded_matrix,beta):
#     encoded_matrix = torch.softmax(encoded_matrix, axis=0)  #
    p_hat = torch.mean(encoded_matrix, axis=0)
#     print('p_hat=', p_hat)
    p = 0.05 #目标值
    KLD = p*(torch.log(p/p_hat))+(1-p)*(torch.log((1-p)/(1-p_hat)))
#     print('KLD=', KLD)
    return beta*torch.sum(KLD)

def mask_edge(edge_index, p):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        self.p = p

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges


# Define training network function, network, loss evaluation, optimizer and training set.
def train(net, trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_rate = []
    lr_t = []#存储学习率

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)# Network parameters, learning rate
#     scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
#     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70,90], gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    mask = Mask(p=0.9)
    for epoch in range(N_EPOCHS):  #
        optimizer.zero_grad()  #

            # forward + backward + optimize
#         encoded = net.encode(trainloader)
        encoded, decoded = net(trainloader)              #
#         print("encoded",encoded)
#         print("decoded",decoded)

       # loss
        inputs1 = trainloader.view(trainloader.size(0),-1)
        loss1 = criterion(decoded,inputs1)
#             print(loss1)

        if use_sparse:#
            #kl_loss = torch.sum(torch.sum(trainloader*(torch.log(trainloader/decoded))+(1-trainloader)*(torch.log((1-trainloader)/(1-decoded)))))
            kl_loss = KL_loss(encoded, BETA)
#             p=trainloader
#             q=decoded
#             kl_loss = F.kl_div(q.log(),p,reduction="sum")+F.kl_div((1-q).log(),1-p,reduction="sum")
            loss = loss1 + kl_loss
#             print(kl_loss)
#             print(loss1)
        else:
            loss = loss1
        loss.backward()                   #
        optimizer.step()                #

            #
        scheduler.step(loss)
        print("[%d] loss: %.5f" % (epoch + 1, loss))
        lr = optimizer.param_groups[0]['lr']
        lr_t.append(lr)
        print("epoch={}, lr={}".format(epoch + 1, lr_t[-1]))
        loss_t = loss.clone()
        loss_rate.append(loss_t.cpu().detach().numpy())
    x = list(range(len(lr_t)))
    plt.subplots(figsize=(10,6))
    plt.subplots_adjust(left=None,wspace=0.5)
    plt.subplot(121)
    plt.title('(a)',x=-0.2,y=1)
#     loss_rate = ['{:.5f}'.format(i) for i in loss_rate]
    plt.plot(x,loss_rate,label='loss change curve',color="#f59164")
    plt.ylabel('loss changes')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('(b)',x=-0.2,y=1)
    plt.plot(x, lr_t, label = 'lr curve',color="#f59164")
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend()
    #plt.savefig("E:/loss_lr.png",dpi=300)
    plt.show()
    print('Finished Training')


# In[58]:
def apply_mask(data, mask_prob=0.1):
    mask = torch.rand(data.shape) < mask_prob  # 生成一个随机掩码，掩码概率为mask_prob
    data_masked = data.clone()  # 复制数据以避免修改原始数据
    data_masked[mask] = 0  # 应用掩码
    return data_masked

# # 假设CHA_t是您的原始输入张量
# CHA_t = apply_mask(CHA_t, mask_prob=0.6)  # 对数据应用80%的掩码

net.train()#Training mode
train(net,CHA_t)


# In[59]:


path = 'D:/pycharm/DSAE_RF-main/DSAE_RF-main/DSAE_RF/Data/DSAE_RF.pth'
save(net, path)


# In[7]:


net.load_state_dict(torch.load( 'D:/pycharm/DSAE_RF-main/DSAE_RF-main/DSAE_RF/Data/DSAE_RF.pth'))


# In[8]:


torch.load('D:/pycharm/DSAE_RF-main/DSAE_RF-main/DSAE_RF/Data/DSAE_RF.pth')



# # predict

# In[62]:


net.eval()    #test

sample_extract=net(CHA_t)
sample_extract


# In[63]:


encoded = sample_extract[0]
decoded = sample_extract[1]

# Reserved features
SAMPLE_feature = encoded.detach().numpy()#
SAMPLE_feature


# In[64]:


SAMPLE_lable = np.array(SAMPLE_lable)# Sample label
SAMPLE_lable

from torch_geometric.data import Data
x_all = torch.tensor(SAMPLE_feature, dtype=torch.float32)
from sklearn.neighbors import kneighbors_graph
import torch_geometric.utils as pyg_utils

A = kneighbors_graph(SAMPLE_feature, n_neighbors=3, mode='connectivity', include_self=False)
edge_index = pyg_utils.dense_to_sparse(torch.tensor(A.toarray(), dtype=torch.float32))[0]
# 构造边的 label：存在为1，不存在为0（这里默认knn图的边都是正例）
edge_labels = torch.ones(edge_index.shape[1], dtype=torch.float32)
# 正样本
positive_edges = edge_index.t().tolist()
positive_labels = [1] * len(positive_edges)

# 随机负样本（不在原始边集中）
import random
num_nodes = x_all.size(0)
negative_edges = set()
while len(negative_edges) < len(positive_edges):
    i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
    if [i, j] not in positive_edges and [j, i] not in positive_edges:
        negative_edges.add((i, j))
negative_edges = list(negative_edges)
negative_labels = [0] * len(negative_edges)

# 合并
all_edges = torch.tensor(positive_edges + negative_edges, dtype=torch.long).t()
all_labels = torch.tensor(positive_labels + negative_labels, dtype=torch.float32).unsqueeze(1)

data = Data(x=x_all, edge_index=edge_index, y=all_labels.unsqueeze(1))


from torch_geometric.nn.aggr import Aggregation

class VariancePreservingAggregation(Aggregation):

    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2
    ) -> Tensor:

        # apply sum aggregation on x
        sum_aggregation = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

        # count the number of neighbours
        shape = [1 for _ in x.shape]
        shape[dim] = x.shape[dim]
        ones = torch.ones(size=shape, dtype=x.dtype, device=x.device)
        counts = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        # simpler variant that consumes more memory:
        # counts = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))


class GraphConv1(GraphConv):
    '''
    Adaptation of torch-geometric's GraphConv layer to variance preserving aggregation.
    '''

    def __init__(self, in_channels: int, out_channels: int, aggr: str, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, **kwargs)

        self.aggr = aggr

        if self.aggr in ['sum', 'add']:
            self.aggr_module = SumAggregation()
        elif self.aggr in ['mean', 'average']:
            self.aggr_module = MeanAggregation()
        elif self.aggr in ['vpa', 'vpp', 'vp']:
            self.aggr_module = VariancePreservingAggregation()
        elif self.aggr == 'max':
            self.aggr_module = MaxAggregation()
        else:
            raise NotImplementedError('Invalid aggregation function.')

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)

class GraphNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, scaling_factor=1.8, aggr="vpa"):
        super(GraphNetEncoder, self).__init__()
        self.conv1 = GraphConv1(in_channels, hidden_channels, aggr=aggr)  # 使用自定义 GraphConv
        self.conv2 = GraphConv1(hidden_channels, hidden_channels, aggr=aggr)  # 也是 GraphConv
        self.scaling_factor = scaling_factor
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x

class GlobalAttn(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super(GlobalAttn, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x



# In[81]:
class Polynormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, global_layers=2,
            in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False):
        super(Polynormer, self).__init__()

        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, hidden_channels)
        self.pred_global = torch.nn.Linear(heads*hidden_channels, hidden_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ## equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1-beta)*self.lns[i](h*x) + beta*x
            x_local = x_local + x

        ## equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

        return x



class GraphNet(torch.nn.Module):
    def __init__(self,num_nodes, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,dropout1=0.5, dropout2=0.1,
                 layers=2, n_head=1,alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(GraphNet, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        self.bga = BGA(num_nodes, num_node_features, hidden_channels, num_classes, layers, n_head,
                       use_patch_attn, dropout1, dropout2)

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        self.polynormer = Polynormer(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes,  # Adjust if output shape differs
            local_layers=2,
            global_layers=2,
            in_dropout=0.15,
            dropout=0.5,
            global_dropout=0.5,
            heads=n_head,
            beta=-1,
            pre_ln=False
        )

        self.global_attn = GlobalAttn(hidden_channels, heads=n_head,num_layers= 2, beta=-1, dropout=0.5)

        self.GTN1 = TransformerConv(num_node_features, hidden_channels)
        self.GTN2 = TransformerConv(hidden_channels, hidden_channels)


        self.egnn = EGNN(in_node_nf=hidden_channels,
                         n_layers=4,
                         hidden_nf=hidden_channels,
                         out_node_nf=hidden_channels,
                         in_edge_nf=1)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)

    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def corrupt(self, x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x_1, edge_index):
        # 调用BGALayer
        patch = edge_index  # 假设边索引用于Patch选择
        x = self.encoder(x_1, edge_index)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)
class GraphNet3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet3, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )


        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)

        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)


class GraphNet4(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet4, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )


        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)

        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)


        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)
        return edge_prediction.view(-1)


class GraphNet5(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet5, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)

        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)


class GraphNet6(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet6, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )


        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)

        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)

# In[65]:

AUC = []
ACC = []
RECALL = []
PREECISION = []
AUPR = []
F1 = []

FPR = []
TPR = []
THR1 = []

THR2= []
PRE = []
REC = []

y_real = []
y_proba = []
num_epochs = args.epochs
print('SAMPLE_feature.shape',SAMPLE_feature.shape)

kf = KFold(n_splits=10, shuffle=True, random_state=36)
# for edge_train_idx, edge_test_idx in kf.split(all_edges.t()):  # 注意是对边做 split
#     edge_index_train = all_edges[:, edge_train_idx]
#     edge_labels_train = all_labels[edge_train_idx]
#
#     edge_index_test = all_edges[:, edge_test_idx]
#     edge_labels_test = all_labels[edge_test_idx]
#
#     # 用 edge_index_train 构建子图，x 使用全图节点特征 x_all
#     train_data = Data(x=x_all, edge_index=edge_index_train, y=edge_labels_train).to(device)
#     test_data = Data(x=x_all, edge_index=edge_index_test, y=edge_labels_test).to(device)
#     model = GraphNet(num_nodes=x_all.shape[0], num_node_features=x_all.shape[1])
#     model.to(device)
for train_index, test_index in kf.split(SAMPLE_sub):
    train_features = SAMPLE_feature[train_index]
    test_features = SAMPLE_feature[test_index]
    train_labels = SAMPLE_lable[train_index]
    test_labels = SAMPLE_lable[test_index]
    # normalize
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    # x_test = x_all[test_index]
    # y_test = SAMPLE_lable[test_index]
    #
    # train_idx = torch.tensor(train_index, dtype=torch.long)
    # test_idx = torch.tensor(test_index, dtype=torch.long)
    #
    # # 获取训练特征和标签
    # x_train = x_all[train_index]
    # # y_train = SAMPLE_lable[train_index]
    #
    # # 获取测试特征和标签
    # x_test = x_all[test_index]
    # # y_test = SAMPLE_lable[test_index]
    #
    # # ✅ 构建训练子图 edge_index（根据训练节点筛边）
    # edge_index_train, edge_mask_train = subgraph(train_idx, edge_index, relabel_nodes=True)
    # src = edge_index_train[0].cpu().numpy()
    # dst = edge_index_train[1].cpu().numpy()
    # MD_train_submatrix = MD[np.ix_(train_idx.numpy(), train_idx.numpy())]
    #
    # edge_labels_train = torch.tensor(MD_train_submatrix[src, dst], dtype=torch.float32).unsqueeze(1).to(device)
    #
    # # ✅ 重新组织成训练图数据结构
    # train_data = Data(
    #     x=torch.tensor(x_train, dtype=torch.float32),
    #     edge_index=edge_index_train,
    #     y=torch.tensor(edge_labels_train, dtype=torch.float32).unsqueeze(1)
    # )
    # edge_index_test, edge_mask_test = subgraph(test_idx, edge_index, relabel_nodes=True)
    #
    # src = edge_index_test[0].cpu().numpy()
    # dst = edge_index_test[1].cpu().numpy()
    # MD_test_submatrix = MD[np.ix_(test_idx.numpy(), test_idx.numpy())]
    # edge_labels_test = torch.tensor(MD_test_submatrix[src, dst], dtype=torch.float32).unsqueeze(1).to(device)
    #
    # test_data = Data(
    #     x=torch.tensor(x_test, dtype=torch.float32),
    #     edge_index=edge_index_test,
    #     y=torch.tensor(edge_labels_test, dtype=torch.float32).unsqueeze(1)
    # )
    # 保持 edge_index 全局一致
    # data1 = Data(x=x_train, edge_index=edge_index, y=y_train)
    # model = GraphNet(num_nodes=x_train.shape[0], num_node_features=x_train.shape[1])
    # model3 = GraphNet3(num_nodes=x_train.shape[0], num_node_features=x_train.shape[1])
    # model4 = GraphNet4(num_nodes=x_train.shape[0], num_node_features=x_train.shape[1])
    # model5 = GraphNet5(num_nodes=x_train.shape[0], num_node_features=x_train.shape[1])
    # model6 = GraphNet6(num_nodes=x_train.shape[0], num_node_features=x_train.shape[1])

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#     margin = 4.0
#     epoch_decay = 0.0046
#     weight_decay = 0.006
#     aucm_optimizer = PESG(model.parameters(),
#                           loss_fn=AUCMLoss(),
#                           lr=args.lr,
#                           momentum=0.4,
#                           margin=margin,
#                           device=device,
#                           epoch_decay=epoch_decay,
#                           weight_decay=weight_decay)
#     model.train()
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         aucm_optimizer.zero_grad()
#
#         out = model(train_data.x, train_data.edge_index)
#         out *= 0.14
#         out_valid_3 = model3(test_data.x, test_data.edge_index)
#         out_valid_3 *= 0.14
#
#         out_valid_4 = model4(test_data.x, test_data.edge_index)
#         out_valid_4 *= 0.14
#
#         out_valid_5 = model5(test_data.x, test_data.edge_index)
#         out_valid_5 *= 0.14
#
#         out_valid_6 = model6(test_data.x, test_data.edge_index)
#         out_valid_6 *= 0.14
#         out_sum += out + out_valid_3 + out_valid_4 + out_valid_5 + out_valid_6
#
#         out_sum /= 0.7 + 0.2

    # out_sum /= 0.28 + 0.2

    # out_valid = out_sum
    # preds_valid = out_valid
    # print(out_valid)
    # y_true_valid = test_data.y.to(device)
#         # preds = out
#         # y_true = train_data.y.to(device)
#         preds = out.view(-1, 1)  # ensure shape
#         y_true = train_data.y.to(device).view(-1, 1)
#
#         preds = preds.to(device)
#         y_true = y_true.to(device)
#
#         num_positive_samples = (y_true == 1).sum()
#         num_negative_samples = (y_true == 0).sum()
#         weight_factor = num_negative_samples.float() / num_positive_samples.float()
#         # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
#         pos_weight = torch.tensor([weight_factor * args.positive_weights], device=device)
#         pos_weight = pos_weight.to(device)
#         bce_loss = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
#         # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
#
#         aucm_module = AUCMLoss()
#         aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
#         total_loss = args.w_celoss * bce_loss + args.w_aucloss * aucm_loss.to(device)
#         total_loss.backward()
#         optimizer.step()
#         aucm_optimizer.step()
#
#         accuracy = compute_accuracy(preds, y_true)
#         roc_auc = compute_auc(preds, y_true)
#         aupr = compute_aupr(preds, y_true)
#
#     # 测试
#     model.eval()
#     with torch.no_grad():
#         logits = model(test_data.x, test_data.edge_index)
#         probs = torch.sigmoid(logits).cpu().numpy()
#         preds = (probs > 0.5).astype(int)
#         tru = test_data.y.cpu().numpy()
#     auc_score = roc_auc_score(tru, probs)
#     acc = accuracy_score(tru, preds)
#     pre = precision_score(tru, preds)
#     rec = recall_score(tru, preds)
#     f1 = f1_score(tru, preds)
#     precision_vals, recall_vals, _ = precision_recall_curve(tru, probs)
#     aupr = auc(recall_vals, precision_vals)
#
#     AUC.append(auc_score)
#     ACC.append(acc)
#     AUPR.append(aupr)
#     RECALL.append(rec)
#     PREECISION.append(pre)
#     F1.append(f1)
#
#     print(
#         f"AUC: {auc_score:.4f}, ACC: {acc:.4f}, AUPR: {aupr:.4f}, RECALL: {rec:.4f}, PRECISION: {pre:.4f}, F1: {f1:.4f}")
#
# # 最终平均结果
# print("\n================ FINAL RESULTS ================")
# print("AUC:", np.mean(AUC))
# print("ACC:", np.mean(ACC))
# print("AUPR:", np.mean(AUPR))
# print("RECALL:", np.mean(RECALL))
# print("PREECISION:", np.mean(PREECISION))
# print("F1:", np.mean(F1))
    # #normalize
    # scaler = StandardScaler()
    # train_features = scaler.fit_transform(train_features)
    # test_features = scaler.transform(test_features)
#     print(SAMPLE_sub[train_index])



    rf = RandomForestClassifier(n_estimators=1100, criterion='entropy', max_depth=23,bootstrap=False,
                                min_samples_leaf=5,min_samples_split=6,
                                max_features='sqrt',random_state = 36).fit(train_features, train_labels)#Enter the training set and the label of the training set.

#     test_score1 = clf1.score(test_features, test_labels)#ACC
    test_predict = rf.predict(test_features)
    pre = rf.predict_proba(test_features)[:, 1]
    tru = test_labels
    pre2 = test_predict

    # auc
    auc = roc_auc_score(tru, pre)
    fpr, tpr, thresholds1 = metrics.roc_curve(tru, pre, pos_label=1)  # The actual value indicated as 1 is a positive sample.
    FPR.append(fpr)
    TPR.append(tpr)
    THR1.append(thresholds1)
    AUC.append(auc)
    print("auc:", auc)
    #ACC
    acc = accuracy_score(tru, pre2)
    ACC.append(acc)
    print("acc:",acc)
    # aupr
    precision, recall, thresholds2 = precision_recall_curve(tru, pre)
    aupr = metrics.auc(recall,precision)#
    AUPR.append(aupr)
    THR2.append(thresholds2)
    PRE.append(precision)
    REC.append(recall)
    y_real.append(test_labels)
    y_proba.append(pre)
    print("aupr:",aupr)
    # recall
    recall1 = metrics.recall_score(tru, pre2,average='macro')#
    RECALL.append(recall1)
    print("recall:",recall1)
	# precision
    precision1 = metrics.precision_score(tru, pre2,average='macro')
    print("precision:",precision1)
    PREECISION.append(precision1)
	#f1_score
    f1 = metrics.f1_score(tru, pre2,average='macro')#F1
    F1.append(f1)
    print("f1:",f1)
print("AUC:", np.average(AUC))
print("ACC:", np.average(ACC))
print("AUPR:", np.average(AUPR))
print("RECALL:", np.average(RECALL))
print("PREECISION:", np.average(PREECISION))
print("F1:", np.average(F1))



