"""import pickle as pkl
#from finch import FINCH

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch
from sklearn.metrics import *
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import matplotlib as mpl
def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
f = open('/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/BadLabels/data_Digits_conv_mnist_0_5.pkl', 'rb')
data = pkl.load(f)
f.close()
all_clean_labels = data[1]
all_noisy_labels = data[2]
all_loss = data[3]
all_indexes = data[5]
all_domains = data[6]
clean_labels = []
noisy_labels = []
losses = torch.zeros(6, max(all_indexes[0])+5) + 100
n_labels = torch.zeros(6,max(all_indexes[0])+5) + 100
c_labels = torch.zeros(6,max(all_indexes[0])+5) + 100

all_domain = torch.zeros(6, max(all_indexes[0])+5) + 100
for i in range(6):
    #print(i)
    index = all_indexes[i]
    losses[i][index] = torch.Tensor(all_loss[i]).squeeze() 
    n_labels[i][index] = torch.Tensor(all_noisy_labels[i])
    c_labels[i][index] = torch.Tensor(all_clean_labels[i])
    all_domain[i][index] = torch.Tensor(all_domains[i]).clone()
#print(losses)

n_labels = n_labels[0]
c_labels = c_labels[0]
test_mask = n_labels == c_labels
target = all_domain[0]
#print(target)
all_indexes = np.arange(0,max(all_indexes[0])+5)
index_no_false = []
index_w_false = []
final_indexes = []
accuracy = []
precision =[]
recall=[]
for d in range(3):
    domain_mask = target == d
    for k in range(8):
        #index_no_false = []
        #index_w_false = []
        #final_indexes = []
        index_no_false_cat = []
        loss_no_false_cat = []
        cat_mask = n_labels[domain_mask] == k 
        select_index = all_indexes[domain_mask][cat_mask] 
        loss_edl = losses[:, domain_mask][:, cat_mask]
        #print(torch.sum(domain_mask))
        loss = np.transpose(loss_edl, (1,0))
        mask = n_labels[domain_mask][cat_mask] == c_labels[domain_mask][cat_mask]
        ind = 0
        if True in mask:
            ind  = 1
        GMM_label = mask.long()
        import hnne
        #print(loss.shape)
        #print(loss.shape)
        try:
            c,_,_,_ = hnne.finch_clustering.FINCH(loss.numpy(),distance='cosine')
            #c, num_clust, req_c = FINCH(loss.numpy(), initial_rank=None, req_clust=2, distance='cosine', verbose=True)
            cat = c[:,0]

            #loss = torch.mean(loss,-1)
            loss_cat = []
            list_cat = []
    
            for i in range(np.max(cat) + 1):
                ma = cat == i
                loss_cat.append(torch.mean(loss[ma]))
                list_cat.append(i)
            list_cat = torch.Tensor(list_cat)
            loss_cat = torch.Tensor(loss_cat)
            classifier = GMM(n_components=2,max_iter=50,init_params='k-means++')
            classifier.fit(loss_cat.unsqueeze(-1).numpy())
            y_train_pred = classifier.predict(loss_cat.unsqueeze(-1).numpy())
            y_ma = y_train_pred == 0
            if torch.mean(loss_cat[y_ma]) < torch.mean(loss_cat[~y_ma]):
                group_indexes = list_cat[y_ma]
                for g_ind in group_indexes:
                    g_ma = cat == g_ind
                    index_no_false.extend(select_index[g_ma])
                group_indexes = list_cat[~y_ma]
                for g_ind in group_indexes:
                    g_ma = cat == g_ind
                    index_w_false.extend(select_index[g_ma])
            else:
                group_indexes = list_cat[~y_ma]
                for g_ind in group_indexes:
                    g_ma = cat == g_ind
                    index_no_false.extend(select_index[g_ma])
                group_indexes = list_cat[y_ma]
                for g_ind in group_indexes:
                    g_ma = cat == g_ind
                    index_w_false.extend(select_index[g_ma])     
        except:
            index_no_false.extend(select_index)     

'''#print(len(test_mask[index_no_false].numpy()))
preds_ = np.zeros(len(test_mask))  
print(max(index_no_false))    
preds_[index_no_false] = 1

gt = test_mask#np.concatenate([test_mask[index_no_false],test_mask[index_w_false]],0)#mask.long().numpy()
pr  = precision_score(gt, preds_)
rec= recall_score(gt, preds_)
acc = accuracy_score(gt, preds_)
accuracy.append(acc)
precision.append(pr)
recall.append(rec)
print(np.mean(acc))
print(np.mean(pr))
print(np.mean(recall))'''

import pickle as pkl
f = open('abl_last_index_mnist_0_5.pkl', 'wb')
pkl.dump(obj=[index_no_false, index_w_false], file=f)
f.close()



"""

import pickle as pkl
#from finch import FINCH

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch
from sklearn.metrics import *
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

f = open('/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/BadLabels/data_DomainNet_infograph_0_8.pkl', 'rb')
data = pkl.load(f)
f.close()
all_clean_labels = data[1]
all_noisy_labels = data[2]
all_loss = data[3]
all_indexes = data[5]
all_domains = data[6]
clean_labels = []
noisy_labels = []
losses = torch.zeros(6, max(all_indexes[0])+5) + 100
n_labels = torch.zeros(6,max(all_indexes[0])+5) + 100
c_labels = torch.zeros(6,max(all_indexes[0])+5) + 100

all_domain = torch.zeros(6, max(all_indexes[0])+5) + 100
for i in range(0,6):
    #print(i)
    index = all_indexes[i]
    losses[i][index] = torch.Tensor(all_loss[i]).squeeze() 
    n_labels[i][index] = torch.Tensor(all_noisy_labels[i])
    c_labels[i][index] = torch.Tensor(all_clean_labels[i])
    all_domain[i][index] = torch.Tensor(all_domains[i]).clone()
#print(losses)

n_labels = n_labels[0]
c_labels = c_labels[0]
test_mask = n_labels == c_labels
target = all_domain[0]
all_indexes = np.arange(0,max(all_indexes[0])+5)
index_no_false = []
index_w_false = []
final_indexes = []
accuracy = []
precision =[]
recall=[]
for d in range(5):
    domain_mask = target == d
    print(torch.sum(domain_mask.bool()))
    for k in range(182):
        #index_no_false = []
        #index_w_false = []
        #final_indexes = []
        index_no_false_cat = []
        loss_no_false_cat = []
        cat_mask = n_labels[domain_mask] == k 
        
        select_index = all_indexes[domain_mask][cat_mask] 
        loss_edl = losses[:, domain_mask][:, cat_mask]
        #print(torch.sum(domain_mask))
        loss = np.transpose(loss_edl, (1,0))
        mask = n_labels[domain_mask][cat_mask] == c_labels[domain_mask][cat_mask]
        ind = 0
        if True in mask:
            ind  = 1
        GMM_label = mask.long()
        import hnne
        #print(loss.shape)
        #print(loss.shape)
        try:
            c,_,_,_ = hnne.finch_clustering.FINCH(loss.numpy(),distance='cosine', verbose=False)
            #c, num_clust, req_c = FINCH(loss.numpy(), initial_rank=None, req_clust=2, distance='cosine', verbose=True)
            cat = c[:,0]

            #loss = torch.mean(loss,-1)
            loss_cat = []
            list_cat = []

            for i in range(np.max(cat) + 1):
                ma = cat == i
                loss_cat.append(torch.mean(loss[ma]))
                list_cat.append(i)


            list_cat = torch.Tensor(list_cat)
            loss_cat = torch.Tensor(loss_cat)

            classifier = GMM(n_components=2,max_iter=50,init_params='k-means++')
            
            classifier.fit(loss_cat.unsqueeze(-1).numpy())
            
            y_train_pred = classifier.predict(loss_cat.unsqueeze(-1).numpy())
            y_ma = y_train_pred == 0

            #print(np.sum(y_ma))


            if torch.mean(loss_cat[y_ma]) < torch.mean(loss_cat[~y_ma]):
                group_indexes = list_cat[y_ma]
                #print(group_indexes)
                #print(cat)
                for g_ind in group_indexes:
                    g_ma = torch.Tensor(cat) == g_ind
                    print(torch.sum(g_ma))
                    #print(select_index[g_ma])
                    index_no_false.extend(select_index[g_ma])
                group_indexes = list_cat[~y_ma]
                for g_ind in group_indexes:
                    g_ma = torch.Tensor(cat) == g_ind
                    index_w_false.extend(select_index[g_ma])
            else:
                #print(group_indexes)
                group_indexes = list_cat[~y_ma]
                for g_ind in group_indexes:
                    g_ma = torch.Tensor(cat) == g_ind
                    #print(select_index[g_ma])
                    index_no_false.extend(select_index[g_ma])
                group_indexes = list_cat[y_ma]
                for g_ind in group_indexes:
                    g_ma = torch.Tensor(cat) == g_ind
                    index_w_false.extend(select_index[g_ma])     
        except:
            index_no_false.extend(select_index)     
'''#print(len(test_mask[index_no_false].numpy()))
preds_ = np.zeros(len(test_mask))  
print(max(index_no_false))    
preds_[index_no_false] = 1

gt = test_mask#np.concatenate([test_mask[index_no_false],test_mask[index_w_false]],0)#mask.long().numpy()
pr  = precision_score(gt, preds_)
rec= recall_score(gt, preds_)
acc = accuracy_score(gt, preds_)
accuracy.append(acc)
precision.append(pr)
recall.append(rec)
print(np.mean(acc))
print(np.mean(pr))
print(np.mean(recall))'''
print(index_no_false)
print(index_w_false)
import pickle as pkl
f = open('domainnet_infograph_0_8.pkl', 'wb')
pkl.dump(obj=[index_no_false, index_w_false], file=f)
f.close()



