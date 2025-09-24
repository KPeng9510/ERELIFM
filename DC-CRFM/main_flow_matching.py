import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset_with_label_noise import SingleDomainData, SingleClassData, MultiDomainData, MultiDomainData_with_no_noise
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet, resnet152, MutiClassifier_simclr
from optimizer.optimizer import get_optimizer, get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision.utils import make_grid,save_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from loss.OVALoss import OVALoss
from train.test import eval
from util.log import log, save_data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.ROC import generate_OSCR
from util.util import ForeverDataIterator, ConnectedDataIterator, split_classes
import random
from model.vision_transformer import *
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from torchcfm.conditional_flow_matching import *
from torch.utils.data import ConcatDataset
assert torch.cuda.is_available()
from edl_losses import *
from torchdyn.core import NeuralODE
import torchdiffeq
import torchsde

import torchvision

from lightly import loss
from lightly import transforms
from lightly.models.modules import heads


class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=256,
            output_dim=512,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

model = 'resnet18'
target = ['photo']
source = ['sketch', 'cartoon', 'art_painting', 'photo']
asym = False
nsr = 0.2
gpu = 0

source.remove(target[0])
from torchcfm.models.unet.unet import UNetModelWrapper

if nsr == 0.5:
    ns = '0_5'
if nsr == 0.2:
    ns = '0_2'
if nsr == 0.8:
    ns = '0_8'    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import adv_attack as attack
from sklearn.mixture import BayesianGaussianMixture
# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(
        self,
        label_condition: int = 1,
        loss_condition: int = 1,
        input_dim: int = 6,
        time_dim: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.loss_dim = loss_condition
        self.label_dim = label_condition
        time_embed_dim = time_dim * 16
        self.input_layer = nn.Linear(6 + time_embed_dim*3+16, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            #Swish(),
            #nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

        
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #self.label_emb = nn.Embedding(6, time_embed_dim)
        self.loss_emb = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.label_emb = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #self.image_emb = resnet18_fast()
        self.image_proj = nn.Linear(512, 16)
    def forward(self,t: Tensor,x: Tensor,loss:Tensor, label:Tensor, img:Tensor) -> Tensor:
        #print(t)
        #print(x.shape)
        while t.dim() > 1:
            #print(timesteps.shape)
            t = t[:, 0]
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = self.time_embed(t.reshape(-1, self.time_dim).float())
        loss = self.loss_emb(loss.reshape(-1, self.time_dim).float())
        label = self.label_emb(label.reshape(-1, self.time_dim).float())
        image_emb = self.image_proj(img)
        h = torch.cat([x, t, loss, label, image_emb], dim=-1)
        h = self.input_layer(h)
        output = self.main(h)
        return output.reshape(*sz)



def eval_train_perturbed(model, all_loss, eval_loader, lam, num_class,criterion, max_dist=False, last_prob=None):
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none').cuda()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs,_, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            #print(inputs.shape)
            #print(targets.shape)
            labels_ = attack.label_flip(model, inputs, targets, num_classes=num_class, step_size=lam, num_steps=1)

            outputs,_ = model.c_forward(inputs)
            loss = CE(outputs, labels_)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    # ls = losses.tolist()
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)

    # fit a BayesGMM to the loss
    prob_sum = np.zeros((len(input_loss),))
    for i in range(1):
        gmm = BayesianGaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4,
                                      weight_concentration_prior_type='dirichlet_process')
        gmm.fit(input_loss.cpu().numpy())
        prob = gmm.predict_proba(input_loss)
        print('gmm%d.means: ' % i, gmm.means_)
        print('gmm%d.converged: ' % i, gmm.converged_)
        print('gmm%d.n_iter_' % i, gmm.n_iter_)
        if max_dist:
            prob = prob[:, gmm.means_.argmax()]
        else:
            prob = prob[:, gmm.means_.argmin()]
        # print(prob.shape)
        prob_sum += prob
        # print(prob_sum)
    prob = prob_sum / 1.
    # print(prob)
    if not gmm.converged_ and last_prob is not None:
        prob = last_prob
        print('*** BayesGMM is not converged .. Load last probability .. ***')

    return torch.Tensor(prob), all_loss
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def adjust_lambda(epoch, warm_up, lam, p_length=1):
    current = np.clip(1 - ((epoch - warm_up) / p_length), 0.0, 1.0)
    return lam * float(current)

def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def knn(ref, query, k):
    ref_c =torch.stack([ref] * query.shape[-1], dim=0).permute(0, 2, 1).reshape(-1, 2).transpose(0, 1)
    query_c = torch.repeat_interleave(query, repeats=ref.shape[-1], dim=1)
    delta = query_c - ref_c
    distances = torch.sqrt(torch.pow(delta, 2).sum(dim=0))
    distances = distances.view(query.shape[-1], ref.shape[-1])
    sorted_dist, indices = torch.sort(distances, dim=-1)
    return sorted_dist[:, :k], indices[:, :k]

def wrap(manifold, samples):
    center = torch.cat([torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1)
    samples = torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2

    return manifold.expmap(center, samples)

def make_weights_for_balanced_classes(labels,labels_d, nclasses, ndomains):                        
    count = torch.zeros(nclasses, ndomains)                                                      
    for item, itemd in zip(labels, labels_d):                                                         
        count[item, itemd] += 1 
    print(count)                                                    
    weight_per_class = torch.zeros(nclasses, ndomains)                                    
    N = len(labels)
    for j in range(ndomains):                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i,j] = N/float(count[i,j])                                 
    weight = [0] * len(labels)                                              
    for idx, (val,val_d) in enumerate(zip(labels, labels_d)):                                          
        weight[idx] = weight_per_class[val, val_d]                                  
    return weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=source)
    parser.add_argument('--target-domain', nargs='+', default=target)
    parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house',])
    parser.add_argument('--unknown-classes', nargs='+', default=['person'])
    '''parser.add_argument('--dataset', default='OfficeHome')
    parser.add_argument('--source-domain', nargs='+', default=['Real_World', 'Product', 'Art'])
    parser.add_argument('--target-domain', nargs='+', default=['Clipart'])
    parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
         'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
         'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
         'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
         'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
         'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
         'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
        
         ])
    parser.add_argument('--unknown-classes', nargs='+', default=[ 
         'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
         'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
         'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 
         'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
         'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
         'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
         ])'''
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
    #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
    #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
    #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
    #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
    #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
    #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
    #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
        
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[      
    #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
    #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 
    #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
    #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
    #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
    #     ])

    # parser.add_argument('--dataset', default='DigitsDG')
    #parser.add_argument('--source-domain', nargs='+', default=['syn', 'mnist_m', 'svhn'])
    #parser.add_argument('--target-domain', nargs='+', default=['mnist'])
    #parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    #parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default=gpu)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--net-name', default=model)
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--meta-lr', type=float, default=0.00001)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    if asym == True:
        parser.add_argument('--save-dir', default='path/to/PACS_'+model+'_asym_snr_'+ns+'_'+target[0]+'/')
    else:
        parser.add_argument('--save-dir', default='path/to/PACS_'+model+'_sym_snr_'+ns+'_'+target[0]+'/')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)
    
    args = parser.parse_args()
    direct = args.save_dir
    if not os.path.exists(direct):
        os.mkdir(direct)
        os.mkdir(direct + '/model')
        os.mkdir(direct + '/model/val')
        os.mkdir(direct + '/param')
        os.mkdir(direct + '/log')
    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = args.known_classes
    unknown_classes = sorted(args.unknown_classes)
    crossval = not args.no_crossval   
    gpu = args.gpu
    batch_size = args.batch_size
    net_name = args.net_name
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before
    transform = transforms.SimCLRTransform(input_size=224, cj_prob=0.8)
    #criterion_simclr = loss.NTXentLoss(temperature=0.98)
    #criterion_simclr = criterion_simclr.cuda()
    #head_simclr = MutiClassifier_simclr().cuda()
    #head_simclr_syn = MutiClassifier_simclr().cuda()
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma = 0.0
    model_unet = UNetModelWrapper(
        dim=(3, 224, 224),
        class_cond=True,
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=6,
    ).cuda()
    optimizer_unet = torch.optim.Adam(model_unet.parameters(), lr=2e-4)
    #FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    # Users can try target FM by changing the above line by
    # FM = TargetConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(model_unet, solver="euler", sensitivity="adjoint")

    def warmup_lr(step):
        return min(step, 5000) / 5009
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer_unet, lr_lambda=warmup_lr)






    if dataset == 'PACS':
        train_dir = 'path/to/PACS_train'
        val_dir = 'path/to//PACS_crossval'
        test_dir = ['path/to//PACS_train', 'path/to//PACS_crossval']
        sub_batch_size = batch_size // 2
        small_img = False
  
        
        # Variable definition
    log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.6) if save_later else 0

    log('GPU: {}'.format(gpu), log_path)

    log('Loading path...', log_path)

    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    log('Loading dataset...', log_path)

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1

    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6

    log('Group length: {}'.format(group_length), log_path)
    
    group_index = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index)

    loader = MultiDomainData(root_dir=train_dir, domain=source_domain, classes=known_classes, nsr=nsr, asymetric=asym,transform=get_transform("train", small_img=small_img), simclr_transform=transform, train=True)
    N = len(loader)
    

    import pickle as pkl
    f = open('index_no_w_2.pkl', 'rb+')
    indexes = pkl.load(f)
    f.close()
    clean_indexes = indexes[0]
    noisy_indexes = indexes[1]
    
    trainset_1 = torch.utils.data.Subset(loader, clean_indexes)
    trainset_2 = torch.utils.data.Subset(loader, noisy_indexes)
    
    all_domains = []
    all_index = []
    all_labels = []

    loader_noisy_forever = ForeverDataIterator(trainset_2)

    num_samples_clean = len(trainset_1)
    num_samples_noisy = len(trainset_2)


    loader_clean = DataLoader(dataset=trainset_1, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    loader_noisy = DataLoader(dataset=trainset_2, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    loader_clean_iter = ForeverDataIterator(loader_clean)
    loader_noisy_iter = ForeverDataIterator(loader_noisy)

    loader = DataLoader(dataset=loader, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1)

    
    #if crossval:
    val_k = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    #else:
    #    val_k = None
    val_u = get_dataloader(root_dir=val_dir, domain=source_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)

    test_k = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)


    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier



    n_epochs = 400000
    model_unet = model_unet
    for epoch in range(n_epochs):
        model_unet = model_unet.cuda()
        img, (view0, view1), labels_x, labels_original, target, index, path = next(loader_clean_iter)
        optimizer_unet.zero_grad()
        x1 = img.cuda()
        y = labels_x
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        #print(y)
        vt = model_unet(t.cuda(), xt.cuda(), y.cuda())
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_unet.parameters(), 2.0)
        optimizer_unet.step()
        sched.step()
        print(f"epoch: {epoch}, loss: {loss.item():.4}", end="\r")

        if epoch % 1000 == 0:
            model_unet.eval()
            USE_TORCH_DIFFEQ = True
            generated_class_list = torch.arange(6, device=device)#.repeat(6)
            with torch.no_grad():
                if USE_TORCH_DIFFEQ:
                    traj = torchdiffeq.odeint(
                        lambda t, x: model_unet.forward(t, x, generated_class_list),
                        torch.randn(6, 3, 224, 224, device=device),
                        torch.linspace(0, 1, 100, device=device),
                        method="euler",
                    )
                else:
                    traj = node.trajectory(
                        torch.randn(6, 3, 224, 224, device=device),
                        t_span=torch.linspace(0, 1, 100, device=device),
                    )

            grid = make_grid(
                traj[-1, :].view([-1, 3, 224, 224]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=3
            )
            img = grid
            img = ToPILImage()(grid)
            traj = traj[-1, :].view([-1, 3, 224, 224]).clip(-1, 1)
            traj = traj / 2 + 0.5

            save_image(traj, f"generated/generated_FM_images_step_{epoch}.png", nrow=3)
            #plt.imsave('generated/test'+str(int(epoch))+'.png', img.cpu().permute(1,2,0).numpy())
            torch.save(model_unet.cpu().state_dict(), 'generated/flow_matching'+str(int(epoch))+'.pt')
            model_unet.train()