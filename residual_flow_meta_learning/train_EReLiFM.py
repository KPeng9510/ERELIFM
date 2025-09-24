import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset_with_label_noise import SingleDomainData, SingleClassData, MultiDomainData, MultiDomainData_with_no_noise
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet, resnet152, MutiClassifier_simclr
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from torchvision.transforms import RandomAffine, RandomRotation, CenterCrop, Resize

from train.test import eval
from util.log import log, save_data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.ROC import generate_OSCR
from util.util import ForeverDataIterator, ConnectedDataIterator, split_classes
import random
from model.vision_transformer_our import *
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from torchcfm.conditional_flow_matching import *
from torch.utils.data import ConcatDataset
assert torch.cuda.is_available()
import shutil
from torchdiffeq import odeint_adjoint as odeint
import random
from functools import partial
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from edl_losses import *
from torchdyn.core import NeuralODE
import torchdiffeq
import torchsde
from models import create_network
from torchdiffeq import odeint_adjoint as odeint
import torchvision

from lightly import loss
from lightly import transforms
from lightly.models.modules import heads

def sample_from_model(model, x_0):
    t = torch.tensor([0.5, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image

model = 'vit_base'
target = ['sketch']
source = ['sketch', 'cartoon', 'art_painting', 'photo']
asym = True
nsr = 0.5
gpu = 0

source.remove(target[0])

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
    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default=gpu)
    parser.add_argument('--batch-size', type=int, default=4)

    parser.add_argument('--net-name', default=model)
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--meta-lr', type=float, default=0.001)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    if asym == True:
        parser.add_argument('--save-dir', default='save/path/')
    else:
        parser.add_argument('--save-dir', default='save/path/')
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
    resize = Resize((224,224))
    dataset = args.dataset
    epochs = args.num_epoch
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
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

    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = "/path/to/DIT/checkpoint"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint.items()
                    if k.startswith(prefix)}

    model.load_state_dict(adapted_dict, strict=True)
    model.cuda()

    first_stage_model = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').cuda()
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    global_step = 0
    
    train_dir = '/path/to/PACS/train'
    val_dir = '/path/to/PACS/test'
    test_dir = ['/path/to/PACS/train', '/path/to/PACS/test']
    sub_batch_size = batch_size // 2
    small_img = False
    
    vf = MLP(# Vector field in the ambient space.
        input_dim=6,
        hidden_dim=32,
    )
    vf.to(device)
    optimizer_fm = torch.optim.SGD(vf.parameters(), lr=0.01)
    FM = ConditionalFlowMatcher(sigma=0.0)
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

    loader = MultiDomainData(root_dir=train_dir, domain=source_domain, classes=known_classes, nsr=nsr, asymetric=asym,transform=get_transform("train", small_img=small_img),transform2=get_transform("train2", small_img=small_img), simclr_transform=transform, train=True)
    N = len(loader)

    import pickle as pkl
    f = open('/path/to/clean-noisy-partition', 'rb+')
    indexes = pkl.load(f)
    f.close()
    clean_indexes = indexes[0]
    noisy_indexes = indexes[1]
        trainset_1 = torch.utils.data.Subset(loader, clean_indexes)
    trainset_2 = torch.utils.data.Subset(loader, noisy_indexes)
    loader_noisy_forever = ForeverDataIterator(trainset_2)

    num_samples_clean = len(trainset_1)
    num_samples_noisy = len(trainset_2)


    loader_clean = DataLoader(dataset=trainset_1, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    loader_noisy = DataLoader(dataset=trainset_2, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    loader_clean_iter = ForeverDataIterator(loader_clean)

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
    s_prev_confidence = torch.ones(N).cuda()*1/N
    w_prev_confidence = torch.ones(N).cuda()*1/N
    ws_prev_confidence = torch.ones(N).cuda()*1/N
    
    w_probs = torch.zeros(N, len(known_classes)).cuda()
    s_probs = torch.zeros(N, len(known_classes)).cuda()

    clean_flags = torch.zeros(N).bool().cuda()
    hard_flags = torch.zeros(N).bool().cuda()
    correction_flags = torch.zeros(N).bool().cuda()
    weak_flags = torch.zeros(N).bool().cuda()
    w_selected_flags = torch.zeros(N).bool().cuda()
    s_selected_flags = torch.zeros(N).bool().cuda()
    selected_flags = torch.zeros(N).bool().cuda()
    class_weight = torch.ones(len(known_classes)).cuda()

    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier

    if net_name == 'resnet18':
        net = muticlassifier(net=resnet18_fast(), num_classes=num_classes+1)
        net_u = muticlassifier(net=resnet18_fast(), num_classes=num_classes)
    elif net_name == 'resnet50':
        net = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    elif net_name == "convnet":
        net = muticlassifier(net=ConvNet(), num_classes=num_classes, feature_dim=256)
    elif net_name == "resnet152":
        net = muticlassifier(net=resnet152(), num_classes=num_classes, feature_dim=2048)
    elif net_name == "vit_base":
        net_ = vit_base_patch16_224_k(pretrained=True)
        net_.head = torch.nn.Identity().cuda()
        net = muticlassifier(net=net_, num_classes=num_classes+1, feature_dim=768)
        net_u = muticlassifier(net=net_, num_classes=num_classes, feature_dim=768)
    net_res = resnet18_fast()
    node = NeuralODE(vf, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)


    log('Network: {}'.format(net_name), log_path)
    log('Number of epoch: {}'.format(num_epoch), log_path)
    log('Learning rate: {}'.format(lr), log_path)
    log('Meta learning rate: {}'.format(meta_lr), log_path)

    if num_epoch_before != 0:
        log('Loading state dict...', log_path)  
        if save_best_test == False:
            net.load_state_dict(torch.load(model_val_path))
        else:
            net.load_state_dict(torch.load(model_test_path))
        for epoch in range(num_epoch_before):
            scheduler.step()
        log('Number of epoch-before: {}'.format(num_epoch_before), log_path)

    log('Without binary classifier: {}'.format(without_bcls), log_path)
    log('Share Parameter: {}'.format(share_param), log_path)

    log('Start training...', log_path)  

    if crossval:
        best_val_acc = 0 
    else:
        best_val_acc = 0
    best_val_test_acc = []
    best_test_acc = best_test_acc_ = 0.0
    best_test_test_acc = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none',label_smoothing=0.0)
    ovaloss = OVALoss()
    if without_bcls:
        ovaloss = lambda *args: 0
    exp_domain_index = 0   
    exp_group_num = (num_group-1) // 3 + 1
    exp_group_index = random.sample(group_index, exp_group_num)
    torch.set_printoptions(threshold=10_000)
    domain_index_list = [i for i in range(num_domain)]
    momentum = 0.99
    fast_parameters = list(net.parameters())
    for weight in net.parameters():
        weight.fast = None
    losses_1 = []
    losses_2 = []
    all_rep = []
    prob1, prob2 = None, None

    optimizer_simclr = get_optimizer(net=[*net.parameters()], instr=optimize_method, lr=lr, nesterov=nesterov)
    scheduler = get_scheduler(optimizer=optimizer_simclr, instr=schedule_method, step_size=int(10000*0.2), gamma=0.1)

    fast_parameters = list(net.parameters())
    for weight in net.parameters():
        weight.fast = None
    net.zero_grad()

    for epoch in range(epochs):

   
        net.train()
        #for batch_idx, (img, (view0, view1), labels_x, labels_original, target, index, path) in enumerate(loader_clean):
        for batch_idx in range(len(loader)): 
            img, img2, labels_x, labels_original, target, index, path = next(loader_clean_iter)
            net.train()
            z_0 = torch.zeros([img.shape[0], 4,32,32])
            z_logits = net(img.cuda())
            celoss = criterion(z_logits, labels_x.cuda())
            with torch.no_grad():
                rand = torch.randn_like(z_0)
                rand2 = torch.randn_like(z_0)
                if labels_x is not None:
                    y = labels_x
                domain_list = list(torch.arange(0,len(source_domain)))
                select_domain = []
                for k in range(len(img)):
                    domain_list.remove(target[k])
                    select_domain.append(random.choice(domain_list))
                    domain_list = list(torch.arange(0,len(source_domain)))
                select_class = []
                class_list = list(torch.arange(0,len(known_classes)))
                for k in range(len(img)):
                    class_list.remove(labels_x[k])
                    select_class.append(random.choice(class_list))
                    class_list = list(torch.arange(0,len(known_classes)))
                d_t = torch.Tensor(select_domain).long().cuda()
                y_t = torch.Tensor(select_class).long().cuda()
                bs = len(y)
                y_target = torch.cat([y.cuda(),y_t.cuda()],0)
                y = torch.cat([y.cuda(),y.cuda()],0)
                d_s = torch.cat([target.cuda(), target.cuda()],0)
                d_t = torch.cat([d_t.cuda(), d_t.cuda()], 0)        
                sample_model = partial(model, y=y.cuda(),y_t=y_target.cuda(), d_s=d_s.long().cuda(), d_t=d_t.long().cuda())
                rand = torch.cat([rand.cuda(), rand2.cuda()],0)
                fake_sample = sample_from_model(sample_model, rand.cuda())[-1]
                fake_sample = first_stage_model.decode(fake_sample / 0.18215).sample

                fake_sample2 = fake_sample[bs:]
                fake_sample = fake_sample[:bs]



            augmented_samples = resize(fake_sample) + img.cuda()
            augmented_samples_2 = resize(fake_sample2) + img.cuda()
            augmented_logits = net(augmented_samples.cuda())
            augmented_logits_2 = net(augmented_samples_2.cuda())
            ce_loss_augmented = criterion(augmented_logits, labels_x.cuda())
            ce_loss_augmented_2 = criterion(augmented_logits_2, ((torch.ones_like(labels_x).cuda())*len(known_classes)).long())
            loss_correct = celoss.mean() + 0.1*ce_loss_augmented.mean() + 0.01 *ce_loss_augmented_2.mean() #+ edlloss.mean()#0.1   0.01

            grad = torch.autograd.grad(loss_correct, net.parameters(),create_graph=True, allow_unused=True)
            for k, weight in enumerate(net.parameters()):
                if grad[k] is not None:
                    if weight.fast is None:
                        weight.fast = weight - meta_lr * grad[k]
                    else:
                        weight.fast = weight.fast - meta_lr * grad[
                            k]
            img_noisy, _, labels_x_noisy, labels_original_noisy, target_noisy, index_noisy,path = next(loader_noisy_iter)
            z_logits_noisy = net(img_noisy.cuda())
            label_self = torch.argmax(z_logits_noisy[:,:-1], -1)
            meta_test_loss = edl_soft_mse_loss(z_logits_noisy[:,:-1], label_self.cuda())
            all_loss = meta_test_loss.mean() + loss_correct.mean()
            all_loss.backward()
            optimizer_simclr.step()
            optimizer_simclr.zero_grad()
            scheduler.step()
            fast_parameters = list(net.parameters())
            for weight in net.parameters():
                weight.fast = None
            net.zero_grad()

            if batch_idx%100 ==0:
                print(batch_idx)
                net.eval()
                if test_u != None:
                    output_k_sum = []
                    b_output_k_sum = []
                    label_k_sum = []  
                    with torch.no_grad():  
                        for input, label, *_ in val_k:
                            input = input.to(device)
                            label = label.to(device)
 
                            output = net(input)
                            output = F.softmax(output, 1)
                            b_output = output 

                            output_k_sum.append(output)
                            b_output_k_sum.append(b_output)
                            label_k_sum.append(label)

                    output_k_sum = torch.cat(output_k_sum, dim=0)
                    b_output_k_sum = torch.cat(b_output_k_sum, dim=0)
                    label_k_sum = torch.cat(label_k_sum)

                    output_u_sum = []
                    b_output_u_sum = []

                    with torch.no_grad():
                        for input, *_ in val_u:
                            input = input.to(device)
                            label = label.to(device)
                            output = net(input)
                            output = F.softmax(output, 1)
                            b_output = output

                            output_u_sum.append(output)
                            b_output_u_sum.append(b_output)

                    output_u_sum = torch.cat(output_u_sum, dim=0)
                    b_output_u_sum = torch.cat(b_output_u_sum, dim=0)
                    best_score = 0
                    best_threshold = 0
                    best_overall_acc = 0.0
                    best_thred_acc = 0.0
                    best_overall_Hscore = 0.0
                    best_thred_Hscore = 0.0
                    for thres in range(0,100):
                        threshold = 0.01*thres
                        num_correct_k = num_correct_u = 0
                        num_total_k = num_total_u = 0

                        argmax_k = torch.argmax(output_k_sum, axis=1)
                        for i in range(len(argmax_k)):
                            if argmax_k[i] == label_k_sum[i] and output_k_sum[i][argmax_k[i]] >= threshold:
                                num_correct_k +=1
                        num_total_k += len(output_k_sum)


                        argmax_u = torch.argmax(output_u_sum, axis=1)
                        for i in range(len(argmax_u)):
                            if output_u_sum[i][argmax_u[i]] < threshold:
                                num_correct_u +=1
                        num_total_u += len(output_u_sum)


                        acc_k = num_correct_k / num_total_k
                        acc_u = num_correct_u / num_total_u
                        acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
                        hs = 2*acc_k*acc_u/(acc_k + acc_u)

                        if acc > best_overall_acc:
                            best_overall_acc = acc
                            best_thred_acc = threshold
                        if hs > best_overall_Hscore:
                            best_overall_Hscore = hs
                            best_threshold = threshold
                if test_u != None:
                    output_k_sum = []
                    b_output_k_sum = []
                    label_k_sum = []  
                    with torch.no_grad():  
                        for input, label, *_ in test_k:
                            input = input.to(device)
                            label = label.to(device)
                            output = net(input)
                            output = F.softmax(output, 1)
                            b_output = output
                            b_output = F.softmax(b_output, 1)

                            output_k_sum.append(output)
                            b_output_k_sum.append(b_output)
                            label_k_sum.append(label)

                    output_k_sum = torch.cat(output_k_sum, dim=0)
                    b_output_k_sum = torch.cat(b_output_k_sum, dim=0)
                    label_k_sum = torch.cat(label_k_sum)

                    output_u_sum = []
                    b_output_u_sum = []
        
                    with torch.no_grad():
                        for input, *_ in test_u:
                            input = input.to(device)
                            label = label.to(device)

                            output = net(input)
                            
                            output = F.softmax(output, 1)
                            b_output = output

                            b_output = F.softmax(b_output, 1)

                            output_u_sum.append(output)
                            b_output_u_sum.append(b_output)

                    output_u_sum = torch.cat(output_u_sum, dim=0)
                    b_output_u_sum = torch.cat(b_output_u_sum, dim=0)

        #################################################################################
                    log('C classifier:', log_path)

                    conf_k, argmax_k = torch.max(output_k_sum, axis=1)
                    conf_u, _ = torch.max(output_u_sum, axis=1)

                    OSCR_C = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)

            
                    log('OSCR_C: {:.4f}'.format(OSCR_C), log_path) 
                

                    best_score = 0
                    best_overall_acc = 0.0
                    best_thred_acc = 0.0
                    best_overall_Hscore = 0.0
                    best_thred_Hscore = 0.0
                    num_correct_k = num_correct_u = 0
                    num_total_k = num_total_u = 0

                    argmax_k = torch.argmax(output_k_sum, axis=1)
                    print('threshold:',best_threshold)
                    for i in range(len(argmax_k)):
                        if argmax_k[i] == label_k_sum[i] and output_k_sum[i][argmax_k[i]] >= best_threshold:
                            num_correct_k +=1
                    num_total_k += len(output_k_sum)


                    argmax_u = torch.argmax(output_u_sum, axis=1)
                    for i in range(len(argmax_u)):
                        if output_u_sum[i][argmax_u[i]] < best_threshold:
                            num_correct_u +=1
                    num_total_u += len(output_u_sum)


                    acc_k = num_correct_k / num_total_k
                    acc_u = num_correct_u / num_total_u
                    acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
                    hs = 2*acc_k*acc_u/(acc_k + acc_u)

                    log('H_score: {:.4f}'.format(hs), log_path) 
        ###################################################################################################################
                    OSCR_B = 0.0


                else:
                    OSCR_C = OSCR_B = 0 
                    log("", log_path)

                
                if val_k != None:
                    acc = eval(net=net, loader=val_k, log_path=log_path, epoch=epoch, device=device, mark="Val") 
                
                acc_ = eval(net=net, loader=test_k, log_path=log_path, epoch=epoch, device=device, mark="Test")     
                torch.save(net.state_dict(), '/'.join(model_val_path.split('/')[:-1]) + 'results_'+'_acc_' + str(acc_.item())[:6]+ str(int(epoch)) +'_OSCR_C_' + str(OSCR_C.item())[:6]+'_H_score_' + str(hs)[:6]+'.pth')
                
                if val_k != None:           
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_test_acc_ = acc_
                        best_val_test_acc = [{
                            "test_acc": "%.4f" % acc_.item(),
                            "OSCR_C": "%.4f" % OSCR_C,
                            "OSCR_B": "%.4f" % OSCR_B,
                            "H_score": "%.4f" % hs,
                        }]
                        best_val_model = copy.deepcopy(net.state_dict())
                        torch.save(best_val_model, model_val_path)
                    elif acc == best_val_acc:
                        best_val_test_acc.append({
                            "test_acc": "%.4f" % acc_.item(),
                            "OSCR_C": "%.4f" % OSCR_C,
                            "OSCR_B": "%.4f" % OSCR_B,
                            "H_score": "%.4f" % hs,
                        })
                        if acc_ > best_test_acc_:
                            best_test_acc_ = acc_
                            best_val_model = copy.deepcopy(net.state_dict())
                            torch.save(best_val_model, model_val_path)
                    log("Current best val accuracy is {:.4f} (Test: {})".format(best_val_acc, best_val_test_acc), log_path)
                    
                if acc_ > best_test_acc:
                    best_test_acc = acc_    
                    best_test_test_acc = [{
                        "OSCR_C": "%.4f" % OSCR_C,
                        "OSCR_B": "%.4f" % OSCR_B,
                        "H_score": "%.4f" % hs,

                    }]    
                    if save_best_test:
                        best_test_model = copy.deepcopy(net.state_dict())
                        torch.save(best_test_model, model_test_path)
                log("Current best test accuracy is {:.4f} ({})".format(best_test_acc, best_test_test_acc), log_path)

            if epoch+1 == renovate_step:
                    log("Reset accuracy history...", log_path)

                    best_val_acc = 0
                    best_val_test_acc = []
                    best_test_acc = 0
                    best_test_test_acc = []
