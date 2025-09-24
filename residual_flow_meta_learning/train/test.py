import torch
from torch import nn
from util.log import log

def eval(net, loader, log_path, epoch=-1, device="cpu", mark="Val"):
    #net_res_clr, head_simclr, head_simclr_syn = net
    #net_res_clr.eval()
    #head_simclr.eval()
    #head_simclr_syn.eval()
    criterion = nn.CrossEntropyLoss()

    num_correct = num_total = loss_val = 0

    for input, label, *_ in loader:
        input = input.to(device)
        label = label.to(device)
        output= net(input)
        #output = net_res_clr.backbone(x=input)
        #output =  head_simclr(output, net_res_clr)
        loss = criterion(output, label)
        loss_val += loss.item() * len(input)
        argmax = torch.argmax(output, axis=1)
        num_correct += (argmax == label).sum()
        num_total += len(input)
    
    loss_avg = loss_val / num_total
    acc = num_correct / num_total

    log('Epoch: {} Loss: {:.4f} Acc: {:.4f} ({})'.format(epoch+1, loss_avg, acc, mark), log_path) 

    return acc
