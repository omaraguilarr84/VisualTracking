#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:32 2019

@author: aayush
"""

# python train.py --model densenet_og --expname TRAIN_OG --bs 16 --epochs 10 --useGPU True --dataset ../../../../../../../storage/ice1/shared/bmed6780/mip_group_4/'openEDS Dataset'/openEDS/openEDS/

import os
import sys
# sys.path.append('/models')
from models import *
from models.models import model_dict
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from dataset import IrisDataset
import torch
from utils import mIoU, CrossEntropyLoss2d,total_metric,get_nparams,Logger,GeneralizedDiceLoss,SurfaceLoss
import numpy as np
from dataset import transform
from opt import parse_args
# from custom_collate import custom_collate_fn
# import os
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
#%%

def plot_metrics(losses, ious, name, save_dir):
    import matplotlib.pyplot as plt

    # Unpack
    losses_sl, losses_CE, losses_dice, losses_total = losses

    # Convert each list of tensors to a list of floats
    def to_float_list(ts):
        out = []
        for t in ts:
            if hasattr(t, 'detach'):
                t = t.detach()
            out.append(float(t.cpu().numpy() if isinstance(t, np.ndarray) else t))
        return out

    sl = to_float_list(losses_sl)
    ce = to_float_list(losses_CE)
    di = to_float_list(losses_dice)
    tot= to_float_list(losses_total)

    # Create epoch index arrays
    epochs_losses = range(1, len(sl) + 1)
    epochs_iou    = range(1, len(ious) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: all losses on the same plot
    ax1.plot(epochs_losses, sl,  label='Surface Loss')
    ax1.plot(epochs_losses, ce,  label='Cross Entropy Loss')
    ax1.plot(epochs_losses, di,  label='Dice Loss')
    ax1.plot(epochs_losses, tot, label='Total Loss')
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Right: IoU over epochs
    ax2.plot(epochs_iou, ious, label='Mean IoU')
    ax2.set_title('Mean IoU')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(f'{save_dir}/{name}_training_metrics.png')
    plt.show()


def lossandaccuracy(loader,model,factor):
    epoch_loss = []
    ious = []    
    model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
#            print (len(batchdata))
            img,labels,index,spatialWeights,maxDist=batchdata
            data = img.to(device)

            target = labels.to(device).long()  
            output = model(data)
            
            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target)
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
            
            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))
            
            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-factor)*loss_sl+factor*(loss_dice)+loss 
            
            epoch_loss.append(loss.item())
            predict = get_predictions(output)
            iou = mIoU(predict,labels)
            ious.append(iou)
    return np.average(epoch_loss),np.average(ious)

def pad_collate(batch):
    imgs, labs, idxs, spats, dists = zip(*batch)

    # find the max spatial size in this batch
    maxH = max(x.shape[-2] for x in imgs)
    maxW = max(x.shape[-1] for x in imgs)

    def pad_tensor(t, pad_value=0):
        # ensure t is [C,H,W]
        if t.dim() == 2:      # a label or single‐channel mask
            t = t.unsqueeze(0)
        C, H, W = t.shape
        out = t.new_full((C, maxH, maxW), pad_value)
        out[:, :H, :W] = t
        return out

    # stack each field
    img_batch  = torch.stack([pad_tensor(img)             for img  in imgs], dim=0)
    lab_batch  = torch.stack([pad_tensor(lab, pad_value=-1) for lab  in labs], dim=0)
    spat_batch = torch.stack([pad_tensor(torch.from_numpy(sp).unsqueeze(0))
                                for sp   in spats], dim=0)
    dist_batch = torch.stack([pad_tensor(torch.from_numpy(dm)) 
                                for dm   in dists], dim=0)

    # optionally remove that extra channel dim on labels if you want [B,H,W]
    lab_batch = lab_batch.squeeze(1)

    return img_batch, lab_batch, idxs, spat_batch, dist_batch

#%%
if __name__ == '__main__':
    start = time()
    args = parse_args()
    kwargs = vars(args)

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    if args.useGPU:
        if args.device is not None:
            device=torch.device(args.device)
        else:
            device=torch.device("cuda")
        torch.cuda.manual_seed(12)
    else:
        device=torch.device("cpu")
        torch.manual_seed(12)
        
    torch.backends.cudnn.deterministic=False
    
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)
    
    LOGDIR = '../../../scratch/train/logs/{}'.format(args.expname)
    os.makedirs(LOGDIR,exist_ok=True)
    os.makedirs(LOGDIR+'/models',exist_ok=True)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))
    
    model = model_dict[args.model]
    model  = model.to(device)
    # torch.save(model.state_dict(), '{}/models/{}{}.pkl'.format(LOGDIR,args.expname,'_0'))
    model.train()
    nparams = get_nparams(model)
    
    try:
        from torchsummary import summary
        summary(model,input_size=(1,640,400))
        print("Max params:", 1024*1024/4.0)
        logger.write_summary(str(model.parameters))
    except:
        print ("Torch summary not found !!!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()

    
    Path2file = args.dataset
    train = IrisDataset(filepath = Path2file,split='train',
                             transform = transform, **kwargs)
    
    valid = IrisDataset(filepath = Path2file , split='validation',
                            transform = transform, **kwargs)
    
    trainloader = DataLoader(train, batch_size = args.bs,
                             shuffle=True, num_workers=args.workers,
                             collate_fn=pad_collate,
                             )
    
    validloader = DataLoader(valid, batch_size = args.bs,
                             shuffle= False, 
                             num_workers=args.workers,
                             collate_fn=pad_collate
                             )
 
    test = IrisDataset(filepath = Path2file , split='test',
                            transform = transform, **kwargs)
    
    testloader = DataLoader(test, batch_size = args.bs,
                             shuffle=False, 
                             num_workers=args.workers,
                             collate_fn=pad_collate
                             )



#    alpha = 1 - np.arange(1,args.epochs)/args.epoch
    ##The weighing function for the dice loss and surface loss 
    alpha=np.zeros(((args.epochs)))
    alpha[0:np.min([125,args.epochs])]=1 - np.arange(1,np.min([125,args.epochs])+1)/np.min([125,args.epochs])
    if args.epochs>125:
        alpha[125:]=1
    ious = []        
    losses_sl = []
    losses_CE = []
    losses_dice = []
    losses_total = []
    for epoch in tqdm(range(args.epochs), total=len(range(args.epochs)), desc="Epoch"):
        for i, batchdata in tqdm(enumerate(trainloader), total=len(trainloader),):
            temp_ious = []
            temp_sl = []
            temp_CE = []
            temp_dice = []
            temp_total = []
#            print (len(batchdata))
            img,labels,index,spatialWeights,maxDist= batchdata
            data = img.to(device)
            target = labels.to(device).long()
            # target = labels.float()
            # target = F.interpolate(target.unsqueeze(1).float(), size=(256, 256), mode='nearest').squeeze(1).to(device).long()

            optimizer.zero_grad()            
            output = model(data)
            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target)
            temp_CE.append(CE_loss)
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
            
            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            temp_dice.append(loss_dice)
            
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))
            temp_sl.append(loss_sl.cpu())

            # maxDist_resized = F.interpolate(maxDist.to(device).unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False).squeeze(1)
            # loss_sl = torch.mean(criterion_SL(output.to(device), maxDist_resized))

            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-alpha[epoch])*loss_sl+alpha[epoch]*(loss_dice)+loss
            temp_total.append(loss)
#            
            predict = get_predictions(output)
            iou = mIoU(predict,labels)
            temp_ious.append(iou)
    
            if i%10 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch,i,len(trainloader),loss.item()))
    
            loss.backward()
            optimizer.step()
        
        ious.append(np.mean(temp_ious))
        losses_sl.append(np.mean(temp_sl))
        losses_CE.append(np.mean(temp_CE))
        losses_dice.append(np.mean(temp_dice))
        losses_total.append(np.mean(temp_total))

        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch,np.average(ious)))
        lossvalid , miou = lossandaccuracy(validloader,model,alpha[epoch])
        totalperf = total_metric(nparams,miou)
        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
        logger.write(f.format(epoch,lossvalid, miou,nparams,totalperf))
        
        scheduler.step(lossvalid)
        
        ##save the model every epoch
        if epoch %1 == 0:
            torch.save(model.state_dict(), '{}/models/{}{}.pkl'.format(LOGDIR, args.expname, epoch))

        ##visualize the ouput every 5 epoch
        if epoch %5 ==0:
            # os.makedirs('test/epoch/labels/',exist_ok=True)
            # os.makedirs('test/epoch/output/',exist_ok=True)
            # os.makedirs('test/epoch/mask/',exist_ok=True)

            os.makedirs(f'{LOGDIR}/test/epoch/labels/', exist_ok=True)
            os.makedirs(f'{LOGDIR}/test/epoch/output/', exist_ok=True)
            os.makedirs(f'{LOGDIR}/test/epoch/mask/', exist_ok=True)
            
            with torch.no_grad():
                for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
                    img,labels,index,x,maxDist= batchdata
                    data = img.to(device)       
                    output = model(data)            
                    predict = get_predictions(output)
                    for j in range (len(index)):       
                        np.save('{}/test/epoch/labels/{}.npy'.format(LOGDIR, index[j]),predict[j].cpu().numpy())
                        try:
                            plt.imsave('{}/test/epoch/output/{}.jpg'.format(LOGDIR, index[j]),255*labels[j].cpu().numpy())
                        except:
                            pass
                        pred_img = predict[j].cpu().numpy()/3.0
                        inp = img[j].squeeze() * 0.5 + 0.5
                        img_orig = np.clip(inp,0,1)
                        img_orig = np.array(img_orig)
                        # img_orig_resized = np.resize(img_orig, (256,256))
                        combine = np.hstack([img_orig,pred_img])
                        plt.imsave('{}/test/epoch/mask/{}.jpg'.format(LOGDIR, index[j]),combine)
    
    end = time()
    print(f"Total training time: {(end-start)/60} mins")
    logger.write('Total training time:{}'.format((end-start)/60))
    plot_metrics((losses_sl, losses_CE, losses_dice, losses_total), 
                 ious, 
                 name=args.expname,
                 save_dir=LOGDIR)

