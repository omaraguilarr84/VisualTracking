#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""

# python test.py --model densenet_og --load models/weights/logs/TRAIN_OG/models/dense_net9.pkl --bs 4 --dataset ../../../../../../../storage/ice1/shared/bmed6780/mip_group_4/'openEDS Dataset'/openEDS/openEDS/ --expname OG

import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args

import sys
# sys.path.append('/models')
# sys.path.append('/models/weights')
from models import *
from models.weights import *
from models.models import model_dict
from tqdm import tqdm
from utils import get_predictions, compute_mean_iou
from time import time
#%%
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


if __name__ == '__main__':

    start = time()
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    filepath = os.path.join(os.getcwd(), filename)
    print(filepath)
    # print(os.path.abspath("/models/weights/TRAIN_OG/models/dense_net9.pkl"))
    # print(os.path.exists("/models/weights/TRAIN_OG/models/dense_net9.pkl"))
    if not os.path.exists(filepath):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filepath))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = args.dataset, split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    counter=0

    test_save_dir = args.testsavedir
    
    os.makedirs(f'{test_save_dir}/{args.expname}/labels/',exist_ok=True)
    # predicted label
    os.makedirs(f'{test_save_dir}/{args.expname}/output/',exist_ok=True)
    # ground truth output
    os.makedirs(f'{test_save_dir}/{args.expname}/mask/',exist_ok=True) 
    # combined image of original image and predicted label
    os.makedirs(f'{test_save_dir}/{args.expname}/imgs/', exist_ok=True)
    # original images

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
            img,labels,index,x,y= batchdata
            data = img.to(device)       
            output = model(data)            
            predict = get_predictions(output)
            all_preds.append(predict)
            all_labels.append(labels)
            
            plt.imsave('{}/{}/imgs/{}.jpg'.format(test_save_dir, args.expname, index), img)

            for j in range (len(index)):       
                np.save('{}/{}/labels/{}.npy'.format(test_save_dir, args.expname, index[j]),predict[j].cpu().numpy())
                try:
                    plt.imsave('{}/{}/output/{}.jpg'.format(test_save_dir, args.expname, index[j]),255*labels[j].cpu().numpy())
                except:
                    pass
                
                # plt.imsave('{}/{}/imgs/{}.jpg'.format(test_save_dir, args.expname, index[j]), img)

                pred_img = predict[j].cpu().numpy()/3.0
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp,0,1)
                img_orig = np.array(img_orig)
                # img_orig_resized = np.resize(img_orig, (256,256))
                combine = np.hstack([img_orig,pred_img])
                plt.imsave('{}/{}/mask/{}.jpg'.format(test_save_dir, args.expname, index[j]),combine)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # miou = compute_mean_iou(all_preds.flatten(), all_labels.flatten(), info=True)
    end = time()
    print(f"Total Testing Time before mIoU calculation: {(end-start)/60}")
    start = time()
    miou, precision, recall, f1 = compute_mean_iou(all_preds.flatten(), all_labels.flatten(), info=True)
    end = time()
    print(f"Total calculation time taken: {(end-start) / 60} min")

    # os.rename('test',args.save)
