#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""

# python test.py --model densenet --load best_model.pkl --bs 4 --dataset ../../../../../../../../storage/ice1/shared/bmed6780/mip_group_4/'openEDS Dataset'/openEDS/openEDS/ --save ../../../../../scratch/test

import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions, compute_mean_iou
#%%

if __name__ == '__main__':
    
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
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = args.dataset, split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    counter=0
    
    os.makedirs('../../../../../scratch/test/labels/',exist_ok=True)
    os.makedirs('../../../../../scratch/test/output/',exist_ok=True)
    os.makedirs('../../../../../scratch/test/mask/',exist_ok=True)
    os.makedirs('../../../../../scratch/test/imgs/', exist_ok=True)

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

            for j in range (len(index)):       
                np.save('../../../../../scratch/test/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
                try:
                    plt.imsave('../../../../../scratch/test/output/{}.jpg'.format(index[j]),255*labels[j].cpu().numpy())
                except:
                    pass
                
                plt.imsave('../../../../../scratch/test/imgs/{}.jpg'.format(index[j]), img)

                pred_img = predict[j].cpu().numpy()/3.0
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp,0,1)
                img_orig = np.array(img_orig)
                combine = np.hstack([img_orig,pred_img])
                plt.imsave('../../../../../scratch/test/mask/{}.jpg'.format(index[j]),combine)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # miou = compute_mean_iou(all_preds.flatten(), all_labels.flatten(), info=True)
    miou, precision, recall, f1 = compute_mean_iou(all_preds.flatten(), all_labels.flatten(), info=True)

    # os.rename('test',args.save)
