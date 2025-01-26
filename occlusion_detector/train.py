from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn 

from data_grew import GREW_frames, worker_init_fn
from model import Occ_Detector
from test import test

# Importing 
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing
import argparse
from os.path import join as osp
import os
import wandb


if __name__ == '__main__':
    
    TEST_CHKPT = 1
    MAX_EPOCHS=50
    LABDA = 10      #Relative weight of regression loss over classification loss
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    
    
    parser = argparse.ArgumentParser(description='Main parser.')
    parser.add_argument('--run_name', type=str,
                        default='debug', help="Run Name")
    parser.add_argument('--num_w', type=int,
                        default=16, help="Number of workers")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--wandb', action='store_true',
                        help="Use wandb")
    parser.add_argument('--dataset', type=str, default='grew', help="Dataset to use")
    parser.add_argument('--classif_only', action='store_true', help="Use only classification loss")
    
    
    opt = parser.parse_args()
    
    assert opt.dataset in ['grew'], "Dataset not supported"
    
    
    use_wandb = opt.wandb
    
    criterion = nn.CrossEntropyLoss()    
    criterion_reg = nn.MSELoss()
    
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="occlusion_detector",
            name = opt.run_name
        )
        
    model = Occ_Detector()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
        model = nn.DataParallel(model)
        
    print(f"Model and loss function loaded!")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    batch_size = opt.batch_size
    
    
    if opt.dataset == 'grew':
        root = 'path/'
        train_dataset = GREW_frames(root = root, mode='train')       
        test_dataset = GREW_frames(root = root, mode='test')
        
    else: 
        raise NotImplementedError("Dataset not supported")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=opt.num_w, collate_fn=train_dataset.collate_fn, worker_init_fn=worker_init_fn, drop_last=True)  
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_w, collate_fn=test_dataset.collate_fn, worker_init_fn=worker_init_fn)
            
    print(f"Data loaded!")
    
    best_acc = 0
    save_dir = osp('./saves', opt.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Starting training loop...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        for i, (occ_frames, occ_types, occ_amounts) in enumerate(train_loader):     
            if torch.cuda.is_available():
                occ_frames = occ_frames.cuda()
                occ_types = occ_types.cuda()
                occ_amounts = occ_amounts.cuda()
            
            optimizer.zero_grad()
            
            logits, pred_amounts = model(occ_frames)
            loss_classif = criterion(logits, occ_types)
            
            if not opt.classif_only:
                loss_reg = criterion_reg(pred_amounts, occ_amounts)
                loss_reg *= LABDA
            else:
                loss_reg = torch.tensor(0)
                if torch.cuda.is_available():
                    loss_reg = loss_reg.cuda()
            
            loss = loss_classif + loss_reg
            
            if use_wandb:
                wandb.log({"loss": loss.item(), "loss_classif": loss_classif.item(), "loss_reg": loss_reg.item()})
            
            
            if i%100 == 0:
                print(f"Iter {i}/{len(train_loader)} complete! Loss classif = {loss_classif.item():.2f}; loss reg: {loss_reg.item():.5f}", end='\r')
            
            loss.backward()
            optimizer.step()
        
        
        
        if epoch%TEST_CHKPT == 0:
            torch.save(model.state_dict(), osp(save_dir, f'epoch-{epoch}.pth'))
            print(f"Model saved at epoch {epoch}!")
            model.eval()
            print(f"Evaluating...")
            acc, mse = test(model, test_loader)
            if use_wandb:
                wandb.log({"acc": acc, "mse": mse})
            print(f"Evaluation complete!")
            if acc>best_acc:
                best_acc = acc
                print(f"New best accuracy = {best_acc:.2f}")
                torch.save(model.state_dict(), osp(save_dir, f'best-model.pth'))
            model.train()
        
        print(f"Epoch {epoch} complete!")
            
            
    
    