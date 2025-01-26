from __future__ import print_function
import torch
import numpy as np
from model import Occ_Detector
#from data import BGC, worker_init_fn
from torch.utils.data import DataLoader
import pprint

CUDA = True

def test(model, test_loader):
    model.eval()
    
    
    if CUDA:
        model = model.cuda()
    
    with torch.no_grad():
        pred_occ_types = []
        actual_occ_types = []
        
        pred_amounts_list = []
        actual_amounts = []
        
        occ_frames_list = []
        
        for i, (occ_frames, occ_types, occ_amounts) in enumerate(test_loader):
            if CUDA:
                occ_frames = occ_frames.cuda()
                #(Num_chunks, c=1, h, w)
            pred_logits, pred_amounts = model(occ_frames)
            #(batch, num_classes)
            pred_classes = torch.argmax(pred_logits, dim=1)
            occ_frames_list.append(occ_frames.detach().cpu().numpy())
            pred_occ_types.append(pred_classes.detach().cpu().numpy())
            actual_occ_types.append(occ_types.numpy())
            pred_amounts_list.append(pred_amounts.detach().cpu().numpy())
            actual_amounts.append(occ_amounts.numpy())
            
            
            
            # if i%1000 == 0:
            #     print(f"Test iter {i}/{len(test_loader)} complete!", end='\r')
            
    cat_occ_frames_list = np.concatenate(occ_frames_list)
    pred_occ_types = np.concatenate(pred_occ_types)
    actual_occ_types = np.concatenate(actual_occ_types)
    pred_amounts_list = np.concatenate(pred_amounts_list)
    actual_amounts = np.concatenate(actual_amounts)
    
    
    
    #Save predictions and actuals in a pkl file
    import pickle
    with open('test_results.pkl', 'wb') as f:
        pickle.dump({'pred_occ_types': pred_occ_types, 
                     'actual_occ_types': actual_occ_types, 
                     "pred_amounts_list": pred_amounts_list,
                     'actual_amounts': actual_amounts,
                     'occ_frames_list': cat_occ_frames_list
                     }, f)
    
    acc = np.sum(pred_occ_types == actual_occ_types)/len(pred_occ_types)
    print(f"Accuracy of occlusion detector on synthetic occlusions = {acc:.4f}")
    
    mse = np.mean((pred_amounts_list - actual_amounts)**2)
    print(f"MSE of occlusion amount prediction on synthetic occlusions = {mse:.5f}")
    return acc, mse



def main(path=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    from model import Occ_Detector
    
    root = 'path/'
    from data_grew import GREW_frames, worker_init_fn
    test_dataset = GREW_frames(root = root, mode='test')
    
 
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=test_dataset.collate_fn, worker_init_fn=worker_init_fn)
    
    model = Occ_Detector()
    model = torch.nn.DataParallel(model)
    
    if CUDA:
        model = model.cuda()
    
    if path is not None:
        model.load_state_dict(torch.load(path))
        print(f"Saved model loaded from {path}")
    
    acc = test(model, test_loader)
    print(f"Evaluation complete")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Main parser.')
    parser.add_argument('--restore_path', type=str,
                        help="Saved model path")
    parser.add_argument('--cuda', action='store_true',
                        help="Use GPU")
    
    opt = parser.parse_args()
    path = opt.restore_path
    
    CUDA = opt.cuda
    
    
    main(path = path)
    
             