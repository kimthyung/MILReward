# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

from aeon.datasets import load_classification

from utils import *
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from custom_dataloader import CustomDatasetLoader

import random
from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings

from models.timemil import TimeMIL

# Suppress all warnings
warnings.filterwarnings("ignore")

def train(trainloader, milnet, criterion, optimizer, epoch, args):
    milnet.train()
    total_loss = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.cuda()
        bag_label = label.cuda()
        
        # window-based random masking
        if args.dropout_patch > 0:
            select_window_idx = random.sample(range(10), int(args.dropout_patch * 10))
            interval = int(len(bag_feats) // 10)
            
            for idx in select_window_idx:
                bag_feats[:, idx*interval:idx*interval+interval, :] = torch.randn(1).cuda()
        
        optimizer.zero_grad()
        
        if epoch < args.epoch_des:
            bag_prediction = milnet(bag_feats, warmup=True)
        else:
            bag_prediction = milnet(bag_feats, warmup=False)
        
        bag_loss = criterion(bag_prediction, bag_label)
        loss = bag_loss
        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % 
                        (batch_id, len(trainloader), bag_loss.item(), loss.item()))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), max_norm=1.0)  # 그래디언트 클리핑
        optimizer.step()
        
        total_loss = total_loss + bag_loss
    
    return total_loss / len(trainloader)

def test(testloader, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions_scores = []
    
    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            
            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction, bag_label)
            
            loss = bag_loss
            total_loss = total_loss + loss.item()
            
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % 
                           (batch_id, len(testloader), loss.item()))
            
            test_labels.extend([label.cpu().numpy()])
            test_predictions_scores.extend([torch.sigmoid(bag_prediction).cpu().numpy()])
    
    test_labels = np.vstack(test_labels)
    test_scores = np.vstack(test_predictions_scores)
    
    roc_auc_ovo_marco, roc_auc_ovo_micro, roc_auc_ovr_marco, roc_auc_ovr_micro = 0, 0, 0, 0

    if args.num_classes == 1:
        # Binary classification
        labels_for_metrics = test_labels.ravel()
        scores_for_roc = test_scores.ravel()
        predictions_for_metrics = (scores_for_roc > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(labels_for_metrics, scores_for_roc)
            roc_auc_ovo_marco, roc_auc_ovo_micro, roc_auc_ovr_marco, roc_auc_ovr_micro = auc, auc, auc, auc
        except ValueError:
            pass # Keep zeros if only one class present
        
        test_predictions = predictions_for_metrics
        test_labels_for_metrics = labels_for_metrics

    else:
        # Multi-class classification
        test_predictions = np.argmax(test_scores, axis=1)
        test_labels_for_metrics = np.argmax(test_labels, axis=1)
        
        try:
            if test_scores.shape[1] == 2:
                # Binary as 2-class
                auc = roc_auc_score(test_labels_for_metrics, test_scores[:, 1])
                roc_auc_ovo_marco, roc_auc_ovr_marco = auc, auc
                roc_auc_ovo_micro = roc_auc_score(test_labels, test_scores, average='micro', multi_class='ovo')
                roc_auc_ovr_micro = roc_auc_score(test_labels, test_scores, average='micro', multi_class='ovr')
            else:
                # True multi-class
                roc_auc_ovo_marco = roc_auc_score(test_labels, test_scores, average='macro', multi_class='ovo')
                roc_auc_ovo_micro = roc_auc_score(test_labels, test_scores, average='micro', multi_class='ovo')
                roc_auc_ovr_marco = roc_auc_score(test_labels, test_scores, average='macro', multi_class='ovr')
                roc_auc_ovr_micro = roc_auc_score(test_labels, test_scores, average='micro', multi_class='ovr')
        except ValueError:
            pass # Keep zeros if only one class present

    avg_score = accuracy_score(test_labels_for_metrics, test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels_for_metrics, test_predictions)
    f1_marco = f1_score(test_labels_for_metrics, test_predictions, average='macro')
    f1_micro = f1_score(test_labels_for_metrics, test_predictions, average='micro')
    p_marco = precision_score(test_labels_for_metrics, test_predictions, average='macro')
    p_micro = precision_score(test_labels_for_metrics, test_predictions, average='micro')
    r_marco = recall_score(test_labels_for_metrics, test_predictions, average='macro')
    r_micro = recall_score(test_labels_for_metrics, test_predictions, average='micro')
    
    results = [avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro, 
               roc_auc_ovo_marco, roc_auc_ovo_micro, roc_auc_ovr_marco, roc_auc_ovr_micro]
    
    return total_loss / len(testloader), results

def main():
    parser = argparse.ArgumentParser(description='TimeMIL with Custom Dataset')
    
    # 안정성 중심 설정
    parser.add_argument('--lr', default=3e-4, type=float, help='Initial learning rate [3e-4]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--dropout_patch', default=0.4, type=float, help='Patch dropout rate [0.4]')
    parser.add_argument('--dropout_node', default=0.15, type=float, help='Bag classifier dropout rate [0.15]')
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [100]')  # 100으로 복원
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')  # 이 줄 추가!
    
    parser.add_argument('--dataset', default="CustomDataset", type=str, help='dataset name')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [1]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=7, type=int, help='Dimension of the feature size [7]')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer [adamw]')
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
    parser.add_argument('--data_dir', default='Custom_Dataset_20_ds', type=str, help='custom dataset directory')
    parser.add_argument('--test_size', default=0.2, type=float, help='test set ratio')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir + 'CustomDataset'
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)
    
    # Set up logging
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)
    
    # Save hyperparams
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Load custom dataset
    print("Loading CustomDataset...")
    trainset = CustomDatasetLoader(data_dir=args.data_dir, split='train', seed=args.seed, max_samples=4000)
    validset = CustomDatasetLoader(data_dir=args.data_dir, split='valid', seed=args.seed, max_samples=4000)
    testset = CustomDatasetLoader(data_dir=args.data_dir, split='test', seed=args.seed, max_samples=4000)

    seq_len, num_classes, L_in = trainset.get_properties()

    print(f'max length {seq_len}')
    args.feats_size = L_in
    args.num_classes = num_classes
    print(f'num class:{args.num_classes}')
    
    # Define MIL network
    milnet = TimeMIL(args.feats_size, mDim=args.embed, n_classes=num_classes, 
                     dropout=args.dropout_node, max_seq_len=seq_len).cuda()
    
    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer)
    elif args.optimizer == 'adamp':
        optimizer = AdamP(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer)
    
    # Data loaders
    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, 
                       num_workers=args.num_workers, drop_last=False, pin_memory=True)
    validloader = DataLoader(validset, args.batchsize, shuffle=False,  # 128 → args.batchsize
                      num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, args.batchsize, shuffle=False,    # 128 → args.batchsize
                      num_workers=args.num_workers, drop_last=False, pin_memory=True)
    
    # Training loop (얼리 스토핑 제거)
    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    os.makedirs(join(args.save_dir, 'lesion'), exist_ok=True)
    results_best = None
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch, args)
        valid_loss_bag, valid_results = test(validloader, milnet, criterion, args)
        test_loss_bag, test_results = test(testloader, milnet, criterion, args)
        
        [valid_avg_score, valid_balanced_avg_score, valid_f1_marco, valid_f1_micro, 
         valid_p_marco, valid_p_micro, valid_r_marco, valid_r_micro, 
         valid_roc_auc_ovo_marco, valid_roc_auc_ovo_micro, valid_roc_auc_ovr_marco, valid_roc_auc_ovr_micro] = valid_results
        
        [test_avg_score, test_balanced_avg_score, test_f1_marco, test_f1_micro, 
         test_p_marco, test_p_micro, test_r_marco, test_r_micro, 
         test_roc_auc_ovo_marco, test_roc_auc_ovo_micro, test_roc_auc_ovr_marco, test_roc_auc_ovr_micro] = test_results
        
        logger.info('\r Epoch [%d/%d] train loss: %.4f valid loss: %.4f test loss: %.4f, valid acc: %.4f, test acc: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, valid_loss_bag, test_loss_bag, valid_avg_score, test_avg_score))
        
        # Validation 성능으로 모델 저장 (얼리 스토핑 없이)
        current_score = valid_avg_score
        if current_score >= best_score:
            results_best = test_results
            best_score = current_score
            print(f"New best validation score: {current_score}")
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
            logger.info('Best model saved at: ' + save_name)
    
    # Final results
    [avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro, 
     roc_auc_ovo_marco, roc_auc_ovo_micro, roc_auc_ovr_marco, roc_auc_ovr_micro] = results_best
    
    logger.info('Final Results - accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f, f1 mirco: %.4f, AUROC macro: %.4f' % 
               (avg_score, balanced_avg_score, f1_marco, f1_micro, roc_auc_ovr_marco))
    
    print("Training completed!")

if __name__ == "__main__":
    main() 