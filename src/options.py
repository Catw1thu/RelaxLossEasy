import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    
    # training params
    parser.add_argument('--data', type=str, default='cifar10',
        help='dataset')
    
    parser.add_argument('--num_epochs', type=int, default=100,
        help='number of training epochs. (max epochs if early stopping is used)')
    
    parser.add_argument('--bs', type=int, default=256,
        help='batch size')
    
    parser.add_argument('--data_augment', type=bool, default=False,
        help='use data augmentation if set True')
    
    parser.add_argument('--early_stop', type=bool, default=False,
        help='use early stopping if set True')
    
    parser.add_argument('--relax_alpha', type=float, default=0.0,
        help='alpha param of relax_loss algorithm')
    
    parser.add_argument('--num_classes', type=int, default=10,
        help='number of classes in your dataset')
    
    # device params
    parser.add_argument('--device',  default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        help="To use cuda, set to a specific GPU ID.")
    
    parser.add_argument('--num_workers', type=int, default=4, 
        help="num. of workers for multithreading")
    
    args = parser.parse_args()
    return args