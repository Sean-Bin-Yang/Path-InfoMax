'''
This script handles the training process.
'''

import argparse
import pickle as pkl
import torch
import time
from models import PIM
from utils import process
from torch.utils.data import Dataset, DataLoader
import numpy as np



def parse_option():
    """command-line interface"""
    parser = argparse.ArgumentParser(description="PyTorch Implementation of PIM")
    parser.add_argument('--gpu', type=int, default=0, help='set GPU')
    """training params"""
    parser.add_argument('--save_model', default='./data/xxxx')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dim of node embedding (default: 512)')
    parser.add_argument('--epoch_flag', type=int, default=30, help=' early stopping (default: 20)')
    parser.add_argument('--nb_epochs', type=int, default=120,
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--l2_coef', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0) 
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()
    
    return args

class TestDataset(Dataset):
    
    def __init__(self):
        
        Path1,_,_,_,_,_=pkl.load(open('./data/train_data_format.pkl','rb'))

        self.Path1Mask = torch.FloatTensor(process.mask_enc(Path1))
        
        self.Path1 = torch.LongTensor(Path1)

        self.len=Path1.shape[0]

    def __getitem__(self,index):
        
        return self.Path1[index], self.Path1Mask[index]
    
    def __len__(self):
        
        return self.len

#==========test epoch======#
def test_epoch(model, Path1, Path1Mask):
    ''' Epoch operation in evaluation phase '''

    with torch.no_grad():

        # forward
        PathEmbed = model.embed(Path1, Path1Mask)

    return PathEmbed


if __name__ == '__main__':

    args = parse_option()
    
    #========= Loading Dataset =========#
    TestDataset = TestDataset()
    DataLoaderTest= DataLoader(dataset=TestDataset, batch_size = 1, shuffle = False, drop_last = False)


    #===========PIM model=======#
    model = PIM(
                n_in = args.hidden_size,
                n_h = args.hidden_size,
                input_size =args.hidden_size,
                hidden_size = args.hidden_size,
                n_layers = args.n_layers,
                dropout = args.dropout)


    print('Model Loading')
    model.load_state_dict(torch.load('./data/xxxx.pkl'))
    
    print('Starting Testing')
    embedding =[]
    #=============start train=========#

    for i, testdata in enumerate(DataLoaderTest,0):
        print(i)
        Path1, Path1Mask = testdata
        if torch.cuda.is_available():
            print('GPU available: Using CUDA')
            torch.cuda.set_device(args.gpu)
            model.cuda()
            Path1 = Path1.cuda()
            Path1Mask =Path1Mask.cuda()
          
            start = time.time()
            PathEmbed = test_epoch(model, Path1, Path1Mask)
            embedding.append(PathEmbed.cpu().numpy())
        else:
            print('CPU available: Using CPU')
            device = torch.device("cpu")
            start = time.time()
            PathEmbed = test_epoch(model, Path1, Path1Mask)
            embedding.append(PathEmbed.numpy())
    pkl.dump(np.array(embedding), open('path_embed_xxx.pkl', 'wb'))


