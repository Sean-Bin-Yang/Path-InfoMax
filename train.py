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


def parse_option():
    """command-line interface"""
    parser = argparse.ArgumentParser(description="PyTorch Implementation of PIM")
    parser.add_argument('--gpu', type=int, default=0, help='set GPU')
    """training params"""
    parser.add_argument('--save_model', default='./data/xxxx')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--n_layers', type=int, default=1)
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


print('------Loading dataset-----')

class TrainDataset(Dataset):
    
    def __init__(self):
        
        Path1, Path2, Path3, Pos1, Neg1, Neg2 = pkl.load(open('./data/train_data_format.pkl','rb'))

        self.Path1Mask = torch.FloatTensor(process.mask_enc(Path1))
        self.Path2Mask = torch.FloatTensor(process.mask_enc(Path2))
        self.Path3Mask = torch.FloatTensor(process.mask_enc(Path3))

        self.Pos1Mask = torch.FloatTensor(process.mask_enc(Pos1))
        self.Neg1Mask = torch.FloatTensor(process.mask_enc(Neg1))
        self.Neg2Mask = torch.FloatTensor(process.mask_enc(Neg2))

        self.Path1 = torch.LongTensor(Path1)
        self.Path2 = torch.LongTensor(Path2)
        self.Path3 = torch.LongTensor(Path3)

        self.Pos1 = torch.LongTensor(Pos1)
        self.Neg1 = torch.LongTensor(Neg1)
        self.Neg2 = torch.LongTensor(Neg2)

        self.len=Path1.shape[0]

    def __getitem__(self,index):
        
        return self.Path1[index], self.Path1Mask[index], self.Path2[index], self.Path2Mask[index],self.Path3[index], self.Path3Mask[index], self.Pos1[index], self.Pos1Mask[index], self.Neg1[index], self.Neg1Mask[index], self.Neg2[index], self.Neg2Mask[index]
    
    def __len__(self):
        
        return self.len


class ValDataset(Dataset):
    
    def __init__(self):
        
        Path1, Path2, Path3,Pos1, Neg1, Neg2 = pkl.load(open('./data/train_data_format.pkl','rb'))

        self.Path1Mask = torch.FloatTensor(process.mask_enc(Path1))
        self.Path2Mask = torch.FloatTensor(process.mask_enc(Path2))
        self.Path3Mask = torch.FloatTensor(process.mask_enc(Path3))

        self.Pos1Mask = torch.FloatTensor(process.mask_enc(Pos1))
        self.Neg1Mask = torch.FloatTensor(process.mask_enc(Neg1))
        self.Neg2Mask = torch.FloatTensor(process.mask_enc(Neg2))

        self.Path1 = torch.LongTensor(Path1)
        self.Path2 = torch.LongTensor(Path2)
        self.Path3 = torch.LongTensor(Path3)

        self.Pos1 = torch.LongTensor(Pos1)
        self.Neg1 = torch.LongTensor(Neg1)
        self.Neg2 = torch.LongTensor(Neg2)

        self.len=Path1.shape[0]

    def __getitem__(self,index):
        
        return self.Path1[index], self.Path1Mask[index], self.Path2[index], self.Path2Mask[index],self.Path3[index], self.Path3Mask[index], self.Pos1[index], self.Pos1Mask[index], self.Neg1[index], self.Neg1Mask[index], self.Neg2[index], self.Neg2Mask[index]
    
    def __len__(self):
        
        return self.len

#==========train epoch and val epoch======#
def train_epoch(model, label1, label2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    # forward
    optimizer.zero_grad()
    logits1, logits2= model(Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask)

    loss1 = process.LogitsLoss(logits1,label1)
    loss2 = process.LogitsLoss(logits2,label2)

    loss = args.alpha*loss1 + args.beta*loss2
    
    loss.backward()
    optimizer.step()

    return loss, loss1, loss2

def eval_epoch(model, label1, label2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    with torch.no_grad():

        # forward
        logits1,logits2 = model(Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask)
        
        loss1 = process.LogitsLoss(logits1,label1)
        loss2 = process.LogitsLoss(logits2,label2)

        loss = args.alpha*loss1 + args.beta*loss2


    return loss, loss1, loss2

if __name__ == '__main__':

    best = 1e9
    cnt_wait =0
    args = parse_option()
    lbl_11 = torch.ones(args.batch_size, args.seq_len * 1)
    lbl_12 = torch.zeros(args.batch_size, args.seq_len * 2)
    lbl1 = torch.cat((lbl_11, lbl_12), 1)

    lbl_21 = torch.ones(args.batch_size, 1 * 1)
    lbl_22 = torch.zeros(args.batch_size, 1 * 2)
    lbl2 = torch.cat((lbl_21, lbl_22), 1)
    
    save_model =args.save_model
    #========= Loading Dataset =========#
    TrainDataset = TrainDataset()
    ValDataset = ValDataset()
    DataLoaderTrain= DataLoader(dataset=TrainDataset, batch_size = args.batch_size, shuffle = True, drop_last = True)
    DataLoaderVal = DataLoader(dataset=ValDataset, batch_size = args.batch_size, shuffle = True, drop_last = True)


    model = PIM(
            n_in = args.hidden_size,
            n_h = args.hidden_size,
            input_size =args.hidden_size,
            hidden_size = args.hidden_size,
            n_layers = args.n_layers,
            dropout = args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    #optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),0.01, args.hidden_size, args.n_warmup_steps)

    print('===> Starting Training')
    for epoch in range(args.nb_epochs):

        print('[ Epoch', epoch, ']')

        for i, traindata in enumerate(DataLoaderTrain,0):
            Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask =traindata
            if torch.cuda.is_available():
                print('GPU available: Using CUDA')
                torch.cuda.set_device(args.gpu)
                model.cuda()
                Path1     = Path1.cuda()
                Path1Mask = Path1Mask.cuda()
                Path2     = Path2.cuda()
                Path2Mask = Path2Mask.cuda()
                Path3     = Path3.cuda()
                Path3Mask = Path3Mask.cuda()
                Pos1      = Pos1.cuda()
                Pos1Mask  = Pos1Mask.cuda()
                Neg1      = Neg1.cuda()
                Neg1Mask  = Neg1Mask.cuda()
                Neg2      = Neg2.cuda()
                Neg2Mask  = Neg2Mask.cuda()
                lbl1      = lbl1.cuda()
                lbl2      = lbl2.cuda()

                start = time.time()
                loss_train, loss_node, loss_node = train_epoch(model, lbl1, lbl2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args, optimizer)
                process.print_performances('Training', epoch, loss_train, start)
            else:
                print('CPU available: Using CPU')
                device = torch.device("cpu")
                start = time.time()
                loss_train, loss_t1, loss_t2 = train_epoch(model, lbl1, lbl2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args, optimizer)
                process.print_performances('Training', epoch, loss_train, start)

        for i, valdata in enumerate(DataLoaderVal,0):
            Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask =valdata
            if torch.cuda.is_available():
                print('GPU available: Using CUDA')
                torch.cuda.set_device(args.gpu)
                model.cuda()
                Path1     = Path1.cuda()
                Path1Mask = Path1Mask.cuda()
                Path2     = Path2.cuda()
                Path2Mask = Path2Mask.cuda()
                Path3     = Path3.cuda()
                Path3Mask = Path3Mask.cuda()
                Pos1      = Pos1.cuda()
                Pos1Mask  = Pos1Mask.cuda()
                Neg1      = Neg1.cuda()
                Neg1Mask  = Neg1Mask.cuda()
                Neg2      = Neg2.cuda()
                Neg2Mask  = Neg2Mask.cuda()
                lbl1      = lbl1.cuda()
                lbl2      = lbl2.cuda()

                start = time.time()
                loss_val, loss_node, loss_path = eval_epoch(model, lbl1, lbl2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args)
                process.print_performances('Validation', epoch, loss_val, start)

                if loss_val < best:
                    best = loss_val
                    cnt_wait =0
                    model_name = save_model + '.pkl'
                    torch.save(model.state_dict(),model_name)
                else:
                    cnt_wait +=1
                if cnt_wait == args.epoch_flag:
                    print('Early stopping!')
                    break
            else:
                device = torch.device("cpu")
                print('CPU available: Using CPU')
                start = time.time()
                loss_val, loss_node, loss_path = eval_epoch(model, lbl1, lbl2, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask, args)
                process.print_performances('Validation', epoch, loss_val, start)

                if loss_val < best:
                    cnt_wait=0
                    best = loss_val
                    model_name = save_model + '.pkl'
                    torch.save(model.state_dict(), model_name)
                else:
                    cnt_wait +=1
                if cnt_wait == args.epoch_flag:
                    print('Early stopping!')
                    break
    print('====>Finishing Training')
