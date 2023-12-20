'''
create by XinZhu Zhou
2022/1/23
'''
import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import copy
import sys
from matplotlib.colors import BoundaryNorm, ListedColormap
# from torch_geometric import utils
import torch.optim as optim
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from evaluator import evaluate
# from evaluator2 import evaluate
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
# from Transformer import TransformerModel
# from Transformer_gru import TransformerModel
from RSSL import TransformerModel
# from GRU import GRUModel
from UTILS import Data
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.5)
# from kalmanmodel import kalman_Model
# from gatgru import GCN_GRU
# from UTILS import get_kalma
from RKNCell import RKNCell
from UTILS import DataGraph
from UTILS import get_trees
device = 'cuda'

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):

    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    k=torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(
                ((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio


class Stock_HyperODE:
    def __init__(self, data_path, market_name, tickers_fname, n_node,
                 parameters, steps=1, epochs=20,early_stop_count=0,early_stop_n=3, batch_size=None, flat=False, gpu=False, in_pro=False,seed=0):

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.early_stop_count=early_stop_count
        self.early_stop_count=early_stop_n
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        # self.eod_data, self.mask_data, self.gt_data, self.price_data = \
        #     load_EOD_data(data_path, market_name, self.tickers, steps)
        with open('../data/tree/'+market_name+'_33tree_File.txt', 'rb') as f:
            trees = pickle.load(f)
        self.get_tree=get_trees(trees)

        print('#tickers selected:', len(self.tickers))
        self.eod_data=np.load('../data/eod_data.npy')
        self.mask_data=np.load('../data/mask_data.npy')
        self.gt_data=np.load('../data/gt_data.npy')
        self.price_data=np.load('../data/price_data.npy')
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat

        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size
        self.Stock_num= len(self.tickers)
        self.in_dim=64
        self.emb_size=64
        self.days=4
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ),                np.expand_dims(
                   self.price_data[:, offset + seq_len], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               ), \
                 np.expand_dims(
                self.price_data[:, offset:offset + seq_len], axis=1
             )

    def train(self,cell_conf):
        global df
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        print('device name:', device)

        model=TransformerModel(9,self.parameters['unit'],3,4,cell_conf).cuda()

        seq_lengths = torch.tensor([len(self.tickers)] * 4)
        batch = torch.tensor([0] * len(self.tickers) * 4)

        index=0
        for p in model.parameters():
            index+=1
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        optimizer_hgat = optim.Adam(model.parameters(),
                                    lr=self.parameters['lr'],
                                    weight_decay=1e-3)

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        best_test_perf={'mse':np.inf,'mrrt':0.0,'btl':0.0}
        best_test_loss=np.inf
        early_stop_count = self.early_stop_count
        early_stop_n=self.early_stop_count
        pred_20_return=np.zeros(
            [self.epochs, len(self.tickers),752],
            dtype=float
        )
        true_20_return=np.zeros(
            [self.epochs, len(self.tickers),752],
            dtype=float
        )
        test_pred_close=np.zeros(
            [self.epochs, len(self.tickers),252],
            dtype=float
        )
        test_true_close=np.zeros(
            [self.epochs, len(self.tickers),252],
            dtype=float
        )
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            model.train()
            cur_train_pred = np.zeros(
                [len(self.tickers), self.valid_index - 4],
                dtype=float
            )
            cur_train_gt = np.zeros(
                [len(self.tickers), self.valid_index - 4],
                dtype=float
            )
            close_test_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            close_test_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )

            cur_train_mask = np.zeros(
                [len(self.tickers), self.valid_index - 4],
                dtype=float
            )

            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):

                emb_batch, mask_batch, price_batch,price_batch2, gt_batch,history_data = self.get_batch(
                    batch_offsets[j])
                optimizer_hgat.zero_grad()
                output= model.forward(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(self.get_tree.tree_data).to(device))
                cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(output,
                                                                                         torch.FloatTensor(
                                                                                             price_batch).to(device),
                                                                                         torch.FloatTensor(gt_batch).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             mask_batch).to(device),
                                                                                         self.parameters['alpha'],
                                                                                         self.batch_size)

                all_loss=cur_loss
                curr_rr_train = output.detach().cpu().numpy().reshape((len(self.tickers), 1))
                cur_train_pred[:, batch_offsets[j]-4] = \
                    copy.copy(curr_rr_train[:, 0])
                cur_train_gt[:, batch_offsets[j]-4] = \
                    copy.copy(price_batch2[:, 0])
                cur_train_mask[:, batch_offsets[j]-4] = \
                    copy.copy(mask_batch[:, 0])

                all_loss.backward()
                optimizer_hgat.step()


                tra_loss += all_loss.detach().cpu().item()
                tra_reg_loss += cur_reg_loss.detach().cpu().item()
                tra_rank_loss += cur_rank_loss.detach().cpu().item()
            pred_20_return[i,:,:]=cur_train_pred
            true_20_return[i,:,:]=cur_train_gt
            print('Train Loss:',
                  tra_loss / (self.test_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.test_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.test_index - self.parameters['seq'] - self.steps + 1))

            with torch.no_grad():

                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()
                for cur_offset in range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch, mask_batch, price_batch,price_batch2,  gt_batch,history_data = self.get_batch(
                        cur_offset)

                    output_val = model(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(self.get_tree.tree_data).to(device))
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_val,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    val_loss += (cur_loss) .detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index -
                                                  self.parameters['seq'] -
                                                  self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Valid MSE:',
                      val_loss / (self.test_index - self.valid_index),
                      val_reg_loss / (self.test_index - self.valid_index),
                      val_rank_loss / (self.test_index - self.valid_index))
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                          cur_valid_mask)
                print('\t Valid preformance:', cur_valid_perf)

                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )

                cur_test_hidden=np.zeros([len(self.tickers),self.trade_dates-self.test_index,9],dtype=float)
                cur_test_mu=np.zeros([len(self.tickers),self.trade_dates-self.test_index,64],dtype=float)
                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()
                for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                        self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    emb_batch, mask_batch, price_batch,price_batch2,  gt_batch,history_data = self.get_batch(cur_offset)


                    output_test = model(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(self.get_tree.tree_data).to(device))
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_test,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    output_test = output_test.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    close_test_pred[:,cur_offset-1004]=output_test[:, 0]
                    close_test_gt[:,cur_offset-1004]=price_batch2[:, 0]
                    test_loss += (cur_loss) .detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()

                    cur_test_pred[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index -
                                                 self.parameters['seq'] -
                                                 self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])


                test_pred_close[i, :, :] = close_test_pred
                test_true_close[i, :, :] = close_test_gt
                print('Test MSE:',
                      test_loss / (self.trade_dates - self.test_index),
                      test_reg_loss / (self.trade_dates - self.test_index),
                      test_rank_loss / (self.trade_dates - self.test_index))
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)


                print('\t Test performance:', 'sharpe5:', cur_test_perf['sharpe5'], 'ndcg_score_top5:',
                      cur_test_perf['ndcg_score_top5'],'mrrt',cur_test_perf['mrrt'],'Test loss',test_loss / (self.test_index - self.valid_index))
                np.set_printoptions(threshold=sys.maxsize)
                if test_loss / (self.test_index - self.valid_index) < best_test_loss:
                    best_test_loss = test_loss / (self.test_index - self.valid_index)
                    best_test_gt=copy.copy(cur_test_gt)
                    best_test_pred=copy.copy(cur_test_pred)
                    best_test_perf = copy.copy(cur_test_perf)
                    early_stop_count=0
                else:
                    early_stop_count+=1
                if early_stop_count>=early_stop_n:
                    print('early_stop_count',early_stop_count)
                    break


            print('\t epoch',i,' best performance:', 'sharpe5:', best_test_perf['sharpe5'], 'ndcg_score_top5:',
                  best_test_perf['ndcg_score_top5'],'mmrt',best_test_perf['mrrt'])
            print('\t epoch', i, 'bestsharpe5:', best_test_perf['sharpe5_max'])

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01-1')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.0001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-node', default=1026,help='n_node')
    parser.add_argument('-seed', default=44, help='seed')
    args = parser.parse_args()


    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    market_name="NASDAQ"

    args.t = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}

    RR_LSTM = Stock_HyperODE(
            data_path=args.p,
            market_name=market_name,
            tickers_fname=args.t,
            n_node=args.node,
            parameters=parameters,
            steps=1, epochs=100,
            early_stop_count=0 ,
            early_stop_n=500,
            batch_size=None, gpu=args.gpu,
            in_pro=args.inner_prod,
            seed=args.seed
    )

    latent_obs_dim = 128

    cell_conf = RKNCell.get_default_config()
    cell_conf.num_basis = 15
    cell_conf.bandwidth = 3
    cell_conf.never_invalid = True
    cell_conf.trans_net_hidden_units = []
    cell_conf.trans_net_hidden_activation = "tanh"
    cell_conf.trans_covar = 0.1
    cell_conf.finalize_modifying()

    pred_all = RR_LSTM.train(cell_conf)