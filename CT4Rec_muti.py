# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:50:07 2022

@author: asus
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import sys
import os
import logging
from time import strftime
from time import localtime
import multiprocessing as mp
from RankingMetrics import *
import reckit

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def l2_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=-1, keepdim=True))

def clip_by_norm(x, clip_norm, dim=-1):
    norm = torch.square(x).sum(dim, keepdim=True)
    output = torch.where(norm > clip_norm ** 2, x * clip_norm / (norm + 1e-6), x)
    return output

class CT4Rec(nn.Module):
    def __init__(self, user_count,item_count,train_data, test_data, embedding_size, eps, min_temp, max_temp, 
                 noise_type, learning_rate, reg_rate, epoch, batch_size):
        super(CT4Rec, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.train_data = train_data
        self.test_data = test_data
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.clip_norm = 1
        
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.noise_type = noise_type
        self.eps = eps
        
        self.train_ui_dict = self.get_train_uit_dict(self.train_data)
        self.test_ui_dict = self.get_test_uit_dict(self.test_data)
        
        self.user_emb_w = nn.Embedding(user_count, embedding_size)
        self.item_emb_w = nn.Embedding(item_count, embedding_size)
        self.bias = nn.Embedding(item_count, 1)
        self.t = nn.Parameter(torch.empty(embedding_size))

        nn.init.normal_(self.t, 0, 0.02)
        nn.init.constant_(self.bias.weight, 0) 
        nn.init.constant_(self.user_emb_w.weight, 0)
        nn.init.normal_(self.item_emb_w.weight, 0, 0.02)
        
        self.item_emb_w.weight.data = clip_by_norm(self.item_emb_w.weight.data, self.clip_norm)
        self.t.data = clip_by_norm(self.t, self.clip_norm)
        self.present_train_size = 0
    
    def forward(self, user, last_item, pos_item, neg_item):
        user_id = self.user_emb_w(user)       
        last_item_id = self.item_emb_w(last_item)
        pos_item_id = self.item_emb_w(pos_item)
        neg_item_id = self.item_emb_w(neg_item)
        pos_bias = self.bias(pos_item)
        neg_bias = self.bias(neg_item)
        
        t = self.t.expand_as(user_id)                 
        seq_output = user_id + last_item_id + t
        pos_score = pos_bias - l2_distance(seq_output, pos_item_id)
        neg_score = neg_bias - l2_distance(seq_output, neg_item_id)
        
        reg = torch.norm(user_id,p=2).sum() + torch.norm(last_item_id,p=2).sum() + \
              torch.norm(pos_item_id,p=2).sum() + torch.norm(neg_item_id,p=2).sum() + \
              torch.norm(pos_bias,p=2).sum() + torch.norm(neg_bias,p=2).sum() + \
              torch.norm(t).sum()
        
        all_users_1, all_items_1 = self.noise_compute(self.noise_type[0])
        all_users_2, all_items_2 = self.noise_compute(self.noise_type[1])
        
        users_emb_1 = nn.functional.normalize(all_users_1, dim=1)
        users_emb_2 = nn.functional.normalize(all_users_2, dim=1)
        items_emb_1 = nn.functional.normalize(all_items_1, dim=1)
        items_emb_2 = nn.functional.normalize(all_items_2, dim=1)
        last_item_id = nn.functional.normalize(last_item_id, dim=1)
        t = nn.functional.normalize(t, dim=1)
        
        user_emb_1 = users_emb_1[user]
        user_emb_2 = users_emb_2[user]
        
        pos_item_emb_1 = items_emb_1[pos_item]
        pos_item_emb_2 = items_emb_2[pos_item]
        
        last_item_emb_1 = items_emb_1[last_item]
        last_item_emb_2 = items_emb_2[last_item]

        item_1 = user_id + last_item_emb_1 + t
        item_2 = user_id + last_item_emb_2 + t
        
        user_1 = pos_item_emb_1 - last_item_emb_1 - t
        user_2 = pos_item_emb_2 - last_item_emb_2 - t

        '''
        pos_user_pair_sim = torch.sum(torch.mul(user_1, user_2),dim=1)
        temperature = self.dy_temperature(pos_user_pair_sim)
        ground_user = torch.exp(torch.sum(torch.mul(user_1, user_emb_2),dim=1) / temperature)
        
        pos_ratings_user = torch.exp(pos_user_pair_sim / temperature) 
        tot_ratings_user = torch.matmul(user_1, torch.transpose(users_emb_2, 0, 1))
        temperature = self.dy_temperature(tot_ratings_user)
        tot_ratings_user = torch.sum(torch.exp(tot_ratings_user / temperature),dim=1)
        ssl_loss_user = - torch.sum(torch.log(pos_ratings_user / ((tot_ratings_user-ground_user).clamp_(min=0) + pos_ratings_user)))
        
        
        pos_item_pair_sim = torch.sum(torch.mul(item_1, item_2),dim=1)
        temperature = self.dy_temperature(pos_item_pair_sim)
        ground_item = torch.exp(torch.sum(torch.mul(item_1, pos_item_emb_2),dim=1) / temperature)
        
        pos_ratings_item = torch.exp(pos_item_pair_sim / temperature)
        tot_ratings_item = torch.matmul(item_1, torch.transpose(items_emb_2, 0, 1))  
        temperature = self.dy_temperature(tot_ratings_item)
        tot_ratings_item = torch.sum(torch.exp(tot_ratings_item / temperature),dim=1) 
        ssl_loss_item = - torch.sum(torch.log(pos_ratings_item / ((tot_ratings_item-ground_item).clamp_(min=0) + pos_ratings_item)))
        '''
        
        pos_user_pair_sim = torch.sum(torch.mul(user_1, user_2),dim=1)
        temperature = self.dy_temperature(pos_user_pair_sim)
        ground_user = torch.exp(torch.sum(torch.mul(user_1, user_emb_2),dim=1) / temperature)
        
        pos_ratings_user = torch.exp(pos_user_pair_sim / temperature) 
        tot_ratings_user = torch.matmul(user_1, torch.transpose(users_emb_2, 0, 1))
        tot_ratings_user = torch.sum(torch.exp(tot_ratings_user / temperature.unsqueeze(dim=1)),dim=1)
        ssl_loss_user = - torch.sum(torch.log(pos_ratings_user / ((tot_ratings_user-ground_user).clamp_(min=0) + pos_ratings_user)))
        
        
        pos_item_pair_sim = torch.sum(torch.mul(item_1, item_2),dim=1)
        temperature = self.dy_temperature(pos_user_pair_sim)
        ground_item = torch.exp(torch.sum(torch.mul(item_1, pos_item_emb_2),dim=1) / temperature)
        
        pos_ratings_item = torch.exp(pos_item_pair_sim / temperature)
        tot_ratings_item = torch.matmul(item_1, torch.transpose(items_emb_2, 0, 1))     
        tot_ratings_item = torch.sum(torch.exp(tot_ratings_item / temperature.unsqueeze(dim=1)),dim=1) 
        ssl_loss_item = - torch.sum(torch.log(pos_ratings_item / ((tot_ratings_item-ground_item).clamp_(min=0) + pos_ratings_item)))
        
        return pos_score - neg_score, reg, ssl_loss_user + ssl_loss_item
    
    def dy_temperature(self, sim):
        temperature = self.min_temp + 0.5 * (self.max_temp - self.min_temp) * (1 + torch.cos(np.pi * (1 + sim)))
        return temperature
    
    def noise_compute(self, noise_type):
        users_emb = self.user_emb_w.weight
        item_emb = self.item_emb_w.weight
        
        random_noise = 0
        all_emb = torch.cat([users_emb, item_emb])
        
        if noise_type == 'uniform_noise':
            random_noise = torch.rand(all_emb.shape)
        else:
            random_noise = torch.randn(size=all_emb.shape)
        
        random_noise = random_noise.cuda()
        all_emb += torch.mul(torch.sign(all_emb), nn.functional.normalize(random_noise)) * self.eps
        users, items = torch.split(all_emb, [self.user_count, self.item_count])
        return users, items
        
    
    def predict(self, user, last_item):
        user =  torch.from_numpy(np.array(user)).long().to('cuda')
        last_item = torch.from_numpy(np.array(last_item)).long().to('cuda')
        
        user_id = self.user_emb_w(user)       
        last_item_id = self.item_emb_w(last_item)
        item_emb = self.item_emb_w.weight.data
        bias = self.bias.weight.data
                
        seq_output = user_id + last_item_id + self.t
        pred = - l2_distance(seq_output, item_emb)   
        pred = pred + bias
        return pred
    
    def clip_by_norm_op(self, last_item, pos_item, neg_item):
        self.item_emb_w(last_item).data = clip_by_norm(self.item_emb_w(last_item).data, self.clip_norm)
        self.item_emb_w(pos_item).data = clip_by_norm(self.item_emb_w(pos_item).data, self.clip_norm)
        self.item_emb_w(neg_item).data = clip_by_norm(self.item_emb_w(neg_item).data, self.clip_norm)
    
    def get_train_uit_dict(self,data):
        data_list = {}
        for entity in data:
            if entity[0] not in data_list.keys():
                data_list[entity[0]] = set()
            data_list[entity[0]].add(entity[1])
            data_list[entity[0]].add(entity[2])
        return data_list
    
    def get_test_uit_dict(self,data):
        data_list = {}
        for entity in data:
            if entity[0] not in data_list.keys():
                data_list[entity[0]] = []
            data_list[entity[0]].append(entity[1])#last_item
            data_list[entity[0]].append(entity[2])#next_item
        return data_list
    
    def get_train_batch(self):
        len_data = len(self.train_data)
        if self.present_train_size+self.batch_size>len_data - 1:
            res = self.train_data[self.present_train_size:len_data] + \
                  self.train_data[0:self.present_train_size+self.batch_size-len_data]
        else:
            res = self.train_data[self.present_train_size:self.present_train_size+self.batch_size]
        self.present_train_size += self.batch_size
        self.present_train_size %= len_data
        return res
        
    def get_feed_dict(self,data):
        user_list = []
        last_item_list = []
        pos_item_list = []
        neg_item_list = []
        for item in data:
            nt = random.sample(range(self.item_count), 1)[0]
            while nt in [item[2]]:
                nt = random.sample(range(self.item_count), 1)[0]
            user_list.append(item[0])
            pos_item_list.append(item[2])
            last_item_list.append(item[1])
            neg_item_list.append(nt)
        return  user_list,last_item_list,pos_item_list,neg_item_list
    
    def save_model(self, model, epoch):
        code_name = os.path.basename(__file__).split('.')[0]
        log_path = "model/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({'model': model.state_dict()}, '{}_{}.pth'.format(log_path + code_name, epoch))
    
class evaluate:
    def __init__(self, test_ui_dic, user_emb, item_emb, t, bias):
        self.test_ui_dict = test_ui_dic
        self.embedding_user = user_emb
        self.embedding_item = item_emb
        self.bias = bias
        self.t = t
    
    def predict(self, user, last_item):
        user =  torch.from_numpy(np.array(user)).long().to('cuda')
        last_item = torch.from_numpy(np.array(last_item)).long().to('cuda')
        
        user_id = self.embedding_user[user]  
        last_item_id = self.embedding_item[last_item]
        item_emb = self.embedding_item.data
        bias = self.bias.data
                
        seq_output = user_id + last_item_id + self.t
        pred = - l2_distance(seq_output, item_emb)   
        pred = pred + bias
        return pred
    
    def evaluator(self, user):
        last_item_ids = [self.test_ui_dict[user][0]] 
        rank_list = []
                    
        pred = self.predict(user, last_item_ids)
        pred = pred.cpu().detach().numpy().squeeze()
                    
        rank_list = reckit.arg_top_k(pred,51)
        ground_list = []
        for i in rank_list:
            if i == last_item_ids:
                continue
            ground_list.append(i)
                    
        test_list = [self.test_ui_dict[user][1]]
        
        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, rank_list[:10], test_list)    
        p_20, r_20, ndcg_20 = precision_recall_ndcg_at_k(20, rank_list[:20], test_list)    
        p_30, r_30, ndcg_30 = precision_recall_ndcg_at_k(30, rank_list[:30], test_list)
        p_40, r_40, ndcg_40 = precision_recall_ndcg_at_k(40, rank_list[:40], test_list)
        p_50, r_50, ndcg_50 = precision_recall_ndcg_at_k(50, rank_list[:50], test_list)
        return [p_10, p_20, p_30, p_40, p_50, r_10, r_20, r_30, r_40, r_50, ndcg_10, ndcg_20, ndcg_30, ndcg_40, ndcg_50]


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)  
    random.seed(seed)
      
def train(user_count, item_count, train_data, test_data, embedding_size, eps, min_temp, max_temp, \
          noise_type, learning_rate, reg_rate, epoch, batch_size, ssl_reg):   
    device = torch.device('cuda')
    setup_seed(2020)
    
    history_p_at_10 = []
    history_p_at_20 = []
    history_p_at_30 = []
    history_p_at_40 = []
    history_p_at_50 = []
    history_r_at_10 = []
    history_r_at_20 = []
    history_r_at_30 = []
    history_r_at_40 = []
    history_r_at_50 = []
    history_ndcg_at_10 = []
    history_ndcg_at_20 = []
    history_ndcg_at_30 = []
    history_ndcg_at_40 = []
    history_ndcg_at_50 = []
    
    batch_total = int(len(train_data)/batch_size)
    ct4Rec = CT4Rec(user_count, item_count,train_data, test_data, embedding_size, eps, min_temp, max_temp,
                          noise_type, learning_rate, reg_rate, epoch, batch_size).to(device)
    optimizer = optim.Adam(ct4Rec.parameters(), lr = learning_rate)
    for epoch in range(epoch):
        ct4Rec_loss = 0
        torch.cuda.empty_cache()
        ct4Rec.train()    
        for k in range(1,batch_total+1):         
            user_list,last_item_list,pos_item_list,neg_item_list = ct4Rec.get_feed_dict(ct4Rec.get_train_batch())
            user = torch.from_numpy(np.array(user_list)).long().to(device)
            last_item = torch.from_numpy(np.array(last_item_list)).long().to(device)
            pos_item = torch.from_numpy(np.array(pos_item_list)).long().to(device)
            neg_item = torch.from_numpy(np.array(neg_item_list)).long().to(device)
            
            pred, reg, ssl_loss = ct4Rec(user, last_item, pos_item, neg_item)
            batch_loss = -torch.log(torch.sigmoid(pred)).sum() + reg_rate * reg + ssl_reg * ssl_loss
            ct4Rec_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            ct4Rec.clip_by_norm_op(last_item, pos_item, neg_item)

        if epoch == 99:
            trans_rec.save_model(trans_rec,epoch)
        if epoch == 299:
            trans_rec.save_model(trans_rec,epoch)
        if epoch == 499:
            trans_rec.save_model(trans_rec,epoch)
        
        
        if (epoch+1) % args.verbose == 0:
            ct4Rec.eval()
            with torch.no_grad():
                evaluation = evaluate(ct4Rec.test_ui_dict, ct4Rec.user_emb_w.weight.detach(), \
                                      ct4Rec.item_emb_w.weight.detach(), ct4Rec.t.detach(), ct4Rec.bias.weight.detach())                   
                    
                user_id = [key for key in ct4Rec.test_ui_dict.keys()]
                pool = mp.Pool(processes = 5)
                res = pool.map(evaluation.evaluator, user_id)
                pool.close() 
                pool.join()
                res = np.array(res)
                res = np.mean(res,axis = 0)
                
                history_p_at_10.append(res[0])
                history_p_at_20.append(res[1])
                history_p_at_30.append(res[2])
                history_p_at_40.append(res[3])
                history_p_at_50.append(res[4])
                history_r_at_10.append(res[5])
                history_r_at_20.append(res[6])
                history_r_at_30.append(res[7])
                history_r_at_40.append(res[8])
                history_r_at_50.append(res[9])
                history_ndcg_at_10.append(res[10])
                history_ndcg_at_20.append(res[11])
                history_ndcg_at_30.append(res[12])
                history_ndcg_at_40.append(res[13])
                history_ndcg_at_50.append(res[14])
            
                print(
                    " %04d Loss: %.2f \t pre10: %.5f  rec10: %.5f  ndcg10: %.5f  pre20: %.5f  rec20: %.5f ndcg20: %.5f  pre30:  %.5f  rec30:  %.5f ndcg30:  %.5f  pre40:  %.5f  rec40:  %.5f ndcg40:  %.5f pre50:  %.5f  rec50:  %.5f ndcg50:  %.5f" % \
                        (epoch+1, ct4Rec_loss, res[0], res[5], res[10], res[1], res[6], res[11], res[2], res[7], res[12], res[3], res[8], res[13], res[4], res[9], res[14]))
                logging.info(
                    " %04d Loss: %.2f \t pre10: %.5f  rec10: %.5f  ndcg10: %.5f  pre20: %.5f  rec20: %.5f ndcg20: %.5f  pre30:  %.5f  rec30:  %.5f ndcg30:  %.5f  pre40:  %.5f  rec40:  %.5f ndcg40:  %.5f pre50:  %.5f  rec50:  %.5f ndcg50:  %.5f" % \
                        (epoch+1, ct4Rec_loss, res[0], res[5], res[10], res[1], res[6], res[11], res[2], res[7], res[12], res[3], res[8], res[13], res[4], res[9], res[14]))
                    
    best_pre10_index = np.argmax(history_p_at_10)
    best_pre10 = history_p_at_10[best_pre10_index]
    best_pre20 = history_p_at_20[best_pre10_index]
    best_pre30 = history_p_at_30[best_pre10_index]
    best_pre40 = history_p_at_40[best_pre10_index]
    best_pre50 = history_p_at_50[best_pre10_index]
    best_rec10 = history_r_at_10[best_pre10_index]
    best_rec20 = history_r_at_20[best_pre10_index]
    best_rec30 = history_r_at_30[best_pre10_index]
    best_rec40 = history_r_at_40[best_pre10_index]
    best_rec50 = history_r_at_50[best_pre10_index]
    best_ndcg10 = history_ndcg_at_10[best_pre10_index]
    best_ndcg20 = history_ndcg_at_20[best_pre10_index]
    best_ndcg30 = history_ndcg_at_30[best_pre10_index]
    best_ndcg40 = history_ndcg_at_40[best_pre10_index]
    best_ndcg50 = history_ndcg_at_50[best_pre10_index]
    
    print(
        "Best Epochs: pre10: %.5f  rec10: %.5f  ndcg10: %.5f  pre20: %.5f  rec20: %.5f ndcg20: %.5f  pre30:  %.5f  rec30:  %.5f ndcg30:  %.5f  pre40:  %.5f  rec40:  %.5f ndcg40:  %.5f pre50:  %.5f  rec50:  %.5f ndcg50:  %.5f" % \
        (best_pre10, best_rec10, best_ndcg10, best_pre20, best_rec20, best_ndcg20, best_pre30, best_rec30, best_ndcg30, best_pre40, best_rec40, best_ndcg40, best_pre50, best_rec50, best_ndcg50))
    logging.info(
        "Best Epochs: pre10: %.5f  rec10: %.5f  ndcg10: %.5f  pre20: %.5f  rec20: %.5f ndcg20: %.5f  pre30:  %.5f  rec30:  %.5f ndcg30:  %.5f  pre40:  %.5f  rec40:  %.5f ndcg40:  %.5f pre50:  %.5f  rec50:  %.5f ndcg50:  %.5f" % \
        (best_pre10, best_rec10, best_ndcg10, best_pre20, best_rec20, best_ndcg20, best_pre30, best_rec30, best_ndcg30, best_pre40, best_rec40, best_ndcg40, best_pre50, best_rec50, best_ndcg50))    
    out_max(best_pre10, best_rec10, best_ndcg10, best_pre20, best_rec20, best_ndcg20, best_pre30, best_rec30, best_ndcg30, best_pre40, best_rec40, best_ndcg40, best_pre50, best_rec50, best_ndcg50)
       
def out_max(pre10, rec10, ndcg10, pre20, rec20, ndcg20, pre30, rec30, ndcg30, pre40, rec40, ndcg40, pre50, rec50, ndcg50):
    log_path_ = "log/%s/" % ("CT4Rec")
    if not os.path.exists(log_path_):
        os.makedirs(log_path_)
    csv_path =  log_path_ + "%s.csv" % (args.dataset)
    log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
            args.embedding_size, args.learning_rate, args.reg_rate, args.ssl_reg, args.min_temp, args.max_temp, args.eps,
            args.noise_type[0], args.noise_type[1],
            pre10, rec10, ndcg10,
            pre20, rec20, ndcg20,
            pre30, rec30, ndcg30,
            pre40, rec40, ndcg40,
            pre50, rec50, ndcg50)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("embedding_size,learning_rate,reg_rate,ssl_reg,min_temperature,max_temperature,eps,noise_type,noise_type, pre10, rec10, ndcg10, pre20, rec20, ndcg20, pre30, rec30, ndcg30, pre40, rec40, ndcg40, pre50, rec50, ndcg50" + '\n')
            f.write(log + '\n')  # 加\n换行显示
            f.close()
    else:
        with open(csv_path, 'a+') as f:
            f.write(log + '\n')  # 加\n换行显示
            f.close()
            
def load_uit(path):
    num_users = -1
    num_items = -1
    data = []
    with open(path) as f:
        for line in f:
            line = [int(i) for i in line.split('\t')]
            data.append(line)
            num_users = max(line[0], num_users)
            num_items = max(line[1], num_items)
            num_items = max(line[2], num_items)            
    num_users, num_items = num_users+1, num_items+1
    return data, num_users, num_items 

def load_data(path):
    print('Loading train and test data...', end='')
    sys.stdout.flush()
    train_data, num_users, num_items, = load_uit(path+'.train')
    test_data, num_users2, num_items2 = load_uit(path+'.test')
    num_users = max(num_users, num_users2)
    num_items = max(num_items, num_items2)
    print('Done.')
    print()
    print('Number of users: %d'%num_users)
    print('Number of items: %d'%num_items)
    print('Number of train data: %d'%len(train_data))
    print('Number of test data: %d'%len(test_data))
    logging.info('Number of users: %d'%num_users)
    logging.info('Number of items: %d'%num_items)
    logging.info('Number of train data: %d'%len(train_data))
    logging.info('Number of test data: %d'%len(test_data))
    sys.stdout.flush()
    return train_data, test_data, num_users, num_items

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['CDAE', 'CML', 'NeuMF', 'GMF', 'MLP', 'BPRMF', 'JRL', 'LRML', 'CT4Rec'],
                        default='CT4Rec') 
    parser.add_argument('--dataset_path', nargs='?', default='./datasets/',
                        help='Data path.')
    parser.add_argument('--dataset', nargs='?', default='Automotive',
                        help='Name of the dataset.')     
    parser.add_argument('--learning_rate', type=float, default=1e-3) 
    
    parser.add_argument('--reg_rate', type=float, default=0.00001)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--num_neg_items', type=int, default=1,
                        help='number of negtative items')     
    parser.add_argument('--epochs', type=int, default=500)    
    parser.add_argument('--batch_size', type=int, default=1024) 
    
    parser.add_argument('--ssl_reg', type=float, default=0.1,
                        help='Regularization coefficient for ssl.')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--max_temp', type=float,default=0.1,
                        help="the hyper-parameter")
    parser.add_argument('--min_temp', type=float,default=0.1,
                        help="the hyper-parameter")
    parser.add_argument('--noise_type', type=list, default=['uniform_noise','uniform_noise'], 
                        help='type of noise. Like [uniform_noise, uniform_noise],[uniform_noise, Gussian_noise],[Gussian_noise, Gussian_noise]')
    
    parser.add_argument('--verbose', type=int, default=2,
                        help='Interval of evaluation.')
    return parser.parse_args()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    print(args)    
    code_name = os.path.basename(__file__).split('.')[0]
    log_path_ = "log/%s_%s/" % (code_name, strftime('%Y-%m-%d', localtime()))
    if not os.path.exists(log_path_):
        os.makedirs(log_path_)
        
    log_path = log_path_ + "%s_embed_size%.4f_reg%.5f_ssl_reg%.5f_lr%0.5f_eps%.3f_max-temperature%.3f_min-temperature%.3f_noise_type%s_%s_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.ssl_reg, args.learning_rate, args.eps, args.max_temp, args.min_temp, 
        args.noise_type[0], args.noise_type[1], strftime('%Y_%m_%d_%H', localtime()))
    
    logging.basicConfig(filename=log_path,level=logging.INFO)  
    logging.info(args) 

    train_data, test_data, user_count, item_count = load_data(args.dataset_path + args.dataset)
    train(user_count,item_count,train_data, test_data, args.embedding_size, args.eps, args.min_temp, args.max_temp, \
          args.noise_type, args.learning_rate, args.reg_rate, args.epochs, args.batch_size, args.ssl_reg)



