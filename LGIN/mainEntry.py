#encoding=utf-8
import numpy as np
import os 
import configparser
import modelTraining

cf = configparser.SafeConfigParser()
cf.read("paramsConfigPython")

root_dir = cf.get("param", "root_dir") 
dataset = cf.get("param", "dataset") 
root_dir = root_dir + dataset + '/'
gpu = cf.get("param", "gpu") 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu 

GNN_inner_dim = cf.getint("param", "GNN_inner_dim")
dropout = cf.getfloat("param", "dropout") 
lr = cf.getfloat("param", "lr")  
lambda_G = cf.getfloat("param", "lambda_G") 
labmda_L = cf.getfloat("param", "labmda_L") 
lambda_l2 = cf.getfloat("param", "lambda_l2") 
dropout = cf.getfloat("param", "dropout") 
start_record_epoch = cf.getint("param", "start_record_epoch") 
train_max_epoch_num = cf.getint("param", "train_max_epoch_num") 
patience = cf.getint("param", "patience") 
test_batch_size = cf.getint("param", "test_batch_size") 

modelTraining.modelTraining(
        root_dir, 
        dataset, 
        GNN_inner_dim,
        dropout,
        start_record_epoch,
        train_max_epoch_num,
        patience,
        test_batch_size,
        lr,
        lambda_G,
        labmda_L,
        lambda_l2
        )

