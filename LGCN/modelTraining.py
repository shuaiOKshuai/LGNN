#encoding=utf-8
import numpy as np
import tensorflow as tf
import datetime, os
import scipy.sparse as sp
import processTools
import gnnModel
from sklearn.metrics import accuracy_score

def modelTraining(
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
        ):
    options = locals().copy() 
    
    if dataset in {"cora", "citeseer"}: 
        adj, features_ori, y_train, y_val, y_test, train_mask, val_mask, unlabeled_mask, test_mask, idx_unlabeled = processTools.load_data_unlabeled(root_dir, dataset)
        features, spars, features_01 = processTools.preprocess_features(features_ori)
        adj = adj.toarray() 
    else: 
        node_file = root_dir + 'graph.node'
        edge_file = root_dir + 'graph.edge'
        trainList, valList, testList = processTools.generateTrainValTs(node_file, trainNum_perClass=20, valAllNum=500, testAllNum=1000)
        adj, features, features_01, y_train, y_val, y_test, train_mask, val_mask, test_mask = processTools.load_data_other_dataset(dataset, trainList, valList, testList, node_file, edge_file)
    
    features = features.A
    neis_nums = adj.sum(axis=1)
    neis_nums = np.squeeze(neis_nums)
    print('Start to load adj-self information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if os.path.exists(root_dir + 'adj_self.npy'):
        adj_self = np.load(root_dir + 'adj_self.npy', allow_pickle=True)
        adj_self_nor = np.load(root_dir + 'adj_self_nor.npy', allow_pickle=True)
    else:
        adj_self, adj_self_nor = processTools.preprocess_adj_normalization(adj)
        np.save(root_dir + 'adj_self.npy', adj_self)
        np.save(root_dir + 'adj_self_nor.npy', adj_self_nor)
    
    nodes_num = np.shape(neis_nums)[0]
    max_degree = int(np.max(neis_nums))
    max_degree_self = max_degree + 1 
    options['classes_num'] = classes_num = y_train.shape[1]
    options['features_num'] = features_num = features.shape[1]
    
    neis, neis_mask, neis_mask_nor, neighboursDict = processTools.processNodeInfo(adj_self, adj_self_nor, nodes_num)
    
    labels_combine = y_train + y_val + y_test 
    labels_combine_sum = np.sum(labels_combine, axis=0) 
    max_index = np.argmax(labels_combine_sum) 
    mi_f1_labels = [i for i in range(classes_num) if i!=max_index] 
    
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32, name='features_tensor')
    dropout_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_tensor')
    labels_tensor = tf.placeholder(dtype=tf.float32, shape=(None, options['classes_num']), name='labels_tensor') 
    adj_2_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name='adj_2_tensor') 
    mask_2_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='mask_2_tensor') 
    adj_1_tensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name='adj_1_tensor') 
    mask_1_tensor = tf.placeholder(dtype=tf.float32, shape=(None, None), name='mask_1_tensor')
    train_update_op, pred_tensor, loss_tensor, acc_tensor = gnnModel.build_model(options, features_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, labels_tensor, dropout_tensor)
    
    saver = tf.train.Saver()
    cur_path = os.getcwd()
    checkpt_file=cur_path + "/modelsSave/model_save_"+dataset+".ckpt"  
    if not os.path.exists(cur_path + "/modelsSave/"): 
        os.makedirs(cur_path + "/modelsSave/")
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    acc_max = 0.0
    loss_min = 10000000.0
    acc_early_stop = 0.0
    loss_early_stop = 0.0
    epoch_early_stop = 0
    curr_step = 0
    
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        train_node_ids = np.array([i for i in range(nodes_num) if train_mask[i]==True])
        val_node_ids = np.array([i for i in range(nodes_num) if val_mask[i]==True])
        test_node_ids = np.array([i for i in range(nodes_num) if test_mask[i]==True])
        y_train_only = y_train[train_node_ids]
        y_val_only = y_val[val_node_ids]
        y_test_only = y_test[test_node_ids]
        
        if dataset in {"cora", "citeseer"}: 
            train_adj_2, train_mask_2, train_mask_2_nor, train_adj_1, train_mask_1, train_mask_1_nor = processTools.prepareDataset(nodes_num, max_degree_self, train_node_ids, neis, neis_mask, neis_mask_nor)
        else:
            train_adj_2, train_mask_2, train_mask_2_nor, train_adj_1, train_mask_1, train_mask_1_nor = processTools.prepareDataset_bigdata(nodes_num, max_degree_self, train_node_ids, neis, neis_mask, neis_mask_nor)
        val_batches = processTools.get_minibatches_idx(val_node_ids.shape[0], test_batch_size)
        test_batches = processTools.get_minibatches_idx(test_node_ids.shape[0], test_batch_size)
        val_dataset = []
        for _, batch in val_batches:
            batch_array = np.array(batch) 
            val_node_ids_batch = val_node_ids[batch_array] 
            if dataset in {"cora", "citeseer"}: 
                val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1, val_batch_mask_1_nor = processTools.prepareDataset(nodes_num, max_degree_self, val_node_ids_batch, neis, neis_mask, neis_mask_nor)
            else:
                val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1, val_batch_mask_1_nor = processTools.prepareDataset_bigdata(nodes_num, max_degree_self, val_node_ids_batch, neis, neis_mask, neis_mask_nor)
            val_tuple = (val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1, val_batch_mask_1_nor, y_val_only[batch_array])
            val_dataset.append(val_tuple)
        test_dataset = []
        for _, batch in test_batches:
            batch_array = np.array(batch) 
            test_node_ids_batch = test_node_ids[batch_array] 
            if dataset in {"cora", "citeseer"}: 
                test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1, test_batch_mask_1_nor = processTools.prepareDataset(nodes_num, max_degree_self, test_node_ids_batch, neis, neis_mask, neis_mask_nor)
            else:
                test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1, test_batch_mask_1_nor = processTools.prepareDataset_bigdata(nodes_num, max_degree_self, test_node_ids_batch, neis, neis_mask, neis_mask_nor)
            test_tuple = (test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1, test_batch_mask_1_nor, y_test_only[batch_array])
            test_dataset.append(test_tuple)
            
        for epoch in range(train_max_epoch_num):
            _, train_loss_value, train_acc_value = sess.run([train_update_op, loss_tensor, acc_tensor],
                                                            feed_dict={adj_2_tensor : train_adj_2, 
                                                                       mask_2_tensor : train_mask_2_nor, 
                                                                       adj_1_tensor : train_adj_1, 
                                                                       mask_1_tensor : train_mask_1_nor,
                                                                       labels_tensor : y_train_only,
                                                                       dropout_tensor : dropout})
            val_preds = []
            val_losses = []
            for val_batch in val_dataset:
                val_batch_adj_2, val_batch_mask_2, val_batch_mask_2_nor, val_batch_adj_1, val_batch_mask_1, val_batch_mask_1_nor, y_val_batch = val_batch
                val_pred_value_batch, val_loss_value_batch, val_acc_value_batch = sess.run([pred_tensor, loss_tensor, acc_tensor],
                                                            feed_dict={adj_2_tensor : val_batch_adj_2, 
                                                                       mask_2_tensor : val_batch_mask_2_nor, 
                                                                       adj_1_tensor : val_batch_adj_1, 
                                                                       mask_1_tensor : val_batch_mask_1_nor,
                                                                       labels_tensor : y_val_batch,
                                                                       dropout_tensor : 0.0})
                val_preds.append(val_pred_value_batch)
                val_losses.append(val_loss_value_batch)
            val_preds_array = np.concatenate(val_preds)
            y_true_val = np.argmax(y_val_only, axis=-1)
            y_pred_val = np.argmax(val_preds_array, axis=-1)
            val_acc_value = accuracy_score(y_true_val, y_pred_val)
            val_loss_value = np.mean(val_losses)
            
            print('Epoch: %d | Train: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f ' %
                (epoch, train_loss_value, train_acc_value, val_loss_value, val_acc_value))
            
            if epoch<=start_record_epoch:
                continue
            
            if val_acc_value >= acc_max or val_loss_value <= loss_min: 
                if val_acc_value >= acc_max and val_loss_value <= loss_min: 
                    acc_early_stop = val_acc_value
                    loss_early_stop = val_loss_value
                    epoch_early_stop = epoch
                    saver.save(sess, checkpt_file)
#                     print('------------------------------------------------------------------------------------')
                acc_max = np.max((val_acc_value, acc_max))
                loss_min = np.min((val_loss_value, loss_min))
                curr_step = 0
            else: 
                curr_step += 1
                if curr_step == patience: 
                    print('Early stop model validation loss: ', loss_early_stop, ', accuracy: ', acc_early_stop, ', epoch: ', epoch_early_stop)
                    break
        
        saver.restore(sess, checkpt_file) 
        test_preds = []
        test_losses = []
        for test_batch in test_dataset:
            batch_array = np.array(batch)
            test_batch_adj_2, test_batch_mask_2, test_batch_mask_2_nor, test_batch_adj_1, test_batch_mask_1, test_batch_mask_1_nor, y_test_batch = test_batch
            test_pred_value_batch, test_loss_value_batch, test_acc_value_batch = sess.run([pred_tensor, loss_tensor, acc_tensor],
                                                        feed_dict={adj_2_tensor : test_batch_adj_2, 
                                                                   mask_2_tensor : test_batch_mask_2_nor, 
                                                                   adj_1_tensor : test_batch_adj_1, 
                                                                   mask_1_tensor : test_batch_mask_1_nor,
                                                                   labels_tensor : y_test_batch,
                                                                   dropout_tensor : 0.0})
            test_preds.append(test_pred_value_batch)
            test_losses.append(test_loss_value_batch)
        test_preds_array = np.concatenate(test_preds)
        y_true_test = np.argmax(y_test_only, axis=-1)
        y_pred_test = np.argmax(test_preds_array, axis=-1)
        test_acc_value = accuracy_score(y_true_test, y_pred_test)
        test_loss_value = np.mean(test_losses)
        test_micro_f1, test_macro_f1 = processTools.micro_macro_f1_removeMiLabels(y_true_test, y_pred_test, mi_f1_labels)
        
        print('----------------------------------------------------------------------------------')
        print('Train early stop epoch == ', epoch_early_stop)
        print('Test acc == ', test_acc_value)
        print('Test mi-f1 == ', test_micro_f1)
        print('Test ma-f1 == ', test_macro_f1)
        print('----------------------------------------------------------------------------------')
        
    
    
    
    
    
    
    
    
    
    