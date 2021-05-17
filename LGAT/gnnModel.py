#encoding=utf-8
'''
'''
import numpy as np
import tensorflow as tf


def build_model(options, features_tensor, adj_2_tensor, mask_2_tensor, mask_2_tensor_nor, adj_1_tensor, mask_1_tensor, mask_1_tensor_nor, labels_tensor, dropout_tensor):
    """
    """
    variables_map, gat_variables, film_variables = init_variables(options)
    model_pred, gammas, betas = forward(options, features_tensor, adj_2_tensor, mask_2_tensor, mask_2_tensor_nor, adj_1_tensor, mask_1_tensor, mask_1_tensor_nor, dropout_tensor, variables_map) # pred，shape=(n, labels_num)
    model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=model_pred, labels=labels_tensor) # loss
    model_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(model_pred, axis=-1), axis=1), tf.argmax(labels_tensor, axis=1)) # acc
    
    model_loss = tf.reduce_mean(model_loss) 
    model_acc = tf.reduce_mean(model_acc)
    
    model_loss += tf.add_n([tf.nn.l2_loss(v) for v in gat_variables]) * options['lambda_G']
    
    model_loss += tf.add_n([tf.nn.l2_loss(v) for v in film_variables]) * options['labmda_L']
    model_loss += tf.add_n([tf.reduce_mean((v-1.0) ** 2) for v in gammas]) * options['lambda_l2']
    model_loss += tf.add_n([tf.reduce_mean(v ** 2) for v in betas]) * options['lambda_l2']
    
    optimizer = tf.train.AdamOptimizer(options['lr'], name='train_update_op')
    train_update_op = optimizer.minimize(model_loss)
    
    return train_update_op, model_pred, model_loss, model_acc



def init_variables(options):
    """
    """
    variablesMap={}
    
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    
    variablesMap['M_c_gamma_1_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_0')
    variablesMap['M_c_gamma_2_2_0'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_0')
    variablesMap['M_c_beta_1_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_0')
    variablesMap['M_c_beta_2_2_0'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_0')
    variablesMap['M_e_gamma_c_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_0')
    variablesMap['M_e_beta_c_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_0')
    variablesMap['M_e_gamma_n_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_0')
    variablesMap['M_e_beta_n_2_0'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_0')
    
    variablesMap['M_c_gamma_1_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_1')
    variablesMap['M_c_gamma_2_2_1'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_1')
    variablesMap['M_c_beta_1_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_1')
    variablesMap['M_c_beta_2_2_1'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_1')
    variablesMap['M_e_gamma_c_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_1')
    variablesMap['M_e_beta_c_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_1')
    variablesMap['M_e_gamma_n_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_1')
    variablesMap['M_e_beta_n_2_1'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_1')
    
    variablesMap['M_c_gamma_1_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_2')
    variablesMap['M_c_gamma_2_2_2'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_2')
    variablesMap['M_c_beta_1_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_2')
    variablesMap['M_c_beta_2_2_2'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_2')
    variablesMap['M_e_gamma_c_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_2')
    variablesMap['M_e_beta_c_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_2')
    variablesMap['M_e_gamma_n_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_2')
    variablesMap['M_e_beta_n_2_2'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_2')
    
    variablesMap['M_c_gamma_1_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_3')
    variablesMap['M_c_gamma_2_2_3'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_3')
    variablesMap['M_c_beta_1_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_3')
    variablesMap['M_c_beta_2_2_3'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_3')
    variablesMap['M_e_gamma_c_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_3')
    variablesMap['M_e_beta_c_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_3')
    variablesMap['M_e_gamma_n_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_3')
    variablesMap['M_e_beta_n_2_3'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_3')
    
    variablesMap['M_c_gamma_1_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_4')
    variablesMap['M_c_gamma_2_2_4'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_4')
    variablesMap['M_c_beta_1_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_4')
    variablesMap['M_c_beta_2_2_4'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_4')
    variablesMap['M_e_gamma_c_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_4')
    variablesMap['M_e_beta_c_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_4')
    variablesMap['M_e_gamma_n_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_4')
    variablesMap['M_e_beta_n_2_4'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_4')
    
    variablesMap['M_c_gamma_1_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_5')
    variablesMap['M_c_gamma_2_2_5'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_5')
    variablesMap['M_c_beta_1_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_5')
    variablesMap['M_c_beta_2_2_5'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_5')
    variablesMap['M_e_gamma_c_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_5')
    variablesMap['M_e_beta_c_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_5')
    variablesMap['M_e_gamma_n_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_5')
    variablesMap['M_e_beta_n_2_5'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_5')
    
    variablesMap['M_c_gamma_1_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_6')
    variablesMap['M_c_gamma_2_2_6'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_6')
    variablesMap['M_c_beta_1_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_6')
    variablesMap['M_c_beta_2_2_6'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_6')
    variablesMap['M_e_gamma_c_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_6')
    variablesMap['M_e_beta_c_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_6')
    variablesMap['M_e_gamma_n_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_6')
    variablesMap['M_e_beta_n_2_6'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_6')
    
    variablesMap['M_c_gamma_1_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_gamma_1_2_7')
    variablesMap['M_c_gamma_2_2_7'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_gamma_2_2_7')
    variablesMap['M_c_beta_1_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_c_beta_1_2_7')
    variablesMap['M_c_beta_2_2_7'] = tf.Variable(he_initializer([options['hid_units'][0], options['features_num']]), dtype=tf.float32, name='M_c_beta_2_2_7')
    variablesMap['M_e_gamma_c_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_c_2_7')
    variablesMap['M_e_beta_c_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_c_2_7')
    variablesMap['M_e_gamma_n_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_gamma_n_2_7')
    variablesMap['M_e_beta_n_2_7'] = tf.Variable(he_initializer([options['features_num'], options['hid_units'][0]]), dtype=tf.float32, name='M_e_beta_n_2_7')
    
    variablesMap['M_c_gamma_1_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['hid_units'][0]*8]), dtype=tf.float32, name='M_c_gamma_1_1')
    variablesMap['M_c_beta_1_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['hid_units'][0]*8]), dtype=tf.float32, name='M_c_beta_1_1')
    variablesMap['M_e_gamma_c_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['classes_num']]), dtype=tf.float32, name='M_e_gamma_c_1')
    variablesMap['M_e_beta_c_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['classes_num']]), dtype=tf.float32, name='M_e_beta_c_1')
    variablesMap['M_e_gamma_n_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['classes_num']]), dtype=tf.float32, name='M_e_gamma_n_1')
    variablesMap['M_e_beta_n_1'] = tf.Variable(he_initializer([options['hid_units'][0]*8, options['classes_num']]), dtype=tf.float32, name='M_e_beta_n_1')
    
    
    D_W_0 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_0") # shape=(nNodes,dim)
    D_f1_0 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_0") # shape=(nNodes,dim)
    D_f1_b_0 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_0") # shape=(nNodes,dim)
    D_f2_0 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_0") # shape=(nNodes,dim)
    D_f2_b_0 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_0") # shape=(nNodes,dim)
    D_b_0 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_0") # shape=(nNodes,dim)
    D_map_b_0 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_0") # shape=(nNodes,dim)
    
    D_W_1 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_1") # shape=(nNodes,dim)
    D_f1_1 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_1") # shape=(nNodes,dim)
    D_f1_b_1 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_1") # shape=(nNodes,dim)
    D_f2_1 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_1") # shape=(nNodes,dim)
    D_f2_b_1 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_1") # shape=(nNodes,dim)
    D_b_1 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_1") # shape=(nNodes,dim)
    D_map_b_1 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_1") # shape=(nNodes,dim)
    
    D_W_2 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_2") # shape=(nNodes,dim)
    D_f1_2 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_2") # shape=(nNodes,dim)
    D_f1_b_2 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_2") # shape=(nNodes,dim)
    D_f2_2 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_2") # shape=(nNodes,dim)
    D_f2_b_2 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_2") # shape=(nNodes,dim)
    D_b_2 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_2") # shape=(nNodes,dim)
    D_map_b_2 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_2") # shape=(nNodes,dim)
    
    D_W_3 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_3") # shape=(nNodes,dim)
    D_f1_3 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_3") # shape=(nNodes,dim)
    D_f1_b_3 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_3") # shape=(nNodes,dim)
    D_f2_3 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_3") # shape=(nNodes,dim)
    D_f2_b_3 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_3") # shape=(nNodes,dim)
    D_b_3 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_3") # shape=(nNodes,dim)
    D_map_b_3 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_3") # shape=(nNodes,dim)
    
    D_W_4 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_4") # shape=(nNodes,dim)
    D_f1_4 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_4") # shape=(nNodes,dim)
    D_f1_b_4 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_4") # shape=(nNodes,dim)
    D_f2_4 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_4") # shape=(nNodes,dim)
    D_f2_b_4 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_4") # shape=(nNodes,dim)
    D_b_4 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_4") # shape=(nNodes,dim)
    D_map_b_4 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_4") # shape=(nNodes,dim)
    
    D_W_5 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_5") # shape=(nNodes,dim)
    D_f1_5 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_5") # shape=(nNodes,dim)
    D_f1_b_5 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_5") # shape=(nNodes,dim)
    D_f2_5 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_5") # shape=(nNodes,dim)
    D_f2_b_5 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_5") # shape=(nNodes,dim)
    D_b_5 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_5") # shape=(nNodes,dim)
    D_map_b_5 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_5") # shape=(nNodes,dim)
    
    D_W_6 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_6") # shape=(nNodes,dim)
    D_f1_6 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_6") # shape=(nNodes,dim)
    D_f1_b_6 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_6") # shape=(nNodes,dim)
    D_f2_6 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_6") # shape=(nNodes,dim)
    D_f2_b_6 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_6") # shape=(nNodes,dim)
    D_b_6 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_6") # shape=(nNodes,dim)
    D_map_b_6 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_6") # shape=(nNodes,dim)
    
    D_W_7 = tf.Variable(tf.random_uniform([options['features_num'], options['hid_units'][0]], -0.01, 0.01), dtype=tf.float32, name="D_W_7") # shape=(nNodes,dim)
    D_f1_7 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_7") # shape=(nNodes,dim)
    D_f1_b_7 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_7") # shape=(nNodes,dim)
    D_f2_7 = tf.Variable(tf.random_uniform([options['hid_units'][0], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_7") # shape=(nNodes,dim)
    D_f2_b_7 = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_7") # shape=(nNodes,dim)
    D_b_7 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_b_7") # shape=(nNodes,dim)
    D_map_b_7 = tf.Variable(tf.random_uniform([options['hid_units'][0],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_7") # shape=(nNodes,dim)
    
    D_W_2_layer = tf.Variable(tf.random_uniform([8*options['hid_units'][0], options['classes_num']], -0.01, 0.01), dtype=tf.float32, name="D_W_2_layer") # shape=(nNodes,dim)
    D_f1_2_layer = tf.Variable(tf.random_uniform([options['classes_num'], 1], -0.01, 0.01), dtype=tf.float32, name="D_f1_2_layer") # shape=(nNodes,dim)
    D_f1_b_2_layer = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f1_b_2_layer") # shape=(nNodes,dim)
    D_f2_2_layer = tf.Variable(tf.random_uniform([options['classes_num'], 1], -0.01, 0.01), dtype=tf.float32, name="D_f2_2_layer") # shape=(nNodes,dim)
    D_f2_b_2_layer = tf.Variable(tf.random_uniform([1], -0.01, 0.01), dtype=tf.float32, name="D_f2_b_2_layer") # shape=(nNodes,dim)
    D_b_2_layer = tf.Variable(tf.random_uniform([options['classes_num'],], -0.01, 0.01), dtype=tf.float32, name="D_b_2_layer") # shape=(nNodes,dim)
    D_map_b_2_layer = tf.Variable(tf.random_uniform([options['classes_num'],], -0.01, 0.01), dtype=tf.float32, name="D_map_b_2_layer") # shape=(nNodes,dim)
    
    variablesMap["D_W_0"]=D_W_0
    variablesMap["D_f1_0"]=D_f1_0
    variablesMap["D_f1_b_0"]=D_f1_b_0
    variablesMap["D_f2_0"]=D_f2_0
    variablesMap["D_f2_b_0"]=D_f2_b_0
    variablesMap["D_b_0"]=D_b_0
    variablesMap["D_map_b_0"]=D_map_b_0
    
    variablesMap["D_W_1"]=D_W_1
    variablesMap["D_f1_1"]=D_f1_1
    variablesMap["D_f1_b_1"]=D_f1_b_1
    variablesMap["D_f2_1"]=D_f2_1
    variablesMap["D_f2_b_1"]=D_f2_b_1
    variablesMap["D_b_1"]=D_b_1
    variablesMap["D_map_b_1"]=D_map_b_1
    
    variablesMap["D_W_2"]=D_W_2
    variablesMap["D_f1_2"]=D_f1_2
    variablesMap["D_f1_b_2"]=D_f1_b_2
    variablesMap["D_f2_2"]=D_f2_2
    variablesMap["D_f2_b_2"]=D_f2_b_2
    variablesMap["D_b_2"]=D_b_2
    variablesMap["D_map_b_2"]=D_map_b_2
    
    variablesMap["D_W_3"]=D_W_3
    variablesMap["D_f1_3"]=D_f1_3
    variablesMap["D_f1_b_3"]=D_f1_b_3
    variablesMap["D_f2_3"]=D_f2_3
    variablesMap["D_f2_b_3"]=D_f2_b_3
    variablesMap["D_b_3"]=D_b_3
    variablesMap["D_map_b_3"]=D_map_b_3
    
    variablesMap["D_W_4"]=D_W_4
    variablesMap["D_f1_4"]=D_f1_4
    variablesMap["D_f1_b_4"]=D_f1_b_4
    variablesMap["D_f2_4"]=D_f2_4
    variablesMap["D_f2_b_4"]=D_f2_b_4
    variablesMap["D_b_4"]=D_b_4
    variablesMap["D_map_b_4"]=D_map_b_4
    
    variablesMap["D_W_5"]=D_W_5
    variablesMap["D_f1_5"]=D_f1_5
    variablesMap["D_f1_b_5"]=D_f1_b_5
    variablesMap["D_f2_5"]=D_f2_5
    variablesMap["D_f2_b_5"]=D_f2_b_5
    variablesMap["D_b_5"]=D_b_5
    variablesMap["D_map_b_5"]=D_map_b_5
    
    variablesMap["D_W_6"]=D_W_6
    variablesMap["D_f1_6"]=D_f1_6
    variablesMap["D_f1_b_6"]=D_f1_b_6
    variablesMap["D_f2_6"]=D_f2_6
    variablesMap["D_f2_b_6"]=D_f2_b_6
    variablesMap["D_b_6"]=D_b_6
    variablesMap["D_map_b_6"]=D_map_b_6
    
    variablesMap["D_W_7"]=D_W_7
    variablesMap["D_f1_7"]=D_f1_7
    variablesMap["D_f1_b_7"]=D_f1_b_7
    variablesMap["D_f2_7"]=D_f2_7
    variablesMap["D_f2_b_7"]=D_f2_b_7
    variablesMap["D_b_7"]=D_b_7
    variablesMap["D_map_b_7"]=D_map_b_7
    
    variablesMap["D_W_2_layer"]=D_W_2_layer
    variablesMap["D_f1_2_layer"]=D_f1_2_layer
    variablesMap["D_f1_b_2_layer"]=D_f1_b_2_layer
    variablesMap["D_f2_2_layer"]=D_f2_2_layer
    variablesMap["D_f2_b_2_layer"]=D_f2_b_2_layer
    variablesMap["D_b_2_layer"]=D_b_2_layer
    variablesMap["D_map_b_2_layer"]=D_map_b_2_layer
    
    gat_variables = [D_W_0, D_f1_0, D_f1_b_0, D_f2_0, D_f2_b_0, D_b_0, D_map_b_0, D_W_1, D_f1_1, D_f1_b_1, D_f2_1, D_f2_b_1, D_b_1, D_map_b_1, D_W_2, D_f1_2, D_f1_b_2, D_f2_2, D_f2_b_2, D_b_2, D_map_b_2, D_W_3, D_f1_3, D_f1_b_3, D_f2_3, D_f2_b_3, D_b_3, D_map_b_3, D_W_4, D_f1_4, D_f1_b_4, D_f2_4, D_f2_b_4, D_b_4, D_map_b_4, D_W_5, D_f1_5, D_f1_b_5, D_f2_5, D_f2_b_5, D_b_5, D_map_b_5, D_W_6, D_f1_6, D_f1_b_6, D_f2_6, D_f2_b_6, D_b_6, D_map_b_6, D_W_7, D_f1_7, D_f1_b_7, D_f2_7, D_f2_b_7, D_b_7, D_map_b_7, D_W_2_layer, D_f1_2_layer, D_f1_b_2_layer, D_f2_2_layer, D_f2_b_2_layer, D_b_2_layer, D_map_b_2_layer]
    film_variables = [
                        variablesMap['M_c_gamma_1_2_0'], variablesMap['M_c_gamma_2_2_0'], variablesMap['M_c_beta_1_2_0'], variablesMap['M_c_beta_2_2_0'], 
                    variablesMap['M_e_gamma_c_2_0'], variablesMap['M_e_beta_c_2_0'], variablesMap['M_e_gamma_n_2_0'], variablesMap['M_e_beta_n_2_0'], 
                      
                    variablesMap['M_c_gamma_1_2_1'], variablesMap['M_c_gamma_2_2_1'], variablesMap['M_c_beta_1_2_1'], variablesMap['M_c_beta_2_2_1'], 
                    variablesMap['M_e_gamma_c_2_1'], variablesMap['M_e_beta_c_2_1'], variablesMap['M_e_gamma_n_2_1'], variablesMap['M_e_beta_n_2_1'], 
                      
                    variablesMap['M_c_gamma_1_2_2'], variablesMap['M_c_gamma_2_2_2'], variablesMap['M_c_beta_1_2_2'], variablesMap['M_c_beta_2_2_2'], 
                    variablesMap['M_e_gamma_c_2_2'], variablesMap['M_e_beta_c_2_2'], variablesMap['M_e_gamma_n_2_2'], variablesMap['M_e_beta_n_2_2'], 

                    variablesMap['M_c_gamma_1_2_3'], variablesMap['M_c_gamma_2_2_3'], variablesMap['M_c_beta_1_2_3'], variablesMap['M_c_beta_2_2_3'], 
                    variablesMap['M_e_gamma_c_2_3'], variablesMap['M_e_beta_c_2_3'], variablesMap['M_e_gamma_n_2_3'], variablesMap['M_e_beta_n_2_3'], 
                      
                    variablesMap['M_c_gamma_1_2_4'], variablesMap['M_c_gamma_2_2_4'], variablesMap['M_c_beta_1_2_4'], variablesMap['M_c_beta_2_2_4'], 
                    variablesMap['M_e_gamma_c_2_4'], variablesMap['M_e_beta_c_2_4'], variablesMap['M_e_gamma_n_2_4'], variablesMap['M_e_beta_n_2_4'], 
                      
                    variablesMap['M_c_gamma_1_2_5'], variablesMap['M_c_gamma_2_2_5'], variablesMap['M_c_beta_1_2_5'], variablesMap['M_c_beta_2_2_5'], 
                    variablesMap['M_e_gamma_c_2_5'], variablesMap['M_e_beta_c_2_5'], variablesMap['M_e_gamma_n_2_5'], variablesMap['M_e_beta_n_2_5'], 

                    variablesMap['M_c_gamma_1_2_6'], variablesMap['M_c_gamma_2_2_6'], variablesMap['M_c_beta_1_2_6'], variablesMap['M_c_beta_2_2_6'], 
                    variablesMap['M_e_gamma_c_2_6'], variablesMap['M_e_beta_c_2_6'], variablesMap['M_e_gamma_n_2_6'], variablesMap['M_e_beta_n_2_6'], 
                      
                    variablesMap['M_c_gamma_1_2_7'], variablesMap['M_c_gamma_2_2_7'], variablesMap['M_c_beta_1_2_7'], variablesMap['M_c_beta_2_2_7'], 
                    variablesMap['M_e_gamma_c_2_7'], variablesMap['M_e_beta_c_2_7'], variablesMap['M_e_gamma_n_2_7'], variablesMap['M_e_beta_n_2_7'], 
                      
                    variablesMap['M_c_gamma_1_1'], variablesMap['M_c_beta_1_1'],
                    variablesMap['M_e_gamma_c_1'], variablesMap['M_e_beta_c_1'], variablesMap['M_e_gamma_n_1'], variablesMap['M_e_beta_n_1']
                      ]
    
    
    return variablesMap, gat_variables, film_variables

    
def forward(options, features_tensor, adj_2_tensor, mask_2_tensor, mask_2_tensor_nor, adj_1_tensor, mask_1_tensor, mask_1_tensor_nor, dropout_tensor, variablesMap, n_heads=[8, 1]):
    """
    """
    attns = []
    gammas = []
    betas = []
    
    adj = adj_2_tensor
    mask = mask_2_tensor
    mask_nor = mask_2_tensor_nor
    embs = features_tensor
    self_adj = adj[:, 0] 
    neis_adj = adj[:, 1:] 
    self_mask = mask[:, 0] 
    neis_mask = mask[:, 1:] 
    neis_mask_nor = mask_nor[:, 1:] 
    
    embs = tf.nn.dropout(embs, 1.0 - dropout_tensor)
    
    all_embs = tf.nn.embedding_lookup(embs, adj) 
    self_embs = all_embs[:, 0]
    neis_embs = all_embs[:, 1:]
    
    for index in range(n_heads[0]):
        h_2, gammas_index, betas_index = layerModel_2(options, all_embs, neis_embs, neis_mask, neis_mask_nor, self_embs, adj_2_tensor, 
                                variablesMap['D_W_'+str(index)], variablesMap['D_f1_'+str(index)], variablesMap['D_f1_b_'+str(index)], variablesMap['D_f2_'+str(index)], variablesMap['D_f2_b_'+str(index)], variablesMap['D_b_'+str(index)], variablesMap['D_map_b_'+str(index)], 
                                variablesMap['M_c_gamma_1_2_'+str(index)], variablesMap['M_c_gamma_2_2_'+str(index)], variablesMap['M_c_beta_1_2_'+str(index)], variablesMap['M_c_beta_2_2_'+str(index)], 
                                variablesMap['M_e_gamma_c_2_'+str(index)], variablesMap['M_e_beta_c_2_'+str(index)], variablesMap['M_e_gamma_n_2_'+str(index)], variablesMap['M_e_beta_n_2_'+str(index)], 
                                dropout_tensor)
        attns.append(h_2)
        gammas.extend(gammas_index)
        betas.extend(betas_index)
    h_1 = tf.concat(attns, axis=-1)
    
    attns = []
    for _ in range(n_heads[1]):
        h_1, gammas_index, betas_index = layerModel_1(options, h_1, adj_1_tensor, mask_1_tensor, mask_1_tensor_nor,
                                variablesMap['D_W_2_layer'], variablesMap['D_f1_2_layer'], variablesMap['D_f1_b_2_layer'], variablesMap['D_f2_2_layer'], variablesMap['D_f2_b_2_layer'], variablesMap['D_b_2_layer'], variablesMap['D_map_b_2_layer'],
                                variablesMap['M_c_gamma_1_1'], variablesMap['M_c_beta_1_1'],
                                variablesMap['M_e_gamma_c_1'], variablesMap['M_e_beta_c_1'], variablesMap['M_e_gamma_n_1'], variablesMap['M_e_beta_n_1'], 
                                dropout_tensor, activation=lambda x: x)
        attns.append(h_1)
        gammas.extend(gammas_index)
        betas.extend(betas_index)
    logits = tf.add_n(attns) / n_heads[1]
    
    return logits, gammas, betas

    
def layerModel_2(options, all_embs, neis_embs, neis_mask, neis_mask_nor, self_embs, adj, W, var_f1, var_f1_b, var_f2, var_f2_b, var_b, map_b, 
                M_c_gamma_1, M_c_gamma_2, M_c_beta_1, M_c_beta_2, 
                M_e_gamma_c, M_e_beta_c, M_e_gamma_n, M_e_beta_n, 
                 in_drop=0.0, activation=tf.nn.elu):
    """
    """
    
    neis_embs_agg = tf.reduce_sum(neis_embs * neis_mask_nor[:,:,None], axis=-2) 
    self_neirs_embs_combine = neis_embs_agg
    
    gamma_W_c = tf.matmul(tf.matmul(self_neirs_embs_combine, M_c_gamma_1), M_c_gamma_2) 
    beta_W_c = tf.matmul(tf.matmul(self_neirs_embs_combine, M_c_beta_1), M_c_beta_2) 
    gamma_W_c = tf.nn.leaky_relu(gamma_W_c) + 1.0
    beta_W_c = tf.nn.leaky_relu(beta_W_c)
    W_local = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(adj)[0], 1, 1]) 
    W_local = W_local * gamma_W_c[:,:,None] + beta_W_c[:,:,None] 
    
    self_edge_gamma = tf.matmul(self_embs, M_e_gamma_c) 
    all_embs_reshape = tf.reshape(all_embs, [-1, tf.shape(all_embs)[-1]]) 
    all_edge_gamma = tf.matmul(all_embs_reshape, M_e_gamma_n) 
    all_edge_gamma = tf.reshape(all_edge_gamma, [tf.shape(all_embs)[0], tf.shape(all_embs)[1], tf.shape(all_edge_gamma)[-1]]) 
    gamma_e = self_edge_gamma[:,None,:] + all_edge_gamma 
    gamma_e = tf.nn.leaky_relu(gamma_e) + 1.0
    self_edge_beta = tf.matmul(self_embs, M_e_beta_c)
    all_edge_beta = tf.matmul(all_embs_reshape, M_e_beta_n) 
    all_edge_beta = tf.reshape(all_edge_beta, [tf.shape(all_embs)[0], tf.shape(all_embs)[1], tf.shape(all_edge_beta)[-1]]) 
    beta_e = self_edge_beta[:,None,:] + all_edge_beta 
    beta_e = tf.nn.leaky_relu(beta_e)
    
    neis_embs_new = tf.matmul(neis_embs, W_local) + map_b 
    self_embs_new = tf.matmul(self_embs[:,None,:], W_local) + map_b 
    all_embs_new = tf.concat([self_embs_new, neis_embs_new], axis=1) 
    
    seq_fts_film = all_embs_new * gamma_e + beta_e
    seq_fts_film_reshape = tf.reshape(seq_fts_film, [-1, tf.shape(seq_fts_film)[-1]]) 
    
    f_1 = tf.matmul(seq_fts_film_reshape, var_f1) + var_f1_b 
    f_2 = tf.matmul(seq_fts_film_reshape, var_f2) + var_f2_b 
    f_1 = tf.squeeze(f_1) 
    f_2 = tf.squeeze(f_2) 
    f_1 = tf.reshape(f_1, [tf.shape(seq_fts_film)[0], tf.shape(seq_fts_film)[1]]) 
    f_2 = tf.reshape(f_2, [tf.shape(seq_fts_film)[0], tf.shape(seq_fts_film)[1]]) 
    
    center_weights = f_1[:, 0] 
    neis_weights = f_2[:,1:] 
    
    logits = center_weights[:,None] + neis_weights 
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + (1.0-neis_mask)*(-1e9)) 
    
    coefs = tf.nn.dropout(coefs, 1.0 - in_drop)
    seq_fts_film = tf.nn.dropout(seq_fts_film, 1.0 - in_drop)
    
    seq_fts = seq_fts_film[:, 1:, :] 
    seq_fts = seq_fts * coefs[:,:,None] * neis_mask[:, :, None] 
    ret = tf.reduce_sum(seq_fts, axis=1) 
    
    ret += var_b
    ret = activation(ret)  
    
    gammas = [gamma_W_c, gamma_e]
    betas = [beta_W_c, beta_e]
    
    return ret, gammas, betas


    
def layerModel_1(options, embs, adj, mask, mask_nor, W, var_f1, var_f1_b, var_f2, var_f2_b, var_b, map_b,
                M_c_gamma_1, M_c_beta_1, 
                M_e_gamma_c, M_e_beta_c, M_e_gamma_n, M_e_beta_n,
                 in_drop=0.0, activation=tf.nn.elu):
    """
    """
    self_adj = adj[:, 0] 
    neis_adj = adj[:, 1:] 
    self_mask = mask[:, 0] 
    neis_mask = mask[:, 1:] 
    neis_mask_nor = mask_nor[:, 1:] 
    
    embs = tf.nn.dropout(embs, 1.0 - in_drop)
    
    all_embs = tf.nn.embedding_lookup(embs, adj) 
    self_embs = tf.nn.embedding_lookup(embs, self_adj) 
    neis_embs = tf.nn.embedding_lookup(embs, neis_adj) 
    neis_embs_agg = tf.reduce_sum(neis_embs * neis_mask_nor[:,:,None], axis=-2) 
    self_neirs_embs_combine = neis_embs_agg
    
    gamma_W_c = tf.matmul(self_neirs_embs_combine, M_c_gamma_1) 
    beta_W_c = tf.matmul(self_neirs_embs_combine, M_c_beta_1) 
    gamma_W_c = tf.nn.leaky_relu(gamma_W_c) + 1.0
    beta_W_c = tf.nn.leaky_relu(beta_W_c)
    W_local = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(adj)[0], 1, 1]) 
    W_local = W_local * gamma_W_c[:,:,None] + beta_W_c[:,:,None] 
    
    self_edge_gamma = tf.matmul(self_embs, M_e_gamma_c) 
    all_embs_reshape = tf.reshape(all_embs, [-1, tf.shape(all_embs)[-1]]) 
    all_edge_gamma = tf.matmul(all_embs_reshape, M_e_gamma_n) 
    all_edge_gamma = tf.reshape(all_edge_gamma, [tf.shape(all_embs)[0], tf.shape(all_embs)[1], tf.shape(all_edge_gamma)[-1]]) 
    gamma_e = self_edge_gamma[:,None,:] + all_edge_gamma 
    gamma_e = tf.nn.leaky_relu(gamma_e) + 1.0
    self_edge_beta = tf.matmul(self_embs, M_e_beta_c) 
    all_edge_beta = tf.matmul(all_embs_reshape, M_e_beta_n) 
    all_edge_beta = tf.reshape(all_edge_beta, [tf.shape(all_embs)[0], tf.shape(all_embs)[1], tf.shape(all_edge_beta)[-1]]) 
    beta_e = self_edge_beta[:,None,:] + all_edge_beta 
    beta_e = tf.nn.leaky_relu(beta_e)    
    
    neis_embs_new = tf.matmul(neis_embs, W_local) + map_b 
    self_embs_new = tf.matmul(self_embs[:,None,:], W_local) + map_b 
    all_embs_new = tf.concat([self_embs_new, neis_embs_new], axis=1) 
    
    seq_fts_film = all_embs_new * gamma_e + beta_e 
    seq_fts_film_reshape = tf.reshape(seq_fts_film, [-1, tf.shape(seq_fts_film)[-1]]) 
    
    f_1 = tf.matmul(seq_fts_film_reshape, var_f1) + var_f1_b
    f_2 = tf.matmul(seq_fts_film_reshape, var_f2) + var_f2_b
    
    f_1 = tf.squeeze(f_1) 
    f_2 = tf.squeeze(f_2) 
    f_1 = tf.reshape(f_1, [tf.shape(seq_fts_film)[0], tf.shape(seq_fts_film)[1]]) 
    f_2 = tf.reshape(f_2, [tf.shape(seq_fts_film)[0], tf.shape(seq_fts_film)[1]]) 
    center_weights = f_1[:, 0] 
    neis_weights = f_2[:,1:] 
    
    logits = center_weights[:,None] + neis_weights 
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + (1.0-neis_mask)*(-1e9)) 
    
    coefs = tf.nn.dropout(coefs, 1.0 - in_drop)
    seq_fts_film = tf.nn.dropout(seq_fts_film, 1.0 - in_drop)
    
    seq_fts = seq_fts_film[:, 1:, :] 
    seq_fts = seq_fts * coefs[:,:,None] * neis_mask[:, :, None] 
    ret = tf.reduce_sum(seq_fts, axis=1) 
    
    ret += var_b
    ret = activation(ret) 
    
    gammas = [gamma_W_c, gamma_e]
    betas = [beta_W_c, beta_e]
    
    return ret, gammas, betas  