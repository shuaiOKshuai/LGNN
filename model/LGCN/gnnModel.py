#encoding=utf-8
import numpy as np
import tensorflow as tf


def build_model(options, features_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, labels_tensor, dropout_tensor):
    """
    build the model
    """
    variables_map, variables, gcn_variables, film_variables = init_variables(options)
    model_pred, gammas_2, betas_2, gammas_1, betas_1 = forward(features_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, dropout_tensor, variables_map) 
    model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=model_pred, labels=labels_tensor) 
    model_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(model_pred, axis=1), axis=1), tf.argmax(labels_tensor, axis=1)) 
    
    model_loss = tf.reduce_mean(model_loss) 
    model_acc = tf.reduce_mean(model_acc)
    
    model_loss += tf.add_n([tf.nn.l2_loss(v) for v in gcn_variables]) * options['lambda_G']
    model_loss += tf.add_n([tf.nn.l2_loss(v) for v in film_variables]) * options['labmda_L']
    
    model_loss += tf.add_n([tf.reduce_mean((v-1.0) ** 2) for v in gammas_2]) * options['lambda_l2']
    model_loss += tf.add_n([tf.reduce_mean((v-1.0) ** 2) for v in gammas_1]) * options['lambda_l2']
    model_loss += tf.add_n([tf.reduce_mean(v ** 2) for v in betas_2]) * options['lambda_l2']
    model_loss += tf.add_n([tf.reduce_mean(v ** 2) for v in betas_1]) * options['lambda_l2']
    
    optimizer = tf.train.AdamOptimizer(options['lr'], name='train_update_op')
    train_update_op = optimizer.minimize(model_loss)
    
    return train_update_op, model_pred, model_loss, model_acc



def init_variables(options):
    variables_map = {}
        
    
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    variables_map['W_1'] = tf.get_variable('W_1', [options['GNN_inner_dim'], options['classes_num']], initializer=he_initializer)
    variables_map['W_2'] = tf.get_variable('W_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_c_gamma_1_2'] = tf.get_variable('M_c_gamma_1_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_c_gamma_2_2'] = tf.get_variable('M_c_gamma_2_2', [options['GNN_inner_dim'], options['features_num']], initializer=he_initializer)
    variables_map['M_c_beta_1_2'] = tf.get_variable('M_c_beta_1_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_c_beta_2_2'] = tf.get_variable('M_c_beta_2_2', [options['GNN_inner_dim'], options['features_num']], initializer=he_initializer)
    variables_map['M_c_gamma_1_1'] = tf.get_variable('M_c_gamma_1_1', [options['GNN_inner_dim'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_c_beta_1_1'] = tf.get_variable('M_c_beta_1_1', [options['GNN_inner_dim'], options['GNN_inner_dim']], initializer=he_initializer)
    
    variables_map['M_e_gamma_c_2'] = tf.get_variable('M_e_gamma_c_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_e_beta_c_2'] = tf.get_variable('M_e_beta_c_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_e_gamma_n_2'] = tf.get_variable('M_e_gamma_n_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_e_beta_n_2'] = tf.get_variable('M_e_beta_n_2', [options['features_num'], options['GNN_inner_dim']], initializer=he_initializer)
    variables_map['M_e_gamma_c_1'] = tf.get_variable('M_e_gamma_c_1', [options['GNN_inner_dim'], options['classes_num']], initializer=he_initializer)
    variables_map['M_e_beta_c_1'] = tf.get_variable('M_e_beta_c_1', [options['GNN_inner_dim'], options['classes_num']], initializer=he_initializer)
    variables_map['M_e_gamma_n_1'] = tf.get_variable('M_e_gamma_n_1', [options['GNN_inner_dim'], options['classes_num']], initializer=he_initializer)
    variables_map['M_e_beta_n_1'] = tf.get_variable('M_e_beta_n_1', [options['GNN_inner_dim'], options['classes_num']], initializer=he_initializer)
    
    variables_map['b_map_2'] = tf.Variable(tf.random_uniform([options['GNN_inner_dim'],], -0.01, 0.01), dtype=tf.float32, name="b_map_2")
    variables_map['b_sum_2'] = tf.Variable(tf.random_uniform([options['GNN_inner_dim'],], -0.01, 0.01), dtype=tf.float32, name="b_sum_2")
    variables_map['b_map_1'] = tf.Variable(tf.random_uniform([options['classes_num'],], -0.01, 0.01), dtype=tf.float32, name="b_map_1")
    variables_map['b_sum_1'] = tf.Variable(tf.random_uniform([options['classes_num'],], -0.01, 0.01), dtype=tf.float32, name="b_sum_1")
    
    variables = [variables_map[key] for key in variables_map]
    gcn_variables = [variables_map['W_1'], variables_map['W_2'], variables_map['b_map_2'], variables_map['b_sum_2'], variables_map['b_map_1'], variables_map['b_sum_1']]
    film_variables = [variables_map['M_c_gamma_1_2'], variables_map['M_c_gamma_2_2'], variables_map['M_c_beta_1_2'], variables_map['M_c_beta_2_2'], 
                      variables_map['M_c_gamma_1_1'], variables_map['M_c_beta_1_1'],
                      variables_map['M_e_gamma_c_2'], variables_map['M_e_beta_c_2'], variables_map['M_e_gamma_n_2'], variables_map['M_e_beta_n_2'], 
                      variables_map['M_e_gamma_c_1'], variables_map['M_e_beta_c_1'], variables_map['M_e_gamma_n_1'], variables_map['M_e_beta_n_1'], 
                      ]
    
    return variables_map, variables, gcn_variables, film_variables
    
def forward(features_tensor, adj_2_tensor, mask_2_tensor, adj_1_tensor, mask_1_tensor, dropout_tensor, variablesMap):
    """
    """
    h_2, gammas_2, betas_2 = layerModel_2(features_tensor, adj_2_tensor, mask_2_tensor, variablesMap['W_2'], 
                                          variablesMap['b_map_2'], variablesMap['b_sum_2'], 
                     variablesMap['M_c_gamma_1_2'], variablesMap['M_c_gamma_2_2'], variablesMap['M_c_beta_1_2'], variablesMap['M_c_beta_2_2'], 
                    variablesMap['M_e_gamma_c_2'], variablesMap['M_e_beta_c_2'], variablesMap['M_e_gamma_n_2'], variablesMap['M_e_beta_n_2'], 
                     dropout_tensor)
    h_1, gammas_1, betas_1 = layerModel_1(h_2, adj_1_tensor, mask_1_tensor, variablesMap['W_1'], 
                                          variablesMap['b_map_1'], variablesMap['b_sum_1'], 
                     variablesMap['M_c_gamma_1_1'], variablesMap['M_c_beta_1_1'],
                     variablesMap['M_e_gamma_c_1'], variablesMap['M_e_beta_c_1'], variablesMap['M_e_gamma_n_1'], variablesMap['M_e_beta_n_1'], 
                       dropout_tensor)
    return h_1, gammas_2, betas_2, gammas_1, betas_1

    
def layerModel_2(embs, adj, mask, W, b_map, b_sum, M_c_gamma_1, M_c_gamma_2, M_c_beta_1, M_c_beta_2, M_e_gamma_c, M_e_beta_c, M_e_gamma_n, M_e_beta_n, in_drop):
    """
    """
    self_adj = adj[:, 0] 
    neis_adj = adj[:, 1:] 
    neis_mask = mask[:, 1:] 
    
    embs = tf.nn.dropout(embs, 1.0 - in_drop)
    
    self_embs = tf.nn.embedding_lookup(embs, self_adj) 
    neis_embs = tf.nn.embedding_lookup(embs, neis_adj) 
    neis_embs_agg = tf.reduce_sum(neis_embs * neis_mask[:,:,None], axis=-2) 
    self_neirs_embs_combine = neis_embs_agg
    
    gamma_W_c = tf.matmul(tf.matmul(self_neirs_embs_combine, M_c_gamma_1), M_c_gamma_2) 
    beta_W_c = tf.matmul(tf.matmul(self_neirs_embs_combine, M_c_beta_1), M_c_beta_2) 
    gamma_W_c = tf.nn.leaky_relu(gamma_W_c) + 1.0
    beta_W_c = tf.nn.leaky_relu(beta_W_c)
    W_local = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(adj)[0], 1, 1])
    W_local = W_local * gamma_W_c[:,:,None] + beta_W_c[:,:,None]
    W_local = tf.nn.dropout(W_local, 1.0 - in_drop)
    
    self_edge_gamma = tf.matmul(self_embs, M_e_gamma_c) 
    neis_embs_reshape = tf.reshape(neis_embs, [-1, tf.shape(neis_embs)[-1]]) 
    neis_edge_gamma = tf.matmul(neis_embs_reshape, M_e_gamma_n) 
    neis_edge_gamma = tf.reshape(neis_edge_gamma, [tf.shape(neis_embs)[0], tf.shape(neis_embs)[1], tf.shape(neis_edge_gamma)[-1]]) 
    self_edge_gamma_reshape = tf.tile(tf.expand_dims(self_edge_gamma, axis=1), [1, tf.shape(neis_edge_gamma)[1], 1])
    gamma_e = self_edge_gamma_reshape + neis_edge_gamma 
    gamma_e = tf.nn.leaky_relu(gamma_e) + 1.0
    
    self_edge_beta = tf.matmul(self_embs, M_e_beta_c) 
    neis_edge_beta = tf.matmul(neis_embs_reshape, M_e_beta_n) 
    neis_edge_beta = tf.reshape(neis_edge_beta, [tf.shape(neis_embs)[0], tf.shape(neis_embs)[1], tf.shape(neis_edge_beta)[-1]])
    self_edge_beta_reshape = tf.tile(tf.expand_dims(self_edge_beta, axis=1), [1, tf.shape(neis_edge_beta)[1], 1])
    beta_e = self_edge_beta_reshape + neis_edge_beta 
    beta_e = tf.nn.leaky_relu(beta_e)
    
    neis_embs_new = tf.matmul(neis_embs, W_local) 
    neis_embs_new += b_map
    
    neis_film = neis_embs_new * gamma_e + beta_e 
    neis_film = tf.nn.dropout(neis_film, 1.0 - in_drop)
    neis_film = neis_film * neis_mask[:, :, None] 
    ret = tf.reduce_sum(neis_film, axis=-2) 
    ret += b_sum
    ret = tf.nn.elu(ret)  
    
    gammas = [gamma_W_c, gamma_e]
    betas = [beta_W_c, beta_e]
    
    return ret, gammas, betas
    

def layerModel_1(embs, adj, mask, W, b_map, b_sum, M_c_gamma_1, M_c_beta_1, M_e_gamma_c, M_e_beta_c, M_e_gamma_n, M_e_beta_n, in_drop):
    """
    """
    self_adj = adj[:, 0] 
    neis_adj = adj[:, 1:] 
    neis_mask = mask[:, 1:] 
    
    embs = tf.nn.dropout(embs, 1.0 - in_drop)
    
    self_embs = tf.nn.embedding_lookup(embs, self_adj) 
    neis_embs = tf.nn.embedding_lookup(embs, neis_adj) 
    neis_embs_agg = tf.reduce_sum(neis_embs * neis_mask[:,:,None], axis=-2) 
    self_neirs_embs_combine = neis_embs_agg
    
    gamma_W_c = tf.matmul(self_neirs_embs_combine, M_c_gamma_1) 
    beta_W_c = tf.matmul(self_neirs_embs_combine, M_c_beta_1) 
    gamma_W_c = tf.nn.leaky_relu(gamma_W_c) + 1.0
    beta_W_c = tf.nn.leaky_relu(beta_W_c)
    W_local = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(adj)[0], 1, 1]) 
    W_local = W_local * gamma_W_c[:,:,None] + beta_W_c[:,:,None] 
    W_local = tf.nn.dropout(W_local, 1.0 - in_drop)
    
    self_edge_gamma = tf.matmul(self_embs, M_e_gamma_c) 
    neis_embs_reshape = tf.reshape(neis_embs, [-1, tf.shape(neis_embs)[-1]]) 
    neis_edge_gamma = tf.matmul(neis_embs_reshape, M_e_gamma_n) 
    neis_edge_gamma = tf.reshape(neis_edge_gamma, [tf.shape(neis_embs)[0], tf.shape(neis_embs)[1], tf.shape(neis_edge_gamma)[-1]]) 
    self_edge_gamma_reshape = tf.tile(tf.expand_dims(self_edge_gamma, axis=1), [1, tf.shape(neis_edge_gamma)[1], 1])
    gamma_e = self_edge_gamma_reshape + neis_edge_gamma 
    gamma_e = tf.nn.leaky_relu(gamma_e) + 1.0
    
    self_edge_beta = tf.matmul(self_embs, M_e_beta_c) 
    neis_edge_beta = tf.matmul(neis_embs_reshape, M_e_beta_n) 
    neis_edge_beta = tf.reshape(neis_edge_beta, [tf.shape(neis_embs)[0], tf.shape(neis_embs)[1], tf.shape(neis_edge_beta)[-1]]) 
    self_edge_beta_reshape = tf.tile(tf.expand_dims(self_edge_beta, axis=1), [1, tf.shape(neis_edge_beta)[1], 1])
    beta_e = self_edge_beta_reshape + neis_edge_beta 
    beta_e = tf.nn.leaky_relu(beta_e)
    
    neis_embs_new = tf.matmul(neis_embs, W_local) 
    neis_embs_new += b_map
   
    neis_film = neis_embs_new * gamma_e + beta_e 
    neis_film = tf.nn.dropout(neis_film, 1.0 - in_drop)
    neis_film = neis_film * neis_mask[:, :, None]
    ret = tf.reduce_sum(neis_film, axis=-2) 
    ret += b_sum
    ret = tf.nn.elu(ret)  
    
    gammas = [gamma_W_c, gamma_e]
    betas = [beta_W_c, beta_e]
    
    return ret, gammas, betas