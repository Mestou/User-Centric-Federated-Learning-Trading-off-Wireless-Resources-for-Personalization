from __future__ import absolute_import, division, print_function
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numdifftools as nd
import numpy as np

def evaluate_gradient(w1, w2, lambdas, eta, j):
    loss_ = []
    for i in range(0,10):
        if i!=j:
            loss_.append(np.dot(w1.reshape(w1.shape[0]*w1.shape[1]),w2[i].reshape(w2[i].shape[0]*w2[i].shape[1])))
        else:
            loss_.append(0)
    m = sorted(range(len(loss_)), key=lambda i: loss_[i], reverse=True)[:2]
    loss = lambda x: np.dot(w1.reshape(w1.shape[0]*w1.shape[1]), x)

    for u in m:
        w2[u] = np.reshape(w2[u].reshape(w2[u].shape[0]*w2[u].shape[1]) - lambdas * eta * nd.Gradient(loss)(np.array(w2[u].reshape(w2[u].shape[0]*w2[u].shape[1]))),[w2[u].shape[0],w2[u].shape[1]])

    return w2

def compute_weights(embeddings_value, msg, nodes):


    for i in range(0, nodes):
        embeddings_value.append(msg[i][-2][-2])
        embeddings_value[i] = np.reshape(embeddings_value[i], msg[i][-2][-2].shape)

    for i in range(0, nodes):
        embeddings_value = evaluate_gradient(embeddings_value[i],embeddings_value,lambdas=10,eta=0.01,j=i)

    return embeddings_value



def user_weighting(delta,nodes,fracs,samples_user,var,collab = 'VAN',cluster_mode = 'OFF',n_cluster = 4):
    max_silhouette = 0
    n = 1
    if collab == 'PER':
        W=np.zeros((nodes,nodes))
        for i in range(0,nodes):
            for j in range(0,nodes):

                 W[i,j] = (-1/(2*np.sqrt(var[i]*var[j])))*(np.linalg.norm(delta[i]-delta[j]))**2

        W=np.exp(W)
        W = np.multiply(W, fracs)
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
    elif collab == 'LOC':
        W = np.diag(np.ones(nodes))
    elif collab == 'ORACLE':
        W = np.block([
            [np.ones((25,25))/25,np.zeros((25,25)),np.zeros((25,25)),np.zeros((25,25))],
            [np.zeros((25,25)),np.ones((25,25))/25,np.zeros((25,25)),np.zeros((25,25))],
            [np.zeros((25,25)),np.zeros((25,25)),np.ones((25,25))/25,np.zeros((25,25))],
            [np.zeros((25,25)),np.zeros((25,25)),np.zeros((25,25)),np.ones((25,25))/25],
            ])
    elif collab == 'VAN':
        W = np.repeat(np.expand_dims(samples_user,axis=0), nodes, axis=0)
    if (cluster_mode == 'on' and collab == 'PER' ):
        for n_cluster in np.arange(2,nodes):
            clusters = KMeans(n_clusters=n_cluster, random_state=1).fit(W)
            cluster_labels = clusters.fit_predict(W)
            silhouette_avg = silhouette_score(W, cluster_labels)
            print("For n_clusters K =", n_cluster,
                  "The average silhouette_score is :", silhouette_avg)
            print("cluster labels = ", cluster_labels)
            sample_silhouette_values = silhouette_samples(W, cluster_labels)
            if max_silhouette < silhouette_avg:
                max_silhouette = silhouette_avg
                n = n_cluster
        clusters = KMeans(n_clusters=n, random_state=1).fit(W)
        print(" ")
        print("######### Best clustering configuration, K = ",n, "Silhouette score = ", max_silhouette, " ##########")
        for i in range(0, nodes):
            W[i, :] = clusters.cluster_centers_[clusters.labels_[i]]
        
    return W




def fomo_weights(delta,delta_old,val,val_old,M,epsilon,P,nodes,rate_limited,it):

    if rate_limited == True:

        if it == 1:
            W_ = np.random.binomial(1,M/nodes,size=(nodes,nodes))

        else:
            W_ = np.diag(np.ones(nodes))

            rand_v = np.random.uniform(0, 1, (nodes, nodes))
            for i in range(0,nodes):
                W_[i,np.where(rand_v[i] < epsilon)] = 1

        if it > 1:

            top_M = np.argpartition(P, -M, axis=1)[:, -M:]

            for i,L in enumerate(top_M):
                for s in L:
                    W_[i,s] = 1

        W=np.zeros((nodes,nodes))
        for i in range(0,nodes):
            for j in np.where(W_[i,:] == 1)[0]:
                W[i,j] = (val_old[i,i]-val[i,j])/(sum([np.linalg.norm(x-y) for x,y in zip(delta_old[i],delta[j])]))


        P=P+W
        P=P/P.sum(axis=1)

        P_ = [np.multiply(W_[i],P[i]) for i in range(0,nodes)]

        W_ = np.diag(np.ones(nodes))

        top_M = np.argpartition(P_, -M, axis=1)[:, -M:]

        for i,L in enumerate(top_M):
            for s in L:
                W_[i,s] = 1

        W = np.array([np.multiply(W_[i],W[i]) for i in range(0,nodes)])
        for i in range(0,nodes):
            W[i][np.where(W[i]<0)] = 0
            if sum(W[i]) == 0:
                W[i][i] = 1
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]

        return W, P

    else :
        W=np.zeros((nodes,nodes))
        for i in range(0,nodes):
            for j in range(0,nodes):
                W[i,j] = (val_old[i,i]-val[i,j])/(sum([np.linalg.norm(x-y) for x,y in zip(delta_old[i],delta[j])]))
        P=P+W
        P=P/P.sum(axis=1)
        W[W<0]=0
        row_sums = W.sum(axis=1)
        for i in range(len(row_sums)):
            if row_sums[i]==0:
                W[i,i]=1
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]

        return W,P




def Ditto_training(mu = 0.5, node_list = [], it = 0, samples_user = []):

    avg_loss = []
    nodes = len(node_list)
    W = user_weighting(delta = None,nodes = nodes, fracs = None, samples_user = samples_user , var = None ,collab = 'VAN',cluster_mode = 'OFF',n_cluster = 4)

    for i in range(0, nodes):
        node_list[i].local_train_Ditto(mu)   # Local personalized updates

    msg = [node_list[i].local_train() for i in range(0, nodes)]  # Local global model updates

    weights = ([[w[0][w_ind] for w in msg] for w_ind in range(0, len(msg[0][0]))])

    node_ind = 0
    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[0, :]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setting the local params


def Fomo_training(M, epsilon, P,samples_user,nodes,node_list,msg_old,loc_val_old,it,rate_limited):

    W = np.repeat(np.expand_dims(samples_user, axis=0), nodes, axis=0)
    msg = [node_list[i].local_train() for i in range(0, nodes)]  # Local
    weights = ([[w[w_ind] for w in msg] for w_ind in range(0, len(msg[0]))])
    loc_val = []

    for node in node_list:
        loc_val.append(np.asarray([node.validate(w[0]) for w in msg]))
    loc_val = np.asarray(loc_val)
    node_ind = 0
    msg = [[np.hstack(np.reshape(x, (-1, 1))) for x in m] for m in msg]
    msg = [np.concatenate(d, axis=0) for d in msg]
    if (it > 0):
        W, P = fomo_weights(msg, msg_old, loc_val, loc_val_old, M, epsilon, P,nodes,rate_limited,it)
    msg_old = msg.copy()
    loc_val_old = loc_val.copy()
    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[node_ind, :]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w[0])  # Setting the local params
        node_ind = node_ind + 1
    return loc_val_old, msg_old, P


def Fedprox_training(mu=0.5,gamma = 0, node_list=[], it=0, samples_user=[]):

        nodes = len(node_list)
        W = user_weighting(delta=0, nodes=nodes, fracs=0, samples_user=samples_user, var=0, collab='VAN',
                           cluster_mode='OFF', n_cluster=4)
        msg = [node_list[i].local_train_Fedprox(gamma,mu) for i in range(0, nodes)]  # Local updates
        weights = ([[w[w_ind] for w in msg] for w_ind in range(0, len(msg[0]))])

        for node in node_list:
            avg_w = [np.average(weights[w_ind], axis=0,weights=W[0]) for w_ind in
                     range(0, len(weights))]  # Averaging local weights
            node.set_model_params(avg_w)  # Setting the local params

        return node_list

def pFedMe_Training( mu = 15, R = 20, K = 3,local_etax = 0.01,global_etax = 0.01,node_list= [],samples_user = []):


    nodes = len(node_list)

    for node in node_list:
        node.epochs = R
        node.local_iterations = K

    # beta (refer to paper) can be assigned by controlling how avg_w is calculated ( beta = 1 is used here )
    W = user_weighting(delta=0, nodes=nodes, fracs=0, samples_user=samples_user, var=0, collab='VAN',
                       cluster_mode='OFF', n_cluster=4)

    msg = [node_list[i].local_train_pFedMe(mu, local_etax=local_etax, global_etax=global_etax) for i in
           range(0, nodes)]  # Local updates
    weights = ([[w[w_ind] for w in msg] for w_ind in range(0, len(msg[0]))])

    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[0]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setti ng the local params

def Scaffold_training(node_list=[], global_c =[],samples_user=[]):

    nodes = len(node_list)
    W = user_weighting(delta=0, nodes=nodes, fracs=0, samples_user=samples_user, var=0, collab='VAN',
                       cluster_mode='OFF', n_cluster=4)
    output = [node_list[i].local_train_Scaffold(global_c) for i in range(0, nodes)]
    c_ = [output[i][1] for i in range(0, nodes)]
    msg = [output[i][0] for i in range(0, nodes)]  # Local updates

    weights = ([[w[w_ind] for w in msg] for w_ind in range(0, len(msg[0]))])
    weights_c = ([[wc[w_indc] for wc in c_] for w_indc in range(0, len(c_[0]))])

    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setting the local params
        global_c = [np.average(weights_c[w_ind], axis=0) for w_ind in
                    range(0, len(weights_c))]  # Averaging local c+

    return global_c

def FedPer_training(W = [], it = -1, node_list = [], init = [], divisor = 2, fracs = [], samples_user = [], cluster_mode = "", n_cluster = 0 ):

    nodes = len(node_list)

    if it == 0:

        '''Evaluate Similarity'''
        delta = [node_list[i].query_gradient(init, samples_user) for i in range(0, nodes)]
        delta = [[np.hstack(np.reshape(x, (-1, 1))) for x in m] for m in delta]
        delta = [np.concatenate(d, axis=0) for d in delta]
        var = [
            node_list[i].Local_variance(init, samples_user, delta, [x / divisor for x in samples_user])
            for i in
            range(0, nodes)]
        W = user_weighting(delta, nodes, fracs, samples_user, var, "PER", cluster_mode,
                           n_cluster)  # Weighting matrix


    msg = [node_list[i].local_train() for i in range(0, nodes)]  # Local updates
    weights = ([[w[0][w_ind] for w in msg] for w_ind in range(0, len(msg[0][0]))])

    node_ind = 0
    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[node_ind, :]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setting the local params

        # node.model.trainable_variables[-2].assign(embeddings[node_ind])
        node_ind = node_ind + 1
    return W

def Local_training( it=-1, node_list=[]):

    nodes = len(node_list)

    W = user_weighting(delta = 0, nodes = nodes, fracs = 0, samples_user = [], var = [], collab ="LOC", cluster_mode="off",
                           n_cluster = 0)  # Weighting matrix

    msg = [node_list[i].local_train() for i in range(0, nodes)]  # Local updates
    weights = ([[w[0][w_ind] for w in msg] for w_ind in range(0, len(msg[0][0]))])

    node_ind = 0
    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[node_ind, :]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setting the local params

        # node.model.trainable_variables[-2].assign(embeddings[node_ind])
        node_ind = node_ind + 1

def FedAvg_training( it=-1 , node_list=[], samples_user= []):

    nodes = len(node_list)

    W = user_weighting(delta = 0, nodes = nodes, fracs = 0, samples_user = samples_user, var = [], collab ="VAN", cluster_mode=False,
                           n_cluster = 0)  # Weighting matrix

    msg = [node_list[i].local_train() for i in range(0, nodes)]  # Local updates
    weights = ([[w[0][w_ind] for w in msg] for w_ind in range(0, len(msg[0][0]))])

    node_ind = 0
    for node in node_list:
        avg_w = [np.average(weights[w_ind], axis=0, weights=W[node_ind, :]) for w_ind in
                 range(0, len(weights))]  # Averaging local weights
        node.set_model_params(avg_w)  # Setting the local params

        node_ind = node_ind + 1
    return node_list

