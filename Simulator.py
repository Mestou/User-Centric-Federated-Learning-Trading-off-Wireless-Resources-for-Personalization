import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
from functions import user_weighting, Ditto_training, Fedprox_training, pFedMe_Training, Scaffold_training, FedPer_training, FedAvg_training, Local_training, Fomo_training
from user import FedUser
import tensorflow as tf
from pathlib import Path
import os


seed =225
np.random.seed(seed)
tf.random.set_seed(seed)
class simulator_:
    def __init__(self, params, train_data, test_data, shape, samples_user, fracs, num_classes,run):
        self.run = run
        self.params = params
        self.modes = self.params.get('data').get('Algorithm') # list ["VAN","Scaffold","Fedprox","Ditto","VAN","LOC","ORACLE"],
        self.dataset = self.params.get('data').get('dataset')
        self.global_c = []
        self.optimizer = self.params.get('model').get('optimizer').get('type')
        self.nodes= self.params.get('simulation').get('n_clients')
        self.epochs =  self.params.get('model').get('epochs')
        self.n_it = 3  # n of local iteration at each node between two subsequent . comm. rounds
        self.eta_x =  self.params.get('model').get('optimizer').get('local_lr') # Local Learning Rate
        self.batchsize = self.params.get('model').get('batch_size')
        self.train_data = train_data
        self.test_data= test_data
        self.shape = shape
        self.samples_user = samples_user
        self.fracs = fracs
        self.cluster_mode = self.params.get("data").get("PER").get("cluster_mode")
        self.num_classes = num_classes
        self.model_id = params.get("model").get("id")
        self.iter = self.params.get("simulation").get("n_rounds") #Number of Federated rounds
    def start(self):

        tr_x, tr_y, te_x, te_y = self.train_data[0],self.train_data[1],self.test_data[0],self.test_data[1]

        dummy_node = FedUser(0, self.fracs[0, :], self.shape, np.max(self.num_classes),
                             [tr_x[0 % self.nodes], tr_y[0 % self.nodes]], [te_x[0 % self.nodes], te_y[0 % self.nodes]],
                             self.batchsize, self.nodes, self.eta_x, self.n_it, self.epochs,self.optimizer,self.model_id)

        for Training_Modes in self.modes:

            print("Training_Algorithm = ", Training_Modes)
            log = []
            self.global_c = []
            loc_val_old = []
            P = np.ones([self.nodes,self.nodes])
            msg_old = []
            accuracy=[]
            '''Create Nodes'''
            node_list = [FedUser(i, self.fracs[i, :], self.shape, np.max(self.num_classes),
                                 [tr_x[i % self.nodes], tr_y[i % self.nodes]],
                                 [te_x[i % self.nodes], te_y[i % self.nodes]], self.batchsize, self.nodes, self.eta_x,
                                 self.n_it, self.epochs,self.optimizer,self.model_id) for i in
                         range(0, self.nodes)]
            '''Initialize the nodes to the same model params'''
            [node.initialize(dummy_node.get_model_params()) for node in node_list]

            W = user_weighting(delta = [], nodes=self.nodes, fracs= [], samples_user=self.samples_user, var = [], collab="VAN",cluster_mode = "OFF",n_cluster = None)  # Weighting matrix

            for it in range(0,self.iter):

                if(Training_Modes == 'Scaffold'):
                    self.global_c = Scaffold_training(node_list=node_list, global_c =self.global_c,samples_user=self.samples_user)

                elif(Training_Modes == 'pFedMe'):
                    # set mu and R, K(local refinement iterations), local and global learning rates
                    mu = self.params.get("data").get("pFedMe").get("mu")
                    R = self.params.get("data").get("pFedMe").get("R")
                    K = self.params.get("data").get("pFedMe").get("K")
                    local_etax = self.params.get("data").get("pFedMe").get("local_etax")
                    global_etax = self.params.get("data").get("pFedMe").get("global_etax")

                    pFedMe_Training(mu=mu, R=R, K=K, local_etax=local_etax, global_etax=global_etax, node_list=node_list, samples_user=self.samples_user)

                elif(Training_Modes == 'Fedprox'):
                    #set mu and gamma
                    mu = self.params.get("data").get("Fedprox").get("mu")
                    gamma = self.params.get("data").get("Fedprox").get("gamma")

                    Fedprox_training(mu=mu, gamma = gamma, node_list=node_list, it=it, samples_user=self.samples_user)

                elif(Training_Modes == 'Ditto'):
                    # set mu
                    mu = self.params.get("data").get("Ditto").get("mu")
                    Ditto_training(mu=mu, node_list=node_list, it=it, samples_user=self.samples_user)

                elif(Training_Modes == 'PER'):
                    if it==0:
                       divisor = self.params.get("data").get("PER").get("divisor")
                       cluster_mode = self.params.get("data").get("PER").get("cluster_mode")

                       if cluster_mode == "on":
                           print('Clustering mode is ON')

                       W = FedPer_training( W = [], it=it, node_list=node_list, init=dummy_node.get_model_params(),
                                        divisor=divisor, fracs=self.fracs,
                                        samples_user=self.samples_user, cluster_mode=cluster_mode, n_cluster=None)
                    else:

                       _ = FedPer_training( W=W, it=it, node_list=node_list, init=dummy_node.get_model_params(), divisor=divisor, fracs=self.fracs,
                                    samples_user=self.samples_user, cluster_mode=cluster_mode, n_cluster=None)

                elif (Training_Modes == "FedFomo"):
                    M = self.params.get("data").get("FedFomo").get("M")
                    epsilon = self.params.get("data").get("FedFomo").get("epsilon")
                    rate_limited = self.params.get("data").get("FedFomo").get("rate_limited")
                    loc_val_old,msg_old,P = Fomo_training(M, epsilon, P, self.samples_user, self.nodes, node_list, msg_old, loc_val_old, it,rate_limited)
                elif (Training_Modes == 'LOC'):
                    Local_training(it=it, node_list=node_list)
                    
                else:
                    FedAvg_training( it= it  , node_list=node_list, samples_user = self.samples_user)

                for node in node_list:
                    if (Training_Modes == 'Ditto' or Training_Modes == 'pFedMe' or Training_Modes == 'Fedprox'):
                        accuracy.append(1 - node.local_test_my())
                    else:
                        accuracy.append(1 - node.local_test())

                if it % 1 == 0:
                    print('--------------Iteration: '+str(it)+' --------------' + 'avg = ' + str(np.average([round(accuracy[len(accuracy)-self.nodes+i],2) for i in range(0,self.nodes)])))
                    print('Test/Accuracy: '+str([round(accuracy[len(accuracy)-self.nodes+i],2) for i in range(0,self.nodes)]))


            accuracy=[accuracy[i:(np.asarray(accuracy)).shape[0]:self.nodes] for i in range(0,self.nodes)]
            log.append(np.asarray(accuracy))

            directory = "Results"
            parent_dir = Path(__file__).parent.resolve()
            path = os.path.join(parent_dir, directory)

            file_name = F"Algorithm = {Training_Modes} - dataset = {self.dataset} - clients = {self.nodes} - clustering = {self.cluster_mode} (relevant for PER only) - run = {self.run}.npy"
            if Training_Modes == "FedFomo":
                file_name = F"Algorithm = {Training_Modes} - dataset = {self.dataset} - clients = {self.nodes} - clustering = {self.cluster_mode} (relevant for PER only) - rate_limited = {rate_limited} - M = {M} - run = {self.run}.npy"

            path = os.path.join(path,file_name)
            np.save(path, log, allow_pickle=True)
