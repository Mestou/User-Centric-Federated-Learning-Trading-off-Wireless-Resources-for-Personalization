import numpy as np
import matplotlib.pyplot as plt
import helper

#plt.rcParams.update({
 #   "text.usetex": True,
  #  "font.family": "sans-serif",
   # "font.sans-serif": ["Helvetica"]})
#Silhouette Score Plot

def smooth(y, box_pts):
    y=np.concatenate((y, np.repeat(y[-1],box_pts-1)))
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth



fig = plt.figure(figsize=(6,6))
# # s_scores=np.load('L_n_cluster=20 - dataset = EMNIST- collab = LOC- random-init = 66-clients = 20 - cluster_mode = OFF -.npy', allow_pickle=True)
# s_sc=np.load('All collaborate.npy', allow_pickle=True)
# VAN=np.load('V_n_cluster=20 - dataset = EMNIST- collab = VAN- random-init = 66-clients = 20 - cluster_mode = OFF - (1).npy', allow_pickle=True)
# PER=np.load('P_n_cluster=20 - dataset = EMNIST- collab = PER- random-init = 66-clients = 20 - cluster_mode = OFF -.npy', allow_pickle=True)
#
# plt.plot(np.arange(1,101),smooth(np.mean(np.mean(s_scores,axis=0),axis=0),5),'b',marker = '*',markersize=8,markevery=5,label='Local')
# #plt.axvline(x=4, color='g', linewidth=1.5)
#
#
# #plt.xticks(list(plt.xticks()[0]) + [4])
# #plt.xlim([0,41])
# plt.ylabel('Accuracy',fontsize=12)
# plt.xlabel('Communication Rounds',fontsize=12)
# plt.grid()
# plt.ylim([0.5,0.85])
# plt.xlim([0,99])
# plt.plot(np.arange(1,100),smooth(np.mean(np.mean(s_sc,axis=0),axis=0),5), 'r',marker='v',markersize=4,markevery=5,label='proposed: NxN collab')
# plt.plot(np.arange(1,101),smooth(np.mean(np.mean(VAN,axis=0),axis=0),5), 'k',marker='o',markersize=4,markevery=5,label='Vanilla FL')
# plt.plot(np.arange(1,101),smooth(np.mean(np.mean(PER,axis=0),axis=0),5), 'g',marker='^',markersize=4,markevery=5,label='proposed: 20 streams')
#
# plt.legend(fontsize=8)
# plt.savefig('NxNN.pdf')

# plt.show()

ul_dl_ratio=1
algorithm_names = ["PER","LOC","VAN","Ditto","Fedprox","pFedME","FedFomo"]

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 's', '^', 'x', 'd', 'v', 'p']

i=0
for n, color, marker in zip(algorithm_names, colors, markers):

    if (n!="PER" and n!="FedFomo"):
        data = np.load(
            f'.\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
        data = np.mean(np.mean(data, axis=0), axis=0)
        plt.plot((1+ul_dl_ratio)*np.arange(0, 250), data, color = color,label=n, marker=marker,markersize=4,markevery=5)
    elif (n == "PER"):
        loss = np.mean(np.load( f'.\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy', allow_pickle=True),axis=1)
        loss = np.mean(loss,axis=0)
        plt.plot((1+ul_dl_ratio*35)*np.arange(0,250),loss,label=n,color=[i/3,1-i/3,i/3], marker='o',markersize=4,markevery=5)
        i=i+1
    else:
        loss = np.mean(np.load( f'.\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy', allow_pickle=True),axis=1)
        loss = np.mean(loss,axis=0)
        plt.plot((1+ul_dl_ratio*35*35)*np.arange(0,250),loss,label=n,color=[i/3,1-i/3,i/3], marker='o',markersize=4,markevery=5)
        i=i+1

# loss = np.mean(np.mean(np.load('_n_cluster=1 - dataset = EMNIST- collab = ORACLE- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True),axis=0),axis=0)
# plt.plot((1+ul_dl_ratio*4)*np.arange(0,150),loss,'k--',label='Oracle', marker='v',markersize=4,markevery=5)
# data,comm_cost=[],[]
# for n in range(0,3):
#     data.append(np.mean(np.load(f'logger-{n}.npy', allow_pickle=True).item().acc_clients,axis=1))
#     comm_cost.append([len(x) for x in np.load(f'logger-{n}.npy', allow_pickle=True).item().clusters])
# loss = np.mean(data,axis=0)
# comm_cost = np.cumsum(1+ul_dl_ratio*np.mean(comm_cost,axis=0))
# plt.plot(comm_cost[0:150],smooth(loss,5),'b-.',label='Clustered FL', marker='^',markersize=4,markevery=5)
# loss = np.load("FOMO_EMNIST_CovShift.npy",allow_pickle = True)
# loss = np.mean(loss,axis = 1)
# loss = np.mean(loss,axis=0)
# plt.plot((1+ul_dl_ratio*10)*np.arange(0,150),smooth(loss,1),'b-.',label='FedFomo',color = "red", marker='^',markersize=4,markevery=5)

plt.xlabel(r'Communication Cost [32$\times$model size bits]',fontsize=12)
plt.xlim([0,500])
plt.ylim([0.1,0.85])
plt.ylabel('Validation Accuracy',fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('CommCostVsAcc.pdf')
plt.show()

#
# i=0
# for n in [1,2,4,8,16]:
#
#     if (n==1):
#         loss = np.mean(np.load(
#             f'_n_cluster=1 - dataset = EMNIST- collab = VAN- random-init = 66-clients = 100 - cluster_mode = on -.npy',
#             allow_pickle=True), axis=1)
#         loss = np.mean(loss, axis=0)
#         plt.plot(np.arange(0, 150), loss, 'k:',label='FedAvg', marker='o',markersize=4,markevery=5)
#
#     else:
#         loss = np.mean(np.load(f'_n_cluster={n} - dataset = EMNIST- collab = PER- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True),axis=1)
#         loss = np.mean(loss,axis=0)
#         plt.plot(np.arange(0,150),loss,label=f'Proposed w/ {n} models',color=[i/3,1-i/3,i/3], marker='o',markersize=4,markevery=5)
#         i=i+1
# loss = np.mean(np.mean(np.load('_n_cluster=1 - dataset = EMNIST- collab = ORACLE- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True),axis=0),axis=0)
# plt.plot(np.arange(0,150),loss,'k--',label='Oracle', marker='v',markersize=4,markevery=5)
# data = np.load(f'logger.npy', allow_pickle=True).item()
# data=[]
# for n in range(0,3):
#     data.append(np.mean(np.load(f'logger-{n}.npy', allow_pickle=True).item().acc_clients,axis=1))
#     comm_cost = np.cumsum([len(x) for x in np.load(f'logger-{n}.npy', allow_pickle=True).item().clusters])
# loss = np.mean(data,axis=0)
# plt.plot(smooth(loss,5),'b-.',label='Clustered FL', marker='^',markersize=4,markevery=5)
# plt.xlabel(r'Communication Cost [32$\times$model size bits]',fontsize=12)
# plt.xlim([0,150])
# plt.ylim([0.5,0.85])
# plt.ylabel('Validation Accuracy',fontsize=12)
# plt.legend(fontsize=12)
# plt.grid()
# plt.tight_layout()
# plt.savefig('CommVsAcc.pdf')
# plt.show()