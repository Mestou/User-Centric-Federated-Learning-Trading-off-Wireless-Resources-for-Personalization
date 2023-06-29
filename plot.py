import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper
from scipy.signal import savgol_filter

matplotlib.use('Qt5Agg')
window_size=2
poly_order=2

algorithm_names = ["PER","LOC","Scaffold","VAN","Ditto","Fedprox","pFedME"]

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'gray']
markers = ['o', 's', '^', 'x', 'd', 'v', 'p']

# Loop over the algorithm names and load the files
for n, color, marker in zip(algorithm_names, colors, markers):
    # Load the data for the current algorithm
    data = np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - distribution = dirichlet - alpha = 0.4 - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
    data =  np.mean(np.mean(data, axis=0), axis=0)
    # Generate x-axis values based on the data length
    x = np.arange(len(data))
    smoothed_y = savgol_filter(data, window_size, poly_order)
    # Plot the performance of the current algorithm
    plt.plot(x, smoothed_y, color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=n,markevery = 4)

# Set the title, labels, and legend
plt.title('Algorithms Performance Comparison')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, len(data), step=10))
plt.yticks(np.arange(0, np.max(data), step=10))
plt.legend(loc='upper left')

# Save the figure as an image (optional)
plt.savefig('algorithm_performance.png', dpi=300)

# Show the plot
plt.show()




#
#
#
#
#
#
#
# ul_dl_ratio=4
# avg_del=1
# t_min=1
# avg_max_del=t_min+avg_del*np.sum(1/np.arange(1,100))
# fig = plt.figure(figsize=(5,5))
#
# def smooth(y, box_pts):
#     y=np.concatenate((y, np.repeat(y[-1],box_pts-1)))
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='valid')
#     return y_smooth
# i=0
# t_max=250
# for n in ["PER","LOC","Scaffold","VAN","Ditto"]:
#     min_loss=0
#
#     loss =np.load(
#         f'.\Results\Algorithm = {n} - dataset = Sentiment - distribution = dirichlet - alpha = 0.4 - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy',
#         allow_pickle=True)
#     min_loss = np.mean(np.squeeze(loss), axis=0)[0:t_max]
#     # print('FedAvg')
#     # print('% 2.1f' % min_loss + ')')
#     loss =  np.mean(np.mean(loss, axis=1), axis=0)
#     plt.plot(np.arange(0, t_max), loss[0:t_max], 'k:',label={n}, marker='o',markersize=4,markevery=5)
#
#
#     # loss = np.load(
#     #     f'Results/_n_cluster={n} - dataset = EMNIST- collab = PER- random-init = 66-clients = 100 - cluster_mode = on -.npy',
#     #     allow_pickle=True)
#     # min_loss = np.mean(np.max(np.min(np.squeeze(loss), axis=1)[0:t_max], axis=1)) * 100
#     # print(f'Proposed with {n}')
#     # print('% 2.1f' % min_loss + ')')
#     # loss = np.mean(np.mean(loss, axis=1), axis=0)
#     # plt.plot((1*ul_dl_ratio+n+avg_max_del)*np.arange(0,t_max),loss[0:t_max],label=f'Proposed w/ {n} models',color=[i/2,1-i/4,i/2],linestyle='--', marker='o',markersize=4,markevery=5)
#     # i=i+1
#
# plt.show()
# loss=np.load('Results/_n_cluster=1 - dataset = EMNIST- collab = ORACLE- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True)
# min_loss = np.mean(np.max(np.min(np.squeeze(loss), axis=1)[0:t_max], axis=1)) * 100
# print(f'Oracle')
# print('% 2.1f' % min_loss + ')')
# loss = np.mean(np.mean(loss, axis=1), axis=0)
# plt.plot((1*ul_dl_ratio+4+avg_max_del)*np.arange(0,t_max),loss[0:t_max],'k',label='Oracle', marker='v',markersize=4,markevery=5)
#
# loss=np.load('Results/FOMO_EMNIST_CovShift.npy', allow_pickle=True)
# min_loss = np.mean(np.max(np.min(np.squeeze(loss), axis=1)[0:t_max], axis=1)) * 100
# print(f'FOMO')
# print('% 2.1f' % min_loss + ')')
# loss = np.mean(np.mean(loss, axis=1), axis=0)
# plt.plot((1*ul_dl_ratio+25+avg_max_del)*np.arange(0,t_max),loss[0:t_max],color='tab:olive', label='FedFOMO',marker='s',markersize=4,markevery=5)
#
#
# data,comm_cost,min_loss=[],[],[]
# for n in range(0,3):
#     data.append(np.mean(np.load(f'Results/logger-{n}.npy', allow_pickle=True).item().acc_clients,axis=1))
#     min_loss.append(np.max(np.min(np.load(f'Results/logger-{n}.npy', allow_pickle=True).item().acc_clients, axis=1)[0:t_max], axis=0))
#     comm_cost.append([len(x) for x in np.load(f'Results/logger-{n}.npy', allow_pickle=True).item().clusters])
# loss = np.mean(data,axis=0)
#
# min_loss = np.mean(min_loss) * 100
# print(f'Clustered')
# print('% 2.1f' % np.mean(min_loss) + ')')
# comm_cost = np.cumsum(1*ul_dl_ratio+np.mean(comm_cost,axis=0)+avg_max_del)
# plt.plot(comm_cost[0:t_max],smooth(loss,10)[0:t_max],color='tab:blue',linestyle='-.',label='Clustered FL', marker='^',markersize=4,markevery=5)
# plt.xlabel(r'Time [s]',fontsize=12)
# plt.xlim([0,500])
# plt.ylim([0.50,0.85])
# plt.ylabel('Validation Accuracy',fontsize=12)
# plt.legend(fontsize=10)
# plt.grid()
# plt.tight_layout()
# plt.savefig('Scenario1.pdf')
# plt.show()
#
# fig = plt.figure(figsize=(5,5))
#
# i=0
# for n in [1,2,4,8]:
#
#     if (n==1):
#         loss = np.mean(np.load(
#             f'Results/_n_cluster=1 - dataset = EMNIST- collab = VAN- random-init = 66-clients = 100 - cluster_mode = on -.npy',
#             allow_pickle=True), axis=1)
#         loss = np.mean(loss, axis=0)
#         plt.plot(np.arange(0, t_max), loss[0:t_max], 'k:',label='FedAvg', marker='o',markersize=4,markevery=5)
#
#     else:
#         loss = np.mean(np.load(f'Results/_n_cluster={n} - dataset = EMNIST- collab = PER- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True),axis=1)
#         loss = np.mean(loss,axis=0)
#         plt.plot(np.arange(0,t_max),loss[0:t_max],label=f'Proposed w/ {n} models',color=[i/2,1-i/4,i/2],linestyle='--',  marker='o',markersize=4,markevery=5)
#         i=i+1
# loss = np.mean(np.mean(np.load('Results/_n_cluster=1 - dataset = EMNIST- collab = ORACLE- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True),axis=0),axis=0)
# plt.plot(np.arange(0,t_max),loss[0:t_max],'k',label='Oracle', marker='v',markersize=4,markevery=5)
# loss=np.load('Results/_n_cluster=1 - dataset = EMNIST- collab = LOC- random-init = 66-clients = 100 - cluster_mode = on -.npy', allow_pickle=True)
# min_loss = np.mean(np.max(np.min(np.squeeze(loss), axis=1)[0:t_max], axis=1)) * 100
# print(f'Local')
# print('% 2.1f' % min_loss + ')')
# loss = np.mean(np.mean(loss, axis=1), axis=0)
# plt.plot(np.arange(0,t_max)[0:t_max],loss[0:t_max],color='tab:red',linestyle=':', label='Local',marker='s',markersize=4,markevery=5)
#
# loss=np.load('Results/FOMO_EMNIST_CovShift.npy', allow_pickle=True)
# min_loss = np.mean(np.max(np.min(np.squeeze(loss), axis=1)[0:t_max], axis=1)) * 100
# print(f'FOMO')
# print('% 2.1f' % min_loss + ')')
# loss = np.mean(np.mean(loss, axis=1), axis=0)
# plt.plot(np.arange(0,t_max)[0:t_max],loss[0:t_max],color='tab:olive', label='FedFOMO',marker='s',markersize=4,markevery=5)
#
# data = np.load(f'Results/logger.npy', allow_pickle=True).item()
# data=[]
# for n in range(0,3):
#     data.append(np.mean(np.load(f'Results/logger-{n}.npy', allow_pickle=True).item().acc_clients,axis=1))
#     comm_cost = np.cumsum([len(x) for x in np.load(f'Results/logger-{n}.npy', allow_pickle=True).item().clusters])
# loss = np.mean(data,axis=0)
# plt.plot(smooth(loss,10)[0:t_max],color='tab:blue',linestyle='-.',label='Clustered FL', marker='^',markersize=4,markevery=5)
# plt.xlabel(r'Communication Round',fontsize=12)
#
# plt.xlim([0,100])
# plt.ylim([0.5,0.85])
# plt.ylabel('Validation Accuracy',fontsize=12)
# plt.legend(fontsize=10)
# plt.grid()
# plt.tight_layout()
# plt.savefig('CommVsAcc.pdf')
# plt.show()
