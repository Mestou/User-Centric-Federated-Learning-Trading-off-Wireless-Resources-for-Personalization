import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper

matplotlib.use('Qt5Agg')
window_size=7


def savgol_filter(y, box_pts):
    y=np.concatenate((y, np.repeat(y[-1],box_pts-1)))
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

#
# algorithm_names = ["PER","LOC","VAN","Ditto","Fedprox","pFedME","FedFomo"]
#
# plt.figure(figsize=(10, 6))
#
# colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'gray','purple']
# markers = ['o', 's', '^', 'x', 'd', 'v', 'p','s']
# nodes = 35
# UL_DL_time_ratio = 4
# epsilon = 0.3
# # Loop over the algorithm names and load the files
# for n, color, marker in zip(algorithm_names, colors, markers):
#
#     if n == "FedFomo":
#         for i in [0]:
#             data = np.load(
#                 f'.\Results\Algorithm = FedFomo - dataset = Sentiment - clients = 35 - clustering = on (relevant for PER only) - rate_limited = False - M = {i} - run = 0.npy')
#             data = np.mean(np.mean(data, axis=0), axis=0)
#             x = np.arange(0, 150)
#             smoothed_y = savgol_filter(data, window_size, poly_order)
#             plt.plot(((35*nodes + 0*(nodes)) + nodes*UL_DL_time_ratio + 1)*x, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=f'{n}-{i}',
#                  markevery=4)
#     else:
# # Load the data for the current algorithm
#         if n == "PER":
#             data = np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
#
#             data = np.mean(np.mean(data, axis=0), axis=0)
#             # # Generate x-axis values based on the data length
#             x = np.arange(0, 150)
#             smoothed_y = savgol_filter(data, window_size, poly_order)
#             # # Plot the performance of the current algorithm
#             plt.plot((1*nodes + nodes*UL_DL_time_ratio + 1)*x, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4,
#                      label=n, markevery=4)
#         elif n=="LOC":
#             data = np.load(
#                 f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
#             data = np.mean(np.mean(data, axis=0), axis=0)
#             # # Generate x-axis values based on the data length
#             x = np.arange(0, 150)
#             smoothed_y = savgol_filter(data, window_size, poly_order)
#             # # Plot the performance of the current algorithm
#             plt.plot(( 1) * x, smoothed_y[:150], color=color, marker=marker, linestyle='-',
#                      linewidth=2, markersize=4, label=n, markevery=4)
#
#
#         else:
#             data = np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
#             data =  np.mean(np.mean(data, axis=0), axis=0)
#             # # Generate x-axis values based on the data length
#             x = np.arange(0, 150)
#             smoothed_y = savgol_filter(data, window_size, poly_order)
#             # # Plot the performance of the current algorithm
#             plt.plot((1 + nodes*UL_DL_time_ratio + 1)*x, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=n,markevery = 4)
#
# # Set the title, labels, and legend
# plt.title('Algorithms Performance Comparison')
# plt.xlabel('Seconds')
# plt.ylabel('Test Accuracy')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.xticks(np.arange(0, (max(((35*nodes + 0.3*(nodes)) + nodes*UL_DL_time_ratio)*x)), step=10000))
# plt.yticks(np.arange(0.3, 1, step=0.1))
# plt.legend(loc='upper left')
# plt.xlim([1000,(50000)])
# # Save the figure as an image (optional)
# plt.savefig('algorithm_performance.png', dpi=300)
#
# # Show the plot
# plt.show()
#


# for n in range(0,4):
#     W =  np.load(f'.\Results\W{n}.npy')

# plt.imshow(W)
# plt.show()
# plt.clf()





algorithm_names = ["PER","VAN","Ditto","Fedprox","pFedME","FedFomo"]
#
plt.figure(figsize=(8, 6))

colors = ['blue', 'green', 'black', 'orange', 'mediumpurple', 'goldenrod','purple']
markers = ['o', 'v', '<', 'o', 'v', '<','>']
nodes = 35
ul_dl_ratio = 4
epsilon = 0.3
# Loop over the algorithm names and load the files
for n, color, marker in zip(algorithm_names, colors, markers):
    i = 0

    if (n== "PER"):
        loss = []
        for j in range(0,1):
            loss.append(np.load(
            f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        # # Generate x -axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        # # Plot the performance of the current algorithm
        plt.plot((1*nodes + nodes+ 1)*x/100, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4,
                 label=n, markevery=4)


        data = np.load(
            f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = on (relevant for PER only) - run = 0.npy')
        i = 16
        data = np.mean(np.mean(data, axis=0), axis=0)
        # # Generate x-axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data, window_size)
        # # Plot the performance of the current algorithm
        plt.plot((1*16 + 16)*x/100, smoothed_y[:150], color="red", marker=">", linestyle='-', linewidth=2, markersize=4,
                 label=f"{n}-{i} streams", markevery=4)

    elif (n=="FedFomo"):
        i = 0
        loss = []
        for j in range(0,3):
            loss.append(np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - rate_limited = False - M = {i} - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        plt.plot((nodes*nodes + ul_dl_ratio*nodes + 1)*np.arange(0, 150)/100, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4,
                     label=n, markevery=1)
        i = 2
        loss = []
        for j in range(0, 3):
            loss.append(np.load(
                f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - rate_limited = True - M = {i} - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        plt.plot((i * nodes +  nodes ) * np.arange(0, 150)/100, smoothed_y[:150], color= "brown", marker=marker, linestyle='-', linewidth=2, markersize=4,
                 label=f"{n}-M={i}", markevery=4)
    # elif n == "LOC":
    #     data = np.load(
    #         f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
    #     data = np.mean(np.mean(data, axis=0), axis=0)
    #     # # Generate x-axis values based on the data length
    #     x = np.arange(0, 150)
    #     smoothed_y = savgol_filter(data, window_size)
    #     # # Plot the performance of the current algorithm
    #     plt.plot((1) * x/100, smoothed_y[:150], color=color, marker=marker, linestyle='-',
    #              linewidth=2, markersize=4, label=n, markevery=4)
    else:
        loss = []
        for j in range(0, 3):
            loss.append(np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = {j}.npy')[0,:,:150])
        data = np.mean(np.mean(loss, axis=1), axis=0)
        # # Generate x-axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data, window_size)
        # # Plot the performance of the current algorithm
        plt.plot((1 + nodes)*x/100, smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=n,markevery = 4)


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

plt.xlabel(r'Communication Cost [$\times$Model_size$\times$32/100]',fontsize=12)
plt.xlim([1,5000/100])
plt.ylim([0.2,0.95])
plt.ylabel('Validation Accuracy',fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('CommCostVsAcc.pdf')
plt.show()






algorithm_names = ["PER","VAN","Ditto","Fedprox","pFedME","FedFomo"]
#
plt.figure(figsize=(10, 8))

colors = ['blue', 'green', 'black', 'orange', 'mediumpurple', 'goldenrod','purple']
markers = ['o', 'v', '<', 'o', 'v', '<','>']
nodes = 35
ul_dl_ratio = 4
epsilon = 0.3
# Loop over the algorithm names and load the files
for n, color, marker in zip(algorithm_names, colors, markers):
    i = 0

    if (n== "PER"):
        loss = []
        for j in range(0,1):
            loss.append(np.load(
            f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        # # Generate x -axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        # # Plot the performance of the current algorithm
        plt.plot(np.arange(0, 300), smoothed_y[:300], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4,
                 label=n, markevery=4)


        data = np.load(
            f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = on (relevant for PER only) - run = 0.npy')
        i = 16
        data = np.mean(np.mean(data, axis=0), axis=0)
        # # Generate x-axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data, window_size)
        # # Plot the performance of the current algorithm
        plt.plot(np.arange(0, 150), smoothed_y[:150], color="red", marker=">", linestyle='-', linewidth=2, markersize=4,
                 label=f"{n}-{i} streams", markevery=4)

    elif (n=="FedFomo"):
        i = 2
        loss = []
        for j in range(0,1):
            loss.append(np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - rate_limited = False - M = {i} - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        plt.plot(np.arange(0, 300), smoothed_y[:300], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4,
                     label=n, markevery=4)
        i = 2
        loss = []
        for j in range(0, 3):
            loss.append(np.load(
                f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - rate_limited = True - M = {i} - run = {j}.npy'))
        data = np.mean(np.mean(loss, axis=0), axis=1)
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data[0], window_size)
        plt.plot(np.arange(0, 150), smoothed_y[:150], color= "brown", marker=marker, linestyle='-', linewidth=2, markersize=4,
                 label=f"{n}-M={i}", markevery=4)
    # elif n == "LOC":
    #     data = np.load(
    #         f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
    #     data = np.mean(np.mean(data, axis=0), axis=0)
    #     # # Generate x-axis values based on the data length
    #     x = np.arange(0, 150)
    #     smoothed_y = savgol_filter(data, window_size)
    #     # # Plot the performance of the current algorithm
    #     plt.plot((1) * x/100, smoothed_y[:150], color=color, marker=marker, linestyle='-',
    #              linewidth=2, markersize=4, label=n, markevery=4)
    else:
        loss = []
        for j in range(0, 3):
            loss.append(np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = {j}.npy')[0,:,:150])
        data = np.mean(np.mean(loss, axis=1), axis=0)
        # # Generate x-axis values based on the data length
        x = np.arange(0, 150)
        smoothed_y = savgol_filter(data, window_size)
        # # Plot the performance of the current algorithm
        plt.plot(np.arange(0, 150), smoothed_y[:150], color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=n,markevery = 4)


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

plt.xlabel(r'Communication Rounds',fontsize=12)
plt.xlim([1,150])
plt.ylim([0.2,0.95])
plt.ylabel('Validation Accuracy',fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('RoundVsAcc.pdf')
plt.show()




