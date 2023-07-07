import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper
from scipy.signal import savgol_filter

matplotlib.use('Qt5Agg')
window_size=3
poly_order=2

algorithm_names = ["PER","LOC","Scaffold","VAN","Ditto","Fedprox","pFedME","FedFomo"]

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'gray','purple']
markers = ['o', 's', '^', 'x', 'd', 'v', 'p','s']

# Loop over the algorithm names and load the files
for n, color, marker in zip(algorithm_names, colors, markers):

    if n == "FedFomo":
        for i in [0]:
            data = np.load(
                f'.\Results\Algorithm = FedFomo - dataset = Sentiment - clients = 35 - clustering = on (relevant for PER only) - rate_limited = False - M = {i} - run = 0.npy')
            data = np.mean(np.mean(data, axis=0), axis=0)
            x = np.arange(len(data))
            smoothed_y = savgol_filter(data, window_size, poly_order)
            plt.plot(x, smoothed_y, color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=f'{n}-{i}',
                 markevery=4)
    else:
# Load the data for the current algorithm
        data = np.load(f'.\Results\Algorithm = {n} - dataset = Sentiment - clients = 35 - clustering = off (relevant for PER only) - run = 0.npy')
        data =  np.mean(np.mean(data, axis=0), axis=0)
        # # Generate x-axis values based on the data length
        x = np.arange(len(data))
        smoothed_y = savgol_filter(data, window_size, poly_order)
        # # Plot the performance of the current algorithm
        plt.plot(x, smoothed_y, color=color, marker=marker, linestyle='-', linewidth=2, markersize=4, label=n,markevery = 4)

# Set the title, labels, and legend
plt.title('Algorithms Performance Comparison')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, len(data), step=10))
plt.yticks(np.arange(0, 1, step=0.1))
plt.legend(loc='upper left')

# Save the figure as an image (optional)
plt.savefig('algorithm_performance.png', dpi=300)

# Show the plot
plt.show()



# for n in range(0,4):
#     W =  np.load(f'.\Results\W{n}.npy')

# plt.imshow(W)
# plt.show()
# plt.clf()