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



