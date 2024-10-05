import numpy as np
import matplotlib.pyplot as plt
print("In similarity.py")
# Load the 1D array from the .npy file
scores = np.load('scores.npy')

# Plot the array
plt.plot(np.arange(1,len(scores)+1),scores)
# plt.title('Similarity.npy')
plt.xlabel('Frame')
plt.ylabel('Similarity')
plt.xlim(0, len(scores))

# Set x-axis ticks labels every 10 units
plt.xticks(np.arange(0, len(scores), 10))  # Label ticks every 10 units
# Set x-axis ticks for grid lines every 1 unit
plt.gca().set_xticks(np.arange(0, len(scores), 1), minor=True)  # Minor ticks for the grid every 1 unit
# Turn the grid on and show grid lines for both major and minor ticks
plt.grid(True, which='both')  # 'both' makes grid for both major and minor ticks
plt.minorticks_on()  # Ensures minor ticks are shown
plt.show()
plt.savefig('scores.png')