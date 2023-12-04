import matplotlib.pyplot as plt
from statistics import mean

# Set the x-axis labels
x_labels = ['ResNet-18, REFINED: 1600', 'ResNet-18, IGTD: 1600']

#x1 = round(mean([5.47, 5.74, 6.32, 5.61, 5.88]), 2)
#x4 = round(mean([6.2, 6.44, 6.35, 5.93, 6.41]), 2)
#x7 = round(mean([6.33, 6.52, 6.36, 6.43, 6.31]), 2)
#x2 = round(mean([5.71, 5.41, 5.53, 5.57, 5.43]), 2)
#x5 = round(mean([6.32, 6.41, 6.42, 6.39, 6.31]), 2)
#x8 = round(mean([6.33, 6.25, 6.36, 6.31, 6.56]), 2)
#x3 = round(mean([5.41, 5.51, 5.39, 5.67, 5.56]), 2)
#x6 = round(mean([6.31, 6.37, 6.44, 6.35, 6.29]), 2)
#x9 = round(mean([6.32, 6.29, 6.31, 6.34, 6.27]), 2)

'''
x1 = round(mean([2.32, 2.44, 2.82, 2.36, 2.41]), 2)
x4 = round(mean([2.93, 3.00, 3.26, 2.91, 3.09]), 2)
x7 = round(mean([2.64, 2.77, 2.91, 2.70, 2.72]), 2)
x2 = round(mean([2.43, 2.22, 2.45, 2.56, 2.31]), 2)
x5 = round(mean([2.83, 2.91, 3.04, 2.95, 2.71]), 2)
x8 = round(mean([2.73, 2.64, 2.81, 2.64, 2.97]), 2)
x3 = round(mean([2.27, 2.37, 2.35, 2.69, 2.42]), 2)
x6 = round(mean([2.75, 2.87, 2.73, 2.91, 2.82]), 2)
x9 = round(mean([2.84, 2.82, 2.74, 2.81, 2.76]), 2)
'''

x1 = 1.67
x4 = 1.52
x2 = 1.69
x5 = 1.51
x3 = 1.67
x6 = 1.46


# Set the heights of the bars for each set
heights = [[x1, x4], [x2, x5], [x3, x6]]

# Set the width of the bars
bar_width = 0.1

# Create the figure and the axis
x_positions = [0.75, 1.25]

colors = ['#87ceeb', '#0000ff', '#00008b']
# Create the figure and the axis
fig, ax = plt.subplots()

# Iterate over the sets of bars
for i, height in enumerate(heights):
    # Plot the bars with the specified x-coordinates
    ax.bar(x=x_positions, height=height, width=bar_width, color=colors[i], alpha=0.5, label=f'Set {i+1}')
    x_positions = [x + bar_width for x in x_positions]
    
# Add the x-axis labels
ax.set_xticks([x - bar_width - 0.1 for x in x_positions])
ax.set_xticklabels(x_labels)

# Add a legend
#ax.legend(['Seed 1 R2', 'Seed 2 R2', 'Seed 3 R2'], bbox_to_anchor=(0.5, 1.15), loc='upper center').set_zorder(10)


# Show the plot
plt.show()