import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Example data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Creating the plot with a triangle marker
plt.plot(x, y, '^', label='Data Label')

# Customizing the legend to not display the marker
legend_without_symbol = []
for label in ['Data Label']:
    legend_without_symbol.append(mlines.Line2D([], [], color='black', marker='', linestyle='-', label=label))
plt.legend(handles=legend_without_symbol)
# plt.legend()

# Display the plot
plt.show()
