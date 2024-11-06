import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('p528_results_Paraleliza1.csv')

# Define colors for specific heights
color_map = {
    1.5: '#ff0000',      # red
    15.0: '#ffa500',     # orange
    30.0: '#800080',     # purple
    60.0: '#008000',     # green
    1000.0: '#00ffff',   # cyan
    10000.0: '#800000'   # dark red
}

# Create plot
plt.figure(figsize=(12, 8))

# For each height, only plot the minimum loss at each distance
for height in sorted(df['h_1__meter'].unique()):
    # Group by distance and get the minimum loss value
    data = df[df['h_1__meter'] == height].groupby('d__km')['Perda_total_dB'].min().reset_index()
    data = data.sort_values('d__km')
    
    if height == 10000.0:
        label = f'h1 = {height/1000:.1f} km'
    else:
        label = f'h1 = {height:.1f} m'
    
    plt.plot(data['d__km'], -data['Perda_total_dB'], 
             label=label,
             color=color_map.get(height, 'black'),
             linewidth=1.5)

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Distance (km)')
plt.ylabel('Basic transmission loss (dB)')
plt.title('Basic Transmission Loss vs Distance')
plt.gca().invert_yaxis()
plt.xlim(0, 1000)
plt.ylim(-200, 100)
plt.legend()

# Add minor gridlines
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.minorticks_on()

plt.tight_layout()
plt.savefig('loss_plot.png', bbox_inches='tight', dpi=300)
plt.close()