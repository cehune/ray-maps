import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("build/figure4a.csv")

# normalize x to [0, 1] param — x goes from -40 to 0
df['param'] = (df['px'] - df['px'].min()) / (df['px'].max() - df['px'].min())

fig, ax = plt.subplots(figsize=(8, 4))
interior = df[df['param'] < 0.8]
df['raymap_norm']    = df['raymap']    / interior['raymap'].mean()
df['photonmap_norm'] = df['photonmap'] / interior['photonmap'].mean()

ax.plot(df['param'], df['raymap_norm'],    color='gold', linewidth=2, label='Ray map')
ax.plot(df['param'], df['photonmap_norm'], color='red',  linewidth=2, label='Photon map')
ax.axhline(1.0, color='black', linestyle='dotted', linewidth=1, label='Reference')

# reference — mean of interior points away from boundary
interior = df[df['param'] < 0.8]
df['raymap_norm']    = df['raymap']    / interior['raymap'].mean()
df['photonmap_norm'] = df['photonmap'] / interior['photonmap'].mean()

ax.plot(df['param'], df['raymap_norm'],    color='gold', linewidth=2, label='Ray map')
ax.plot(df['param'], df['photonmap_norm'], color='red',  linewidth=2, label='Photon map')
ax.axhline(1.0, color='black', linestyle='dotted', linewidth=1, label='Reference')
ax.set_xlabel('Path position (interior → boundary)')
ax.set_ylabel('Irradiance estimate')
ax.set_title('Figure 4(a) reproduction — Boundary bias at convex corner')
ax.legend()
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('figure4a.png', dpi=150)
plt.show()