import numpy as np
import matplotlib.pyplot as plt

# load data
xbins = np.load('data/binx.npy')
ybins = np.load('data/biny.npy')
# get center of bins if needed
center_of_xbins = (xbins[:-1] + xbins[1:]) / 2
center_of_ybins = (ybins[:-1] + ybins[1:]) / 2
# load histograms (smoothed ones)
hist_hum_smoothed = np.load('data/hist_hum_smoothed.npy')
hist_ml_smoothed = np.load('data/hist_ml_smoothed.npy')
hist_ml_mag_cut_smoothed = np.load('data/hist_ml_mag_cut_smoothed.npy')

# plot
fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(15, 6))
fontsize = 17

# plot histograms
ax[0].imshow(hist_hum_smoothed.T, origin='lower', extent=(xbins.min(), xbins.max(), ybins.min(), ybins.max()),
             cmap='inferno', interpolation='nearest')
ax[1].imshow(hist_ml_smoothed.T, origin='lower', extent=(xbins.min(), xbins.max(), ybins.min(), ybins.max()),
             cmap='inferno', interpolation='nearest')
ax[2].imshow(hist_ml_mag_cut_smoothed.T, origin='lower', extent=(xbins.min(), xbins.max(), ybins.min(), ybins.max()),
             cmap='inferno', interpolation='nearest')

# add labels and other cosmetics for the plots
# title
ax[0].set_title('HUM', fontsize=fontsize)
ax[1].set_title('ML', fontsize=fontsize)
ax[2].set_title('ML mag-cut', fontsize=fontsize)
# axis labels
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
# tick parameters
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
# set limits (also to invert the y-axis)
ax[0].set_ylim(1.9, -2)
ax[0].set_xlim(-1.2, 2.4)

# final adjustment and save figure
fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig('plot_output/color_color_densities.png')


