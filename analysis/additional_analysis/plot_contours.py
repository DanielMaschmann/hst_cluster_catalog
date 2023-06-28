import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# load data
xbins = np.load('data/binx.npy')
ybins = np.load('data/biny.npy')

# load the raw color-color-points
color_vi_hum_class_1 = np.load('data/color_vi_hum_class_1.npy')
color_ub_hum_class_1 = np.load('data/color_ub_hum_class_1.npy')
color_vi_ml_class_1 = np.load('data/color_vi_ml_class_1.npy')
color_ub_ml_class_1 = np.load('data/color_ub_ml_class_1.npy')
color_vi_ml_mag_cut_class_1 = np.load('data/color_vi_ml_mag_cut_class_1.npy')
color_ub_ml_mag_cut_class_1 = np.load('data/color_ub_ml_mag_cut_class_1.npy')

# plot
fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(15, 6))
fontsize = 17

# contours levels
levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


def make_contours(axis, x, y):
    # exclude all bad values
    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))
    x = x[good_values]
    y = y[good_values]

    # create a representation of a kernel-density estimate using Gaussian kernels
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    axis.contour(xi, yi, zi, levels=levels, linewidths=(2,), origin='lower')


# plot contours
make_contours(axis=ax[0], x=color_vi_hum_class_1, y=color_ub_hum_class_1)
make_contours(axis=ax[1], x=color_vi_ml_class_1, y=color_ub_ml_class_1)
make_contours(axis=ax[2], x=color_vi_ml_mag_cut_class_1, y=color_ub_ml_mag_cut_class_1)

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
fig.savefig('plot_output/color_color_contours.png')


