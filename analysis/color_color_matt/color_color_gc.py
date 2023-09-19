import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse


def contours(ax, x, y, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    origin = 'lower'
    cs = ax.contour(xi, yi, zi, levels=levels,
                    linewidths=(2,),
                    origin=origin)

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)


def density_with_points(ax, x, y, binx=None, biny=None, threshold=1, kernel_std=2.0, plot_scatter=True):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))


    kernel = Gaussian2DKernel(x_stddev=kernel_std)
    hist = convolve(hist, kernel)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask

    hist[hist <= threshold] = np.nan
    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap='inferno',
              interpolation='nearest', aspect='auto'
              )
    if plot_scatter:
        ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.')
    ax.set_ylim(ax.get_ylim()[::-1])


# load model tracks
# solar metallicity
model_ub_sol = np.load('data/model_ub_sol.npy')
model_vi_sol = np.load('data/model_vi_sol.npy')
# 1/50 sol met
model_ub_sol50 = np.load('data/model_ub_sol50.npy')
model_vi_sol50 = np.load('data/model_vi_sol50.npy')

# load the colors of all clusters
color_vi_hum = np.load('data/color_vi_hum.npy')
color_ub_hum = np.load('data/color_ub_hum.npy')
# load masks of all clusters which are detected
detect_u_hum = np.load('data/detect_u_hum.npy')
detect_b_hum = np.load('data/detect_b_hum.npy')
detect_v_hum = np.load('data/detect_v_hum.npy')
detect_i_hum = np.load('data/detect_i_hum.npy')
# load array with the classes
clcl_color_hum = np.load('data/clcl_color_hum.npy')
mask_class_1_hum = clcl_color_hum == 1

# load the colors of all clusters
color_vi_ml = np.load('data/color_vi_ml.npy')
color_ub_ml = np.load('data/color_ub_ml.npy')
# load masks of all clusters which are detected
detect_u_ml = np.load('data/detect_u_ml.npy')
detect_b_ml = np.load('data/detect_b_ml.npy')
detect_v_ml = np.load('data/detect_v_ml.npy')
detect_i_ml = np.load('data/detect_i_ml.npy')
# load array with the classes
clcl_color_ml = np.load('data/clcl_color_ml.npy')
mask_class_1_ml = clcl_color_ml == 1

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum

# define limits for plotting limits
x_lim_vi = (-0.7, 2.4)
y_lim_ub = (2.1, -2.2)
# binning for the heat maps
n_bins = 190
kernal_std = 3.0

# mask of color points which make sense
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_hum)

# plot densitiy plot
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
fontsize = 17

# plot the density contours
density_with_points(ax=ax, x=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                    y=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std,
                    plot_scatter=True)

# plot model tracks
ax.plot(model_vi_sol, model_ub_sol, color='green', linewidth=4, linestyle='-', zorder=10)
ax.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=4, linestyle='--', zorder=10)

ax.set_title('Class 1', fontsize=fontsize)

ax.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax.set_xlim(x_lim_vi)

ax.set_ylim(y_lim_ub)

ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
fig.savefig('plot_output/color_color_c1_heat_map.png')
fig.savefig('plot_output/color_color_c1_heat_map.pdf')
fig.clf()
plt.cla()


# plot contours
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
fontsize = 17

# plot the density contours
contours(ax=ax, x=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
         y=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum])

# plot model tracks
ax.plot(model_vi_sol, model_ub_sol, color='green', linewidth=4, linestyle='-', zorder=10)
ax.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=4, linestyle='--', zorder=10)

ax.set_title('Class 1', fontsize=fontsize)

ax.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax.set_xlim(x_lim_vi)

ax.set_ylim(y_lim_ub)

ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
fig.savefig('plot_output/color_color_c1_contour.png')
fig.savefig('plot_output/color_color_c1_contour.pdf')
fig.clf()
plt.cla()

