"""
Tutorial to create heatmaps with plotted points outside a certain density threshold
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve


def density_with_points(ax, x, y, bin_x=None, bin_y=None, threshold=1, smoothing=2.0):
    """
    Function to create a heat map of dense data points according to a binning and a number threshold of data points.
    The data is further smoothed with a gaussian kernel
    Parameters
    ----------
    ax : ``matplotlib.axes.Axes``
    x : ``numpy.ndarray``
    y : ``numpy.ndarray``
    bin_x : ``numpy.ndarray``
    bin_y : ``numpy.ndarray``
    threshold : float
    smoothing : float
     :
    Returns
    -------
    None
    """
    if bin_x is None:
        bin_x = np.linspace(-1.5, 7.5, 130)
    if bin_y is None:
        bin_y = np.linspace(-1.5, 14.5, 130)

    # make sure that there are no nan values
    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    # create histogram for the data
    hist, x_edges, y_edges = np.histogram2d(x[good], y[good], bins=(bin_x, bin_y))

    # convolve the histogram
    kernel = Gaussian2DKernel(x_stddev=smoothing)
    hist = convolve(hist, kernel)

    # get a mask of pixels where there is an over density
    over_dense_regions = hist > threshold

    # loop over the pixels and mask the data points which not in pixels with a higher density
    mask_high_dens = np.zeros(len(x), dtype=bool)
    for x_index in range(len(x_edges)-1):
        for y_index in range(len(y_edges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = ((x > x_edges[x_index]) & (x < x_edges[x_index + 1]) &
                        (y > y_edges[y_index]) & (y < y_edges[y_index + 1]))
                mask_high_dens += mask

    # don't plot the pixels where there is not a high enough density
    hist[hist <= threshold] = np.nan
    # plot the pixels
    ax.imshow(hist.T, origin='lower',
              extent=(np.min(bin_x), np.max(bin_x), np.min(bin_y), np.max(bin_y)),
              cmap='inferno', interpolation='nearest')
    # plot the dataa point outside the high density region
    ax.scatter(x[np.invert(mask_high_dens)], y[np.invert(mask_high_dens)], color='k', marker='.')


# create 2D gaussian mock data
mu_x = 3
mu_y = 6

sigma_x = 1
sigma_y = 2

n_data_points = int(1e4)
# sample data
x_data = np.random.normal(loc=mu_x, scale=sigma_x, size=n_data_points)
y_data = np.random.normal(loc=mu_y, scale=sigma_y, size=n_data_points)

# plot the data points to know how it looks like
fig_first_look, ax_first_look = plt.subplots()
ax_first_look.scatter(x_data, y_data, marker='.', color='k')
ax_first_look.set_xlabel('x-data')
ax_first_look.set_ylabel('y-data')
fig_first_look.savefig('plot_output/first_data_inspection.png')
plt.close()

fig_heat_map, ax_heat_map = plt.subplots()
# creating the contours and data points. You can give the function also custom bins,
# thresholds for each bin and smoothing length
density_with_points(ax=ax_heat_map, x=x_data, y=y_data)
ax_heat_map.set_xlabel('x-data')
ax_heat_map.set_ylabel('y-data')
fig_heat_map.savefig('plot_output/heatmap_with_points.png')
plt.close()

print(' Congratulations! You created some awsome plots !!!')
