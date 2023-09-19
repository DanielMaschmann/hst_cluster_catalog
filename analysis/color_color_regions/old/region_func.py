import numpy as np
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
import math
from scipy.spatial import ConvexHull
from scipy import odr
import photometry_tools
import dust_tools.extinction_tools


def sort_counterclockwise(points, centre = None):
  if centre:
    centre_x, centre_y = centre
  else:
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  return counterclockwise_points


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)


def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, threshold_fact=2, kernal_std=4.0, contrast=0.1):

    # calculate combined errors
    data_err = np.sqrt(x_data_err**2 + y_data_err**2)
    noise_cut = np.percentile(data_err, 90)

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
    noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for color_index in range(len(x_data)):
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                        sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
        gauss_map += gauss
        if data_err[color_index] > noise_cut:
            noise_map += gauss

    gauss_map -= np.nanmean(noise_map)

    kernel = make_2dgaussian_kernel(kernal_std, size=9)  # FWHM = 3.0

    conv_gauss_map = convolve(gauss_map, kernel)
    threshold = len(x_data) / threshold_fact
    # threshold = np.nanmax(conv_gauss_map) / threshold_fact

    seg_map = detect_sources(conv_gauss_map, threshold, npixels=50)
    seg_deb_map = deblend_sources(conv_gauss_map, seg_map, npixels=50, nlevels=32, contrast=contrast, progress_bar=False)
    numbers_of_seg = len(np.unique(seg_deb_map))
    return_dict = {
        'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map, 'seg_deb_map': seg_deb_map}

    return return_dict


def get_contours(ax, x_bins, y_bins, data_map, contour_index=0, save_str=None):
    cs = ax.contour(x_bins, y_bins, data_map, colors='darkgray', linewidth=2, levels=[0.01])
    p = cs.collections[0].get_paths()[contour_index]
    v = p.vertices
    # get all points from contour
    x_cont = []
    y_cont = []
    for point in v:
        x_cont.append(point[0])
        y_cont.append(point[1])

    x_cont = np.array(x_cont)
    y_cont = np.array(y_cont)
    counterclockwise_points = sort_counterclockwise(points=np.array([x_cont, y_cont]).T)

    counterclockwise_points = np.array(counterclockwise_points)

    if save_str is not None:

        x_convex_hull = counterclockwise_points[:, 0]
        y_convex_hull = counterclockwise_points[:, 1]

        np.save('data_output/x_convex_hull_%s.npy' % save_str, x_convex_hull)
        np.save('data_output/y_convex_hull_%s.npy' % save_str, y_convex_hull)

        # ax.scatter(x_convex_hull, y_convex_hull)


def plot_reg_map(ax, gauss_map, seg_map, x_lim, y_lim, n_bins,
                 color_1='Blues', color_2='Greens', color_3='Reds', color_4='Purples',
                 plot_cont_1=False, plot_cont_2=False, plot_cont_3=False, plot_cont_4=False,
                 save_str_1=None, save_str_2=None, save_str_3=None, save_str_4=None):

    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh

    kernel = make_2dgaussian_kernel(4.0, size=9)  # FWHM = 3.0
    conv_gauss_map = convolve(gauss_map, kernel)
    gauss_map_no_seg = conv_gauss_map.copy()
    gauss_map_seg1 = conv_gauss_map.copy()
    gauss_map_seg2 = conv_gauss_map.copy()
    gauss_map_seg3 = conv_gauss_map.copy()
    gauss_map_seg4 = conv_gauss_map.copy()
    gauss_map_no_seg[seg_map._data != 0] = np.nan
    gauss_map_seg1[seg_map._data != 1] = np.nan
    gauss_map_seg2[seg_map._data != 2] = np.nan
    gauss_map_seg3[seg_map._data != 3] = np.nan
    gauss_map_seg4[seg_map._data != 4] = np.nan

    if np.sum(seg_map._data == 0) > 0:
        ax.imshow(gauss_map_no_seg, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap='Greys', vmin=0, vmax=np.nanmax(conv_gauss_map)/1)
    if np.sum(seg_map._data == 1) > 0:
        ax.imshow(gauss_map_seg1, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_1, vmin=0, vmax=np.nanmax(gauss_map_seg1)/1)
        if plot_cont_1:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 1),
                         contour_index=0, save_str=save_str_1)
    if np.sum(seg_map._data == 2) > 0:
        ax.imshow(gauss_map_seg2, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_2, vmin=0, vmax=np.nanmax(gauss_map_seg2)/1)
        if plot_cont_2:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 2),
                         contour_index=0, save_str=save_str_2)
    if np.sum(seg_map._data == 3) > 0:
        ax.imshow(gauss_map_seg3, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_3, vmin=0, vmax=np.nanmax(gauss_map_seg3)/1)
        if plot_cont_3:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 3),
                         contour_index=0, save_str=save_str_3)
    if np.sum(seg_map._data == 4) > 0:
        ax.imshow(gauss_map_seg4, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_4, vmin=0, vmax=np.nanmax(gauss_map_seg3)/1)
        if plot_cont_4:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 4),
                         contour_index=0, save_str=save_str_4)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def lin_func(p, x):
    gradient, intersect = p
    return gradient*x + intersect


def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)


def fit_line(x_data, y_data, x_data_err, y_data_err, hull_str):
    x_convex_hull = np.load('data_output/x_convex_hull_%s.npy' % hull_str)
    y_convex_hull = np.load('data_output/y_convex_hull_%s.npy' % hull_str)

    hull = ConvexHull(np.array([x_convex_hull, y_convex_hull]).T)
    in_hull = points_in_hull(np.array([x_data, y_data]).T, hull)

    # Create a model for fitting.
    lin_model = odr.Model(lin_func)

    # Create a RealData object using our initiated data from above.
    data = odr.RealData(x_data[in_hull], y_data[in_hull], sx=x_data_err[in_hull], sy=y_data_err[in_hull])

    # Set up ODR with the model and data.
    odr_object = odr.ODR(data, lin_model, beta0=[0., 1.])

    # Run the regression.
    out = odr_object.run()

    # Use the in-built pprint method to give us results.
    # out.pprint()

    gradient, intersect = out.beta
    gradient_err, intersect_err = out.sd_beta

    return {
        'gradient': gradient,
        'intersect': intersect,
        'gradient_err': gradient_err,
        'intersect_err': intersect_err
    }


def plot_reddening_vect(ax, vi_int, ub_int, max_av, fontsize, linewidth=2):
    catalog_access = photometry_tools.data_access.CatalogAccess()

    v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
    i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
    u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
    b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
    max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
    max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)

    slope_av_vector = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
    angle_av_vector = - np.arctan(slope_av_vector) * 180/np.pi

    ax.annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
                xytext=(vi_int, ub_int), fontsize=fontsize,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=linewidth, ls='-'))

    ax.text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = %.1f' % max_av,
            horizontalalignment='left', verticalalignment='bottom',
            rotation=angle_av_vector, fontsize=fontsize)






