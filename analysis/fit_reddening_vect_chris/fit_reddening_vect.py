import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from astropy.convolution import convolve
from scipy.spatial import ConvexHull
import math
from astropy.table import QTable
from scipy import odr
from dust_extinction.parameter_averages import CCM89
import astropy.units as u


hst_wfc3_uvis1_bands_mean_wave = {
            'F218W': 2231.14,
            'FQ232N': 2327.12,
            'F225W': 2377.24,
            'FQ243N': 2420.59,
            'F275W': 2718.36,
            'F280N': 2796.98,
            'F300X': 2867.82,
            'F336W': 3365.86,
            'F343N': 3438.50,
            'F373N': 3730.19,
            'FQ378N': 3792.78,
            'FQ387N': 3873.61,
            'F390M': 3898.62,
            'F390W': 3952.50,
            'F395N': 3955.38,
            'F410M': 4109.81,
            'FQ422M': 4219.70,
            'F438W': 4338.57,
            'FQ436N': 4367.41,
            'FQ437N': 4371.30,
            'G280': 4628.43,
            'F467M': 4683.55,
            'F469N': 4688.29,
            'F475W': 4827.71,
            'F487N': 4871.54,
            'FQ492N': 4933.83,
            'F502N': 5009.93,
            'F475X': 5076.23,
            'FQ508N': 5091.59,
            'F555W': 5388.55,
            'F547M': 5459.04,
            'FQ575N': 5756.92,
            'F606W': 5999.27,
            'F200LP': 6043.00,
            'FQ619N': 6198.49,
            'F621M': 6227.39,
            'F625W': 6291.29,
            'F631N': 6304.27,
            'FQ634N': 6349.37,
            'F645N': 6453.59,
            'F350LP': 6508.00,
            'F656N': 6561.54,
            'F657N': 6566.93,
            'F658N': 6585.64,
            'F665N': 6656.23,
            'FQ672N': 6717.13,
            'FQ674N': 6730.58,
            'F673N': 6766.27,
            'F680N': 6880.13,
            'F689M': 6885.92,
            'FQ727N': 7275.84,
            'FQ750N': 7502.54,
            'F763M': 7623.09,
            'F600LP': 7656.67,
            'F775W': 7683.41,
            'F814W': 8117.36,
            'F845M': 8449.34,
            'FQ889N': 8892.56,
            'FQ906N': 9058.19,
            'F850LP': 9207.49,
            'FQ924N': 9247.91,
            'FQ937N': 9372.90,
            'F953N': 9531.11,
        }


def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

    model_ccm89 = CCM89(Rv=r_v)
    reddening1 = model_ccm89(wave1*u.micron) * r_v
    reddening2 = model_ccm89(wave2*u.micron) * r_v

    wave_v = 5388.55 * 1e-4
    reddening_v = model_ccm89(wave_v*u.micron) * r_v

    return (reddening1 - reddening2)*av/reddening_v


def plot_hst_reddening_vect(ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                        x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=False, fontsize=20, text_color='k', x_text_offset=0.1, y_text_offset=-0.3):

    nuv_wave = hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
    u_wave = hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
    b_wave = hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
    v_wave = hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
    i_wave = hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4

    x_wave_1 = locals()[x_color_1 + '_wave']
    x_wave_2 = locals()[x_color_2 + '_wave']
    y_wave_1 = locals()[y_color_1 + '_wave']
    y_wave_2 = locals()[y_color_2 + '_wave']

    color_ext_x = color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_val)
    color_ext_y = color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_val)

    slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

    angle_av_vector =  np.arctan(color_ext_y/color_ext_x) * 180/np.pi

    ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                xytext=(x_color_int, y_color_int), fontsize=fontsize,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

    if text:
        if isinstance(av_val, int):
            arrow_text = r'A$_{\rm V}$=%i mag' % av_val
        else:
            arrow_text = r'A$_{\rm V}$=%.1f mag' % av_val
        ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
                horizontalalignment='left', verticalalignment='bottom',
                transform_rotates_text=True, rotation_mode='anchor',
                rotation=angle_av_vector, fontsize=fontsize, color=text_color)


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


def seg2hull(seg_map, x_lim, y_lim, n_bins, seg_index=1, contour_index=0,
             save_str=None, x_label=None, y_label=None):
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh

    # kernel = make_2dgaussian_kernel(smooth_kernel, size=9)  # FWHM = 3.0
    # conv_gauss_map = convolve(gauss_map, kernel)
    # gauss_map_seg = conv_gauss_map.copy()
    # gauss_map_seg[seg_map._data != seg_index] = np.nan

    cs = plt.contour(x_bins_gauss, y_bins_gauss, (seg_map._data != seg_index), colors='darkgray', linewidth=2, levels=[0.01])
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

    x_convex_hull = counterclockwise_points[:, 0]
    y_convex_hull = counterclockwise_points[:, 1]
    x_convex_hull = np.concatenate([x_convex_hull, np.array([x_convex_hull[0]])])
    y_convex_hull = np.concatenate([y_convex_hull, np.array([y_convex_hull[0]])])

    if save_str is not None:
        table = QTable([x_convex_hull, y_convex_hull],  names=(x_label, y_label))
        table.write('data_output/convex_hull_%s.fits' % save_str, overwrite=True)

    return x_convex_hull, y_convex_hull


def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)


def lin_func(p, x):
    gradient, intersect = p
    return gradient*x + intersect


def fit_line(x_data, y_data, x_data_err, y_data_err):

    # Create a model for fitting.
    lin_model = odr.Model(lin_func)

    # Create a RealData object using our initiated data from above.
    data = odr.RealData(x_data, y_data, sx=x_data_err, sy=y_data_err)

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



age_mod_sol = np.load('data/age_mod_sol.npy')
model_ub_sol = np.load('data/model_ub_sol.npy')
model_vi_sol = np.load('data/model_vi_sol.npy')

color_vi_ml = np.load('data/color_vi_ml.npy')
color_ub_ml = np.load('data/color_ub_ml.npy')
color_vi_err_ml = np.load('data/color_vi_err_ml.npy')
color_ub_err_ml = np.load('data/color_ub_err_ml.npy')
detect_u_ml = np.load('data/detect_u_ml.npy')
detect_b_ml = np.load('data/detect_b_ml.npy')
detect_v_ml = np.load('data/detect_v_ml.npy')
detect_i_ml = np.load('data/detect_i_ml.npy')
clcl_color_ml = np.load('data/clcl_color_ml.npy')


# color range limitations
x_lim_vi = (-0.6, 1.9)
y_lim_ub = (0.9, -1.9)

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                            (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                            mask_detect_ubvi_ml)

# get gauss und segmentations
n_bins_ubvi = 120
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_ml_3 = calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

vi_convex_hull, ub_convex_hull = seg2hull(gauss_dict_ubvi_ml_3['seg_deb_map'], x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
         seg_index=1, contour_index=0, save_str=None, x_label=None, y_label=None)


hull_young_ml = ConvexHull(np.array([vi_convex_hull, ub_convex_hull]).T)
in_hull_young_ml = points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)

x_bins_vi = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_ubvi)
y_bins_ub = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_ubvi)
x_mesh_vi, y_mesh_ub = np.meshgrid(x_bins_vi, y_bins_ub)

gauss_map_ml_3_bkg = gauss_dict_ubvi_ml_3['gauss_map'].copy()
gauss_map_ml_3_young = gauss_dict_ubvi_ml_3['gauss_map'].copy()

in_hull_map_ml = np.array(points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_ub.flatten()]).T, hull_young_ml), dtype=bool)
in_hull_map_ml = np.reshape(in_hull_map_ml, newshape=(n_bins_ubvi, n_bins_ubvi))

gauss_map_ml_3_bkg[in_hull_map_ml] = np.nan
gauss_map_ml_3_young[np.invert(in_hull_map_ml)] = np.nan


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
fontsize = 23

vmax_ml_3 = np.nanmax(gauss_dict_ubvi_ml_3['gauss_map']) / 1.2
ax.imshow(gauss_map_ml_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_3)
ax.imshow(gauss_map_ml_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_3)

lin_fit_result_ubvi_c3_ml = fit_line(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml])

dummy_x_data = np.linspace(x_lim_vi[0], x_lim_vi[1], 100)

dummy_y_data_ubvi_c3_ml = lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']), x=dummy_x_data)

ax.plot(dummy_x_data, dummy_y_data_ubvi_c3_ml, color='k', linewidth=2, linestyle='--')

x_text_pos = 1.0

text_anle_c3_ml = np.arctan(lin_fit_result_ubvi_c3_ml['gradient']) * 180/np.pi

ax.text(x_text_pos-0.12,
              lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']),
                       x=x_text_pos)+0.05,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['gradient_err']),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=text_anle_c3_ml, fontsize=fontsize - 5)


ax.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}$')

vi_int = 1.1
ub_int = -1.6
av_value = 1

plot_hst_reddening_vect(ax=ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=True, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)

v_wave = hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=av_value)
max_color_ext_ub = color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=av_value)

slope_av_vector = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector = np.arctan(slope_av_vector) * 180/np.pi
print('slope_av_vector ', slope_av_vector)
print('angle_av_vector ', angle_av_vector)

ax.text(vi_int - 0.12, ub_int + 0.05,
              r'slope = %.2f' % slope_av_vector,
        horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
        rotation=angle_av_vector, fontsize=fontsize - 5)


ax.set_xlim(x_lim_vi)
ax.set_ylim(y_lim_ub)

ax.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax.legend(frameon=False, loc=3, bbox_to_anchor=(0, 0.05), fontsize=fontsize-3)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
plt.savefig('plot_output/fit_reddening_vect.png')



exit()
