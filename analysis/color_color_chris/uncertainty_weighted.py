import numpy as np
import matplotlib.pyplot as plt

from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
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


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)


def conv_gauss_map(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, kernal_std=1.0):

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)

    # get an empty 2d array
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for data_point_index in range(len(x_data)):
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[data_point_index], y0=y_data[data_point_index],
                        sig_x=x_data_err[data_point_index], sig_y=y_data_err[data_point_index])
        gauss_map += gauss

    kernel = make_2dgaussian_kernel(kernal_std, size=9)  # FWHM = 3.0
    conv_gauss_map = convolve(gauss_map, kernel)

    return gauss_map, conv_gauss_map


age_mod_sol = np.load('data/age_mod_sol.npy')
model_ub_sol = np.load('data/model_ub_sol.npy')
model_vi_sol = np.load('data/model_vi_sol.npy')

color_vi_hum = np.load('data/color_vi_hum.npy')
color_ub_hum = np.load('data/color_ub_hum.npy')
color_vi_err_hum = np.load('data/color_vi_err_hum.npy')
color_ub_err_hum = np.load('data/color_ub_err_hum.npy')
detect_u_hum = np.load('data/detect_u_hum.npy')
detect_b_hum = np.load('data/detect_b_hum.npy')
detect_v_hum = np.load('data/detect_v_hum.npy')
detect_i_hum = np.load('data/detect_i_hum.npy')
clcl_color_hum = np.load('data/clcl_color_hum.npy')

# color range limitations
x_lim_vi = (-0.6, 1.9)
y_lim_ub = (0.9, -2.2)

mask_class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                             (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)


# get gauss parameters
n_bins_ubvi = 120
kernal_std = 4.0

gauss_map, conv_gauss_map = conv_gauss_map(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data=color_ub_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data_err=color_ub_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              kernal_std=kernal_std)

figure = plt.figure(figsize=(20, 22))
fontsize = 38

ax_cc_reg = figure.add_axes([0.1, 0.08, 0.88, 0.88])

vmax = np.nanmax(gauss_map)/1.3
ax_cc_reg.imshow(gauss_map, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax)



ax_cc_reg.scatter([], [], color='white', label=r'N = %i' % (sum(mask_class_12_hum)))
ax_cc_reg.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=6, linestyle='-', label=r'BC03, Z$_{\odot}$')


vi_int = 1.2
ub_int = -1.6
av_value = 1

plot_hst_reddening_vect(ax=ax_cc_reg, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.00, y_text_offset=-0.05,
                                linewidth=4, line_color='k', text=True, fontsize=fontsize)

ax_cc_reg.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Human)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_cc_reg.set_title('The PHANGS-HST Bright Star Cluster Sample', fontsize=fontsize)

ax_cc_reg.set_xlim(x_lim_vi)
ax_cc_reg.set_ylim(y_lim_ub)

ax_cc_reg.text(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1]-0.15, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
ax_cc_reg.text(model_vi_sol[age_mod_sol == 5]+0.1, model_ub_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg.text(model_vi_sol[age_mod_sol == 10]+0.03, model_ub_sol[age_mod_sol == 10]-0.07, r'10 Myr',
                   horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_cc_reg.text(model_vi_sol[age_mod_sol == 102]-0.05, model_ub_sol[age_mod_sol == 102]+0.1, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg.text(model_vi_sol[age_mod_sol == 13750], model_ub_sol[age_mod_sol == 13750]+0.05, r'13 Gyr',
                   horizontalalignment='center', verticalalignment='top', fontsize=fontsize)



ax_cc_reg.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_reg.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_reg.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_reg.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.tight_layout()
plt.savefig('plot_output/reg_hum_c12.png')
plt.savefig('plot_output/reg_hum_c12.pdf')







#
#
#
#
# def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, threshold_fact=2):
#
#     # calculate combined errors
#     data_err = np.sqrt(x_data_err**2 + y_data_err**2)
#     noise_cut = np.percentile(data_err, 90)
#
#     # bins
#     x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
#     y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
#     # get a mesh
#     x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
#     gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
#     noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
#
#     for color_index in range(len(x_data)):
#         gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
#                         sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
#         gauss_map += gauss
#         if data_err[color_index] > noise_cut:
#             noise_map += gauss
#
#     gauss_map -= np.nanmean(noise_map)
#
#     kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
#
#     conv_gauss_map = convolve(gauss_map, kernel)
#     threshold = len(x_data) / threshold_fact
#     # threshold = np.nanmax(conv_gauss_map) / threshold_fact
#
#     seg_map = detect_sources(conv_gauss_map, threshold, npixels=20)
#     seg_deb_map = deblend_sources(conv_gauss_map, seg_map, npixels=20, nlevels=32, contrast=0.001, progress_bar=False)
#
#     return_dict = {
#         'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map, 'seg_deb_map': seg_deb_map}
#
#     return return_dict
#
#
#
# def plot_reg_map(ax, gauss_map, seg_map, x_lim, y_lim):
#
#     gauss_map_no_seg = gauss_map.copy()
#     gauss_map_seg1 = gauss_map.copy()
#     gauss_map_seg2 = gauss_map.copy()
#     gauss_map_seg3 = gauss_map.copy()
#     gauss_map_no_seg[seg_map._data != 0] = np.nan
#     gauss_map_seg1[seg_map._data != 1] = np.nan
#     gauss_map_seg2[seg_map._data != 2] = np.nan
#     gauss_map_seg3[seg_map._data != 3] = np.nan
#     ax.imshow(gauss_map_no_seg, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Greys', vmin=0, vmax=np.nanmax(gauss_map)/1.2)
#     ax.imshow(gauss_map_seg1, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Blues', vmin=0, vmax=np.nanmax(gauss_map)/1.2)
#     ax.imshow(gauss_map_seg2, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Greens', vmin=0, vmax=np.nanmax(gauss_map)/0.9)
#     ax.imshow(gauss_map_seg3, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
#                     cmap='Reds', vmin=0, vmax=np.nanmax(gauss_map)/1.4)
#
#
#
#     ax.set_xlim(x_lim)
#     ax.set_ylim(y_lim)
#
