import numpy as np
import matplotlib.pyplot as plt

from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)


def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, kernal_std=4.0):

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
    # noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for color_index in range(len(x_data)):
        x_err = np.sqrt(x_data_err[color_index]**2 + 0.01**2)
        y_err = np.sqrt(y_data_err[color_index]**2 + 0.01**2)
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                        sig_x=x_err, sig_y=y_err)
        gauss_map += gauss

    kernel = make_2dgaussian_kernel(kernal_std, size=9)
    conv_gauss_map = convolve(gauss_map, kernel)
    return {'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map}


# load the data of all galaxies
target_name_hum = np.load('data_output/target_name_hum.npy')
index_hum = np.load('data_output/index_hum.npy')
phangs_cluster_id_hum = np.load('data_output/phangs_cluster_id_hum.npy')
cluster_class_hum = np.load('data_output/cluster_class_hum.npy')
color_color_class_hum = np.load('data_output/color_color_class_hum.npy')
color_vi_hum_vega = np.load('data_output/color_vi_hum_vega.npy')
color_ub_hum_vega = np.load('data_output/color_ub_hum_vega.npy')
color_bv_hum_vega = np.load('data_output/color_bv_hum_vega.npy')
color_nuvb_hum_vega = np.load('data_output/color_nuvb_hum_vega.npy')
color_vi_err_hum_vega = np.load('data_output/color_vi_err_hum_vega.npy')
color_ub_err_hum_vega = np.load('data_output/color_ub_err_hum_vega.npy')
color_bv_err_hum_vega = np.load('data_output/color_bv_err_hum_vega.npy')
color_nuvb_err_hum_vega = np.load('data_output/color_nuvb_err_hum_vega.npy')
color_vi_hum_ab = np.load('data_output/color_vi_hum_ab.npy')
color_ub_hum_ab = np.load('data_output/color_ub_hum_ab.npy')
color_bv_hum_ab = np.load('data_output/color_bv_hum_ab.npy')
color_nuvb_hum_ab = np.load('data_output/color_nuvb_hum_ab.npy')
color_vi_err_hum_ab = np.load('data_output/color_vi_err_hum_ab.npy')
color_ub_err_hum_ab = np.load('data_output/color_ub_err_hum_ab.npy')
color_bv_err_hum_ab = np.load('data_output/color_bv_err_hum_ab.npy')
color_nuvb_err_hum_ab = np.load('data_output/color_nuvb_err_hum_ab.npy')
detect_nuv_hum = np.load('data_output/detect_nuv_hum.npy')
detect_u_hum = np.load('data_output/detect_u_hum.npy')
detect_b_hum = np.load('data_output/detect_b_hum.npy')
detect_v_hum = np.load('data_output/detect_v_hum.npy')
detect_i_hum = np.load('data_output/detect_i_hum.npy')
age_hum = np.load('data_output/age_hum.npy')
ebv_hum = np.load('data_output/ebv_hum.npy')
mass_hum = np.load('data_output/mass_hum.npy')
ra_hum = np.load('data_output/ra_hum.npy')
dec_hum = np.load('data_output/dec_hum.npy')
x_hum = np.load('data_output/x_hum.npy')
y_hum = np.load('data_output/y_hum.npy')

#load the UBVI hulls

ycl_hull = np.genfromtxt('ubvi_hulls/hlsp_phangs-cat_hst_multi_hull_multi_v1_ycl-human-ubvi.txt')
vi_color_ycl_hull = ycl_hull[:, 0]
ub_color_ycl_hull = ycl_hull[:, 1]

map_hull = np.genfromtxt('ubvi_hulls/hlsp_phangs-cat_hst_multi_hull_multi_v1_map-human-ubvi.txt')
vi_color_map_hull = map_hull[:, 0]
ub_color_map_hull = map_hull[:, 1]

ogcc_hull = np.genfromtxt('ubvi_hulls/hlsp_phangs-cat_hst_multi_hull_multi_v1_ogcc-human-ubvi.txt')
vi_color_ogcc_hull = ogcc_hull[:, 0]
ub_color_ogcc_hull = ogcc_hull[:, 1]


x_lim_vi = (-0.4, 1.6)
y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (1.1, -2.2)
y_lim_bv = (1.9, -0.7)

n_bins = 190
kernal_std = 3.0

# get some quality cuts
mask_class_12_hum = (cluster_class_hum == 1) | (cluster_class_hum == 2)
mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_good_colors_ubvi_hum = ((color_vi_hum_vega > (x_lim_vi[0] - 1)) & (color_vi_hum_vega < (x_lim_vi[1] + 1)) &
                             (color_ub_hum_vega > (y_lim_ub[1] - 1)) & (color_ub_hum_vega < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)

# get gauss und segmentations
n_bins_ubvi = 90
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_hum_12 = calc_seg(x_data=color_vi_hum_vega[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data=color_ub_hum_vega[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_data_err=color_vi_err_hum_vega[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data_err=color_ub_err_hum_vega[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              kernal_std=kernal_std)


figure = plt.figure(figsize=(20, 22))
fontsize = 38

ax_cc_reg_hum = figure.add_axes([0.1, 0.08, 0.88, 0.88])

vmax = np.nanmax(gauss_dict_ubvi_hum_12['gauss_map'])
ax_cc_reg_hum.imshow(gauss_dict_ubvi_hum_12['gauss_map'], origin='lower',
                     extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax/10, vmax=vmax/1.1)

ax_cc_reg_hum.plot(vi_color_ycl_hull, ub_color_ycl_hull, color='blue', linewidth=3)
ax_cc_reg_hum.plot(vi_color_map_hull, ub_color_map_hull, color='green', linewidth=3)
ax_cc_reg_hum.plot(vi_color_ogcc_hull, ub_color_ogcc_hull, color='red', linewidth=3)


ax_cc_reg_hum.set_title('The PHANGS-HST Bright Star Cluster Sample', fontsize=fontsize)

ax_cc_reg_hum.set_xlim(x_lim_vi)
ax_cc_reg_hum.set_ylim(y_lim_ub)


ax_cc_reg_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_reg_hum.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.savefig('plot_output/class_12_ubvi_gaus_map.png')
























