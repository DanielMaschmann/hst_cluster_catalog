import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf, plotting_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from scipy.stats import gaussian_kde

import dust_tools.extinction_tools
from astropy.convolution import convolve

from photutils.segmentation import make_2dgaussian_kernel
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.simbad import Simbad

target = 'ngc0628'
galaxy_name = 'ngc0628'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)
phangs_photometry = photometry_tools.analysis_tools.AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')


simbad_table = Simbad.query_object(galaxy_name)
central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                               unit=(u.hourangle, u.deg))

cutout_size = (250, 250)


alma_hdu = fits.open('../overview_img/data/ngc0628_12m+7m+tp_co21_broad_tpeak.fits')
cutout_alma = hf.get_img_cutout(img=alma_hdu[0].data, wcs=WCS(alma_hdu[0].header),
                                         coord=central_coordinates, cutout_size=cutout_size)

# print(cutout_alma.data.shape)
#
# exit()
#
# band_list = ['F555W']
#
# phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=band_list, load_err=False)
#
#
# # get a cutout
# cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
#                                                                dec_cutout=central_coordinates.dec.to(u.deg),
#                                                                cutout_size=cutout_size, include_err=False,
#                                                                band_list=band_list)

# hst_wcs = cutout_dict_large_rgb['F555W_img_cutout'].wcs
# hst_shape = cutout_dict_large_rgb['F555W_img_cutout'].data.shape




# hst_img = plotting_tools.reproject_image(data=cutout_alma.data, wcs=cutout_alma.wcs,
#                                           new_wcs=hst_wcs,
#                                           new_shape=new_shape)





target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'darkorange'


target = 'ngc0628c'



cc_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')

ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')

# x_ml_12, y_ml_12 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')
# x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')


pos_ml_12 = cutout_alma.wcs.world_to_pixel(SkyCoord(ra=ra_ml_12*u.deg, dec=dec_ml_12*u.deg))
pos_ml_3 = cutout_alma.wcs.world_to_pixel(SkyCoord(ra=ra_ml_3*u.deg, dec=dec_ml_3*u.deg))


x_ml_12, y_ml_12 = pos_ml_12[0], pos_ml_12[1]
x_ml_3, y_ml_3 = pos_ml_3[0], pos_ml_3[1]


n_bins = 1000
kernal_std = 20

new_shape = cutout_alma.data.shape

x_bins_hist = np.linspace(0, new_shape[0], new_shape[0])
y_bins_hist = np.linspace(0, new_shape[1], new_shape[1])

hist_class_1 = np.histogram2d(x_ml_12[cc_ml_12 == 1], y_ml_12[cc_ml_12 == 1], bins=(x_bins_hist, y_bins_hist))[0]
hist_class_2 = np.histogram2d(x_ml_12[cc_ml_12 == 2], y_ml_12[cc_ml_12 == 2], bins=(x_bins_hist, y_bins_hist))[0]
hist_class_3 = np.histogram2d(x_ml_3, y_ml_3, bins=(x_bins_hist, y_bins_hist))[0]



hist_class_1 = hist_class_1 / np.sum(hist_class_1)
hist_class_2 = hist_class_2 / np.sum(hist_class_2)
hist_class_3 = hist_class_3 / np.sum(hist_class_3)

kernel = make_2dgaussian_kernel(kernal_std, size=51)  # FWHM = 3.0
# conv_hist_class_1 = convolve(hist_class_1, kernel)
# conv_hist_class_2 = convolve(hist_class_2, kernel)
# conv_hist_class_3 = convolve(hist_class_3, kernel)


grey_hst_r = mcf.greyRGBize_image(conv_hist_class_1, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_hst_g = mcf.greyRGBize_image(conv_hist_class_2, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
grey_hst_b = mcf.greyRGBize_image(conv_hist_class_3, rescalefn='asinh', scaletype='perc', min_max=[0.0, 99.9999],
                                  gamma=4.0, checkscale=False)
hst_r_purple = mcf.colorize_image(grey_hst_r, '#FF4433', colorintype='hex', gammacorr_color=2.8)
hst_g_orange = mcf.colorize_image(grey_hst_g, '#0FFF50', colorintype='hex', gammacorr_color=2.8)
hst_b_blue = mcf.colorize_image(grey_hst_b, '#0096FF', colorintype='hex', gammacorr_color=2.8)

class_1_image = mcf.combine_multicolor([hst_r_purple], gamma=2.8, inverse=False)
class_2_image = mcf.combine_multicolor([hst_g_orange], gamma=2.8, inverse=False)
class_3_image = mcf.combine_multicolor([hst_b_blue], gamma=2.8, inverse=False)

rgb_hst_image = mcf.combine_multicolor([hst_r_purple, hst_g_orange, hst_b_blue], gamma=2.8, inverse=False)


# fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 6))
# fontsize = 18

figure = plt.figure(figsize=(38, 10))
fontsize = 20
ax_img_class_1 = figure.add_axes([-0.30, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_class_2 = figure.add_axes([-0.05, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_class_3 = figure.add_axes([0.19, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)
ax_img_rgb = figure.add_axes([0.43, 0.07, 0.88, 0.88], projection=cutout_alma.wcs)

#
# class_1_image[class_1_image < 0.001] = 1.0
# class_2_image[class_2_image < 0.001] = 1.0
# class_3_image[class_3_image < 0.001] = 1.0
# rgb_hst_image[rgb_hst_image < 0.001] = 1.0



ax_img_class_1.imshow(class_1_image)
ax_img_class_2.imshow(class_2_image)
ax_img_class_3.imshow(class_3_image)

ax_img_rgb.imshow(rgb_hst_image)
ax_img_rgb.contour(cutout_alma.data, levels=[0.8, 1, 1.3, 1.6, 1.9], colors='white', alpha=0.5, linewidth=2)


ax_img_class_1.set_title('ML Class 1', fontsize=fontsize)
ax_img_class_2.set_title('ML Class 2', fontsize=fontsize)
ax_img_class_3.set_title('ML Compact Associations', fontsize=fontsize)

plotting_tools.arr_axis_params(ax=ax_img_class_1, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_class_2, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_class_3, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_rgb, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)


plt.savefig('plot_output/composit_spatial_dist.png')


exit()












exit(9)

y_bins_hist = np.linspace(y_lim[1], y_lim[0], n_bins)
gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

for color_index in range(len(x_data)):
    gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                    sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
    gauss_map += gauss
    if data_err[color_index] > noise_cut:
        noise_map += gauss




fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'




    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    detect_ub_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0))

    class_1_hum = cluster_class_hum_12 == 1
    class_2_hum = cluster_class_hum_12 == 2
    good_colors_hum = ((color_vi_hum_12 > -1.5) & (color_vi_hum_12 < 2.5) &
                       (color_ub_hum_12 > -3) & (color_ub_hum_12 < 1.5) &
                       detect_vi_hum_12 & detect_ub_hum_12)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))

    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    good_colors_ml = ((color_vi_ml_12 > -1.5) & (color_vi_ml_12 < 2.5) &
                   (color_ub_ml_12 > -3) & (color_ub_ml_12 < 1.5) &
                   detect_vi_ml_12 & detect_ub_ml_12)

    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_ub_ml_12[good_colors_ml], levels=None)


    ax[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_ub_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_ub_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=10)

    ax[row_index, col_index].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
                                      xytext=(vi_int, ub_int), fontsize=fontsize,
                                      textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

ax[0, 0].set_ylim(1.25, -2.8)
ax[0, 0].set_xlim(-1.0, 2.3)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1|2', ha='center', fontsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_panel_1.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_panel_1.pdf', bbox_inches='tight', dpi=300)


fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(20, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    detect_ub_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0))

    class_1_hum = cluster_class_hum_12 == 1
    class_2_hum = cluster_class_hum_12 == 2
    good_colors_hum = ((color_vi_hum_12 > -1.5) & (color_vi_hum_12 < 2.5) &
                       (color_ub_hum_12 > -3) & (color_ub_hum_12 < 1.5) &
                       detect_vi_hum_12 & detect_ub_hum_12)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))

    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    good_colors_ml = ((color_vi_ml_12 > -1.5) & (color_vi_ml_12 < 2.5) &
                   (color_ub_ml_12 > -3) & (color_ub_ml_12 < 1.5) &
                   detect_vi_ml_12 & detect_ub_ml_12)

    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_ub_ml_12[good_colors_ml], levels=None)


    ax[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_ub_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_ub_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=10)

    ax[row_index, col_index].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
                                      xytext=(vi_int, ub_int), fontsize=fontsize,
                                      textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

ax[0, 0].set_ylim(1.25, -2.8)
ax[0, 0].set_xlim(-1.0, 2.3)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1|2', ha='center', fontsize=fontsize)


ax[4, 3].scatter([], [], c=color_c1, s=30, label='Human class 1')
ax[4, 3].scatter([], [], c=color_c2, s=30, label='Human class 2')
ax[4, 3].plot([], [], color='k', label='ML')
ax[4, 3].legend(frameon=False, fontsize=fontsize)
ax[4, 3].axis('off')

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_panel_2.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_panel_2.pdf', bbox_inches='tight', dpi=300)


exit()




fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
fig_hum.set_size_inches(16, 18)
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
fig_ml.set_size_inches(16, 18)
fontsize = 18

row_index = 0
col_index = 0
for index in range(20, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_ml_12 == 1) & (cluster_class_qual_ml_12 >= 0.9)
    class_2_ml = (cluster_class_ml_12 == 2) & (cluster_class_qual_ml_12 >= 0.9)
    class_3_ml = (cluster_class_ml_3 == 3) & (cluster_class_qual_ml_3 >= 0.9)

    ax_hum[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_hum[row_index, col_index].scatter(color_vi_hum_3[cluster_class_hum_3 == 3],
                                         color_ub_hum_3[cluster_class_hum_3 == 3], c='royalblue', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 1],
                                         color_ub_hum_12[cluster_class_hum_12 == 1], c='forestgreen', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 2],
                                         color_ub_hum_12[cluster_class_hum_12 == 2], c='darkorange', s=1)

    ax_ml[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_ml[row_index, col_index].scatter(color_vi_ml_3[class_3_ml], color_ub_ml_3[class_3_ml], c='royalblue', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_1_ml], color_ub_ml_12[class_1_ml], c='forestgreen', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_2_ml], color_ub_ml_12[class_2_ml], c='darkorange', s=1)

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)

    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                            direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0


ax_hum[0, 0].set_ylim(1.25, -2.2)
ax_hum[0, 0].set_xlim(-1.0, 2.3)
fig_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_hum.text(0.5, 0.89, 'Class 1|2|3 Human', ha='center', fontsize=fontsize)
ax_hum[4, 3].axis('off')

ax_ml[0, 0].set_ylim(1.25, -2.2)
ax_ml[0, 0].set_xlim(-1.0, 2.3)
fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)
ax_ml[4, 3].axis('off')

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/ub_vi_hum_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_2.pdf', bbox_inches='tight', dpi=300)

