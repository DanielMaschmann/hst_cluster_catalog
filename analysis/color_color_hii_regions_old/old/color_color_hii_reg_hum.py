import os.path

import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
import dust_tools.extinction_tools
from matplotlib.colorbar import ColorbarBase
import matplotlib
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import astropy.units as u


def contours(ax, x, y, levels=None, axis_offse=(-0.2, 0.1, -0.55, 0.6)):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]

    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min()+axis_offse[0]:x.max()+axis_offse[1]:x.size**0.5*1j,
             y.min()+axis_offse[2]:y.max()+axis_offse[3]:y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    # ax[0].scatter(xi.flatten(), yi.flatten(), c=zi)
    cs = ax.contour(xi, yi, zi, levels=levels,
                    colors='k',
                    linewidths=(1,),
                    origin='lower')


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)


# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol


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

# reddening vector
vi_int = 0.8
ub_int = -2.2
max_av = 2
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_vi_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av+0.1)
max_color_ext_ub_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av+0.1)



# color mask
cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=5, vmax=100)


fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    class_3_ml = cluster_class_ml_3 == 3

    color_ub_ml = np.concatenate([color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = np.concatenate([color_vi_ml_12, color_vi_ml_3])

    if target == 'ngc0628c':
        target_hii = 'ngc628c'
    elif target == 'ngc0628e':
        target_hii = 'ngc628e'
    else:
        target_hii = target
    hii_mask_file_name = '/home/benutzer/data/PHANGS_products/NB_HIIregions/v0/HII_masks/%s_NB_HII_mask.fits' % target_hii.upper()
    if os.path.isfile(hii_mask_file_name) & (target != 'ngc4254') & (target != 'ngc4298'):
        hii_reg_tab_hdu = fits.open(hii_mask_file_name)
        hii_reg_data_table = hii_reg_tab_hdu[0].data

        wcs = WCS(hii_reg_tab_hdu[0].header)
        cc_coords_world = SkyCoord(ra=ra_hum_12*u.deg, dec=dec_hum_12*u.deg)
        cc_coords_pix = wcs.world_to_pixel(cc_coords_world)
        x_values = np.array(cc_coords_pix[0], dtype=int)
        y_values = np.array(cc_coords_pix[1], dtype=int)
        mask_hii_reg = hii_reg_tab_hdu[0].data[y_values, x_values] > 0
    else:
        mask_hii_reg = np.zeros(len(color_vi_hum_12), dtype=bool)

    good_colors = (color_vi_hum_12 > -1.5) & (color_vi_hum_12 < 2.5) & (color_ub_hum_12 > -3) & (color_ub_hum_12 < 1.5)

    contours(ax=ax[row_index, col_index], x=color_vi_hum_12[good_colors], y=color_ub_hum_12[good_colors], levels=None)

    ax[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 1) & ~mask_hii_reg],
                                     color_ub_hum_12[(cluster_class_hum_12 == 1) & ~mask_hii_reg],
                                     c='silver', s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 2) & ~mask_hii_reg],
                                         color_ub_hum_12[(cluster_class_hum_12 == 2) & ~mask_hii_reg],
                                     c='silver', s=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 2) & mask_hii_reg],
                                         color_ub_hum_12[(cluster_class_hum_12 == 2) & mask_hii_reg],
                                     c=color_c2, s=30)
    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 1) & mask_hii_reg],
                                     color_ub_hum_12[(cluster_class_hum_12 == 1) & mask_hii_reg],
                                     c=color_c1, s=30)

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
fig.savefig('plot_output/ub_vi_hii_reg_hum_1.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_hii_reg_hum_1.pdf', bbox_inches='tight', dpi=300)


fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
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
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    class_3_ml = cluster_class_ml_3 == 3

    color_ub_ml = np.concatenate([color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = np.concatenate([color_vi_ml_12, color_vi_ml_3])

    if target == 'ngc0628c':
        target_hii = 'ngc628c'
    elif target == 'ngc0628e':
        target_hii = 'ngc628e'
    else:
        target_hii = target
    hii_mask_file_name = '/home/benutzer/data/PHANGS_products/NB_HIIregions/v0/HII_masks/%s_NB_HII_mask.fits' % target_hii.upper()
    if os.path.isfile(hii_mask_file_name) & (target != 'ngc4254') & (target != 'ngc4298') & (target != 'ngc1087') & (target != 'ngc1512') & (target != 'ngc1300'):
        hii_reg_tab_hdu = fits.open(hii_mask_file_name)
        hii_reg_data_table = hii_reg_tab_hdu[0].data

        wcs = WCS(hii_reg_tab_hdu[0].header)
        cc_coords_world = SkyCoord(ra=ra_hum_12*u.deg, dec=dec_hum_12*u.deg)
        cc_coords_pix = wcs.world_to_pixel(cc_coords_world)
        x_values = np.array(cc_coords_pix[0], dtype=int)
        y_values = np.array(cc_coords_pix[1], dtype=int)
        mask_hii_reg = hii_reg_tab_hdu[0].data[y_values, x_values] > 0
    else:
        mask_hii_reg = np.zeros(len(color_vi_hum_12), dtype=bool)

    good_colors = (color_vi_hum_12 > -1.5) & (color_vi_hum_12 < 2.5) & (color_ub_hum_12 > -3) & (color_ub_hum_12 < 1.5)

    contours(ax=ax[row_index, col_index], x=color_vi_hum_12[good_colors], y=color_ub_hum_12[good_colors], levels=None)

    ax[row_index, col_index].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 1) & ~mask_hii_reg],
                                     color_ub_hum_12[(cluster_class_hum_12 == 1) & ~mask_hii_reg],
                                     c='silver', s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 2) & ~mask_hii_reg],
                                         color_ub_hum_12[(cluster_class_hum_12 == 2) & ~mask_hii_reg],
                                     c='silver', s=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 2) & mask_hii_reg],
                                         color_ub_hum_12[(cluster_class_hum_12 == 2) & mask_hii_reg],
                                     c=color_c2, s=30)
    ax[row_index, col_index].scatter(color_vi_hum_12[(cluster_class_hum_12 == 1) & mask_hii_reg],
                                     color_ub_hum_12[(cluster_class_hum_12 == 1) & mask_hii_reg],
                                     c=color_c1, s=30)


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

ax_cbar = fig.add_axes([0.73, 0.1, 0.11, 0.015])
ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_xlabel(r'$\Sigma$ H$\alpha$ 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)




fig.savefig('plot_output/ub_vi_hii_reg_hum_2.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_hii_reg_hum_2.pdf', bbox_inches='tight', dpi=300)

