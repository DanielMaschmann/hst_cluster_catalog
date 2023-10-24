import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from scipy.stats import gaussian_kde
import matplotlib.mlab as mlab
import scipy.ndimage
import dust_tools.extinction_tools

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
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')
age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')



target_list = catalog_access.target_hst_cc
dist_list = []
delta_ms_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
    delta_ms_list.append(catalog_access.get_target_delta_ms(target=target))



sort = np.argsort(delta_ms_list)[::-1]
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]
delta_ms_list = np.array(delta_ms_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


color_c1 = 'tab:green'
color_c2 = 'mediumblue'
# color_c3 = 'darkorange'

vi_int = 1.2
ub_int = -1.9
nuvb_int = -3.0
av_value = 1

x_lim_vi = (-1.0, 2.3)
y_lim_ub = (1.25, -2.8)
y_lim_nuvb = (3.2, -4.3)


fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
    target = target_list[index]
    dist = dist_list[index]
    delta_ms = delta_ms_list[index]
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

    good_colors_hum = ((color_vi_hum_12 > (x_lim_vi[0] - 1)) & (color_vi_hum_12 < (x_lim_vi[1] + 1)) &
                                   (color_ub_hum_12 > (y_lim_ub[1] - 1)) & (color_ub_hum_12 < (y_lim_ub[0] + 1)) &
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
    good_colors_ml = ((color_vi_ml_12 > (x_lim_vi[0] - 1)) & (color_vi_ml_12 < (x_lim_vi[1] + 1)) &
                                   (color_ub_ml_12 > (y_lim_ub[1] - 1)) & (color_ub_ml_12 < (y_lim_ub[0] + 1)) &
                                   detect_vi_ml_12 & detect_ub_ml_12)
    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_ub_ml_12[good_colors_ml], levels=None)

    hf.display_models(ax=ax[row_index, col_index], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol,
                      age_sol=age_mod_sol, display_age_dots=False, color_sol='tab:red', linewidth_sol=3, size_age_dots=40,)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_ub_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=20, alpha=0.7)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_ub_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=20, alpha=0.7)

    if (row_index == 0) & (col_index == 0):
        text_flag = True
    else:
        text_flag = False

    hf.plot_reddening_vect(ax=ax[row_index, col_index], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=2, line_color='k', text=text_flag, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.2)


    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+ r'  ($\Delta$MS=%.2f)'%delta_ms +'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+ r'  ($\Delta$MS=%.2f)'%delta_ms +'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0


ax[0, 0].scatter([], [], c=color_c1, s=30, label='class 1 (Hum)')
ax[0, 0].scatter([], [], c=color_c2, s=30, label='class 2 (Hum)')
ax[0, 0].legend(frameon=True, ncols=3, bbox_to_anchor=[1.23, 1.16], fontsize=fontsize-6)

ax[0, 3].plot([], [], color='k', label='ML')
ax[0, 3].legend(frameon=True, ncols=1, bbox_to_anchor=[0.3, 1.16], fontsize=fontsize-6)

ax[0, 0].set_ylim(y_lim_ub)
ax[0, 0].set_xlim(x_lim_vi)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1+2 Clusters', ha='center', fontsize=fontsize)

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
    delta_ms = delta_ms_list[index]
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
    good_colors_hum = ((color_vi_hum_12 > (x_lim_vi[0] - 1)) & (color_vi_hum_12 < (x_lim_vi[1] + 1)) &
                       (color_ub_hum_12 > (y_lim_ub[1] - 1)) & (color_ub_hum_12 < (y_lim_ub[0] + 1)) &
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
    good_colors_ml = ((color_vi_ml_12 > (x_lim_vi[0] - 1)) & (color_vi_ml_12 < (x_lim_vi[1] + 1)) &
                       (color_ub_ml_12 > (y_lim_ub[1] - 1)) & (color_ub_ml_12 < (y_lim_ub[0] + 1)) &
                       detect_vi_ml_12 & detect_ub_ml_12)

    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_ub_ml_12[good_colors_ml], levels=None)

    hf.display_models(ax=ax[row_index, col_index], x_color_sol=model_vi_sol, y_color_sol=model_ub_sol,
                      age_sol=age_mod_sol, display_age_dots=False, color_sol='tab:red', linewidth_sol=3, size_age_dots=40,)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_ub_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=20, alpha=0.7)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_ub_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=20, alpha=0.7)

    if (row_index == 0) & (col_index == 0):
        text_flag = True
    else:
        text_flag = False
    hf.plot_reddening_vect(ax=ax[row_index, col_index], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=2, line_color='k', text=text_flag, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.2)


    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+ r'  ($\Delta$MS=%.2f)'%delta_ms +'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+ r'  ($\Delta$MS=%.2f)'%delta_ms +'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

ax[0, 0].set_ylim(y_lim_ub)
ax[0, 0].set_xlim(x_lim_vi)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1+2 Clusters', ha='center', fontsize=fontsize)


ax[4, 3].scatter([], [], c=color_c1, s=30, label='Human class 1')
ax[4, 3].scatter([], [], c=color_c2, s=30, label='Human class 2')
ax[4, 3].plot([], [], color='k', label='ML')
ax[4, 3].legend(frameon=False, fontsize=fontsize)
ax[4, 3].axis('off')

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_panel_2.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_panel_2.pdf', bbox_inches='tight', dpi=300)






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
    color_nuvb_hum_12 = catalog_access.get_hst_color_nuvb_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_nuvb_hum_3 = catalog_access.get_hst_color_nuvb_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    detect_nuvb_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0))

    class_1_hum = cluster_class_hum_12 == 1
    class_2_hum = cluster_class_hum_12 == 2

    good_colors_hum = ((color_vi_hum_12 > (x_lim_vi[0] - 1)) & (color_vi_hum_12 < (x_lim_vi[1] + 1)) &
                                   (color_nuvb_hum_12 > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum_12 < (y_lim_nuvb[0] + 1)) &
                                   detect_vi_hum_12 & detect_nuvb_hum_12)
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_nuvb_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))

    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    good_colors_ml = ((color_vi_ml_12 > (x_lim_vi[0] - 1)) & (color_vi_ml_12 < (x_lim_vi[1] + 1)) &
                                   (color_nuvb_ml_12 > (y_lim_nuvb[1] - 1)) & (color_nuvb_ml_12 < (y_lim_nuvb[0] + 1)) &
                                   detect_vi_ml_12 & detect_nuvb_ml_12)

    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_nuvb_ml_12[good_colors_ml], levels=None)


    ax[row_index, col_index].plot(model_vi_sol, model_nuvb_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_nuvb_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_nuvb_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=10)

    if (row_index == 0) & (col_index == 0):
        text_flag = True
    else:
        text_flag = False
    hf.plot_reddening_vect(ax=ax[row_index, col_index], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=3, line_color='k', text=text_flag, fontsize=fontsize)


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

ax[0, 0].set_ylim(y_lim_nuvb)
ax[0, 0].set_xlim(x_lim_vi)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'NUV (F275W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1|2', ha='center', fontsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/nuvb_vi_panel_1.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/nuvb_vi_panel_1.pdf', bbox_inches='tight', dpi=300)


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
    color_nuvb_hum_12 = catalog_access.get_hst_color_nuvb_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_nuvb_hum_3 = catalog_access.get_hst_color_nuvb_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))
    detect_nuvb_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F275W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0))

    class_1_hum = cluster_class_hum_12 == 1
    class_2_hum = cluster_class_hum_12 == 2
    good_colors_hum = ((color_vi_hum_12 > (x_lim_vi[0] - 1)) & (color_vi_hum_12 < (x_lim_vi[1] + 1)) &
                                   (color_nuvb_hum_12 > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum_12 < (y_lim_nuvb[0] + 1)) &
                                   detect_vi_hum_12 & detect_nuvb_hum_12)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    detect_nuvb_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') > 0) &
                        (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0))

    class_1_ml = cluster_class_ml_12 == 1
    class_2_ml = cluster_class_ml_12 == 2
    good_colors_ml = ((color_vi_ml_12 > (x_lim_vi[0] - 1)) & (color_vi_ml_12 < (x_lim_vi[1] + 1)) &
                      (color_nuvb_ml_12 > (y_lim_nuvb[1] - 1)) & (color_nuvb_ml_12 < (y_lim_nuvb[0] + 1)) &
                      detect_vi_ml_12 & detect_nuvb_ml_12)

    contours(ax=ax[row_index, col_index], x=color_vi_ml_12[good_colors_ml], y=color_nuvb_ml_12[good_colors_ml], levels=None)


    ax[row_index, col_index].plot(model_vi_sol, model_nuvb_sol, color='tab:red', linewidth=2, zorder=10)

    ax[row_index, col_index].scatter(color_vi_hum_12[class_1_hum * good_colors_hum],
                                     color_nuvb_hum_12[class_1_hum * good_colors_hum], c=color_c1, s=10)
    ax[row_index, col_index].scatter(color_vi_hum_12[class_2_hum * good_colors_hum],
                                     color_nuvb_hum_12[class_2_hum * good_colors_hum], c=color_c2, s=10)

    if (row_index == 0) & (col_index == 0):
        text_flag = True
    else:
        text_flag = False
    hf.plot_reddening_vect(ax=ax[row_index, col_index], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=3, line_color='k', text=text_flag, fontsize=fontsize)

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

ax[0, 0].set_ylim(y_lim_nuvb)
ax[0, 0].set_xlim(x_lim_vi)
fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'NUV (F275W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.5, 0.89, 'Class 1|2', ha='center', fontsize=fontsize)


ax[4, 3].scatter([], [], c=color_c1, s=30, label='Human class 1')
ax[4, 3].scatter([], [], c=color_c2, s=30, label='Human class 2')
ax[4, 3].plot([], [], color='k', label='ML')
ax[4, 3].legend(frameon=False, fontsize=fontsize)
ax[4, 3].axis('off')

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/nuvb_vi_panel_2.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/nuvb_vi_panel_2.pdf', bbox_inches='tight', dpi=300)


