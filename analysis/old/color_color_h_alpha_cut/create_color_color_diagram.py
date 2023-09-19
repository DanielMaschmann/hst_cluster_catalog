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
flux_f275w_sol = data_mod_sol['F275W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
mag_nuv_sol = hf.conv_mjy2vega(flux=flux_f275w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W',
                                                                                    mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W',
                                                                                  mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W',
                                                                                  mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W',
                                                                                  mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W',
                                                                                  mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))

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

# get the arrow
vi_int = 0.8
ub_int = -2.2
max_av = 1
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_bv = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=b_wave, wave2=v_wave, av=max_av)


color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'tab:gray'

cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=5, vmax=100)


extra_file_path = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/'


def plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=3, save_pdf=False):
    fig_1, ax_1 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fig_2, ax_2 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fontsize = 18

    row_index = 0
    col_index = 0
    for index in range(0, 20):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)

        color_x = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_y = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_x_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify='ml')
        color_y_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify='ml')
        cluster_id = catalog_access.get_hst_cc_phangs_id(target=target, classify=classify, cluster_class=cluster_class)

        h_alpha_intensity = np.zeros(len(cluster_id))
        hii_mask = np.zeros(len(cluster_id))
        if target == 'ngc0685':
            target_ext = 'ngc685'
        elif target == 'ngc0628c':
            target_ext = 'ngc628c'
        elif target == 'ngc0628e':
            target_ext = 'ngc628e'
        else:
            target_ext = target
        extra_file = extra_file_path + '%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
        if os.path.isfile(extra_file):
            extra_tab_hdu = fits.open(extra_file)
            extra_data_table = extra_tab_hdu[1].data
            extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
            extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
            extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev%i' % h_alpha_cut]

            for running_index, cluster_id in enumerate(cluster_id):
                index_extra_table = np.where(extra_cluster_id == cluster_id)
                h_alpha_intensity[running_index] = extra_h_alpha_intensity[index_extra_table]
                hii_mask[running_index] = extra_hii_mask[index_extra_table]
        else:
            print(extra_file, ' does not exist')

        h_alpha_mask = (h_alpha_intensity > h_alpha_cut) & (hii_mask > 0)

        # plot contours, arrow and model
        good_colors_ml_12 = (color_x_ml_12 > -5) & (color_x_ml_12 < 5) & (color_y_ml_12 > -5) & (color_y_ml_12 < 5)
        contours(ax=ax_1[row_index, col_index], x=color_x_ml_12[good_colors_ml_12], y=color_y_ml_12[good_colors_ml_12],
                 levels=None)
        # arrow
        ax_1[row_index, col_index].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub),
                                            xycoords='data', xytext=(vi_int, ub_int), fontsize=fontsize,
                                            textcoords='data',
                                            arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
        # model
        model_x = globals()['mag_%s_sol' % x_color[0]] - globals()['mag_%s_sol' % x_color[1]]
        model_y = globals()['mag_%s_sol' % y_color[0]] - globals()['mag_%s_sol' % y_color[1]]
        ax_1[row_index, col_index].plot(model_x, model_y, color='tab:red', linewidth=2, zorder=10)
        # data points
        ax_1[row_index, col_index].scatter(color_x[~h_alpha_mask], color_y[~h_alpha_mask],
                                           color='silver', s=10)
        ax_1[row_index, col_index].scatter(color_x[h_alpha_mask], color_y[h_alpha_mask],
                                           c=h_alpha_intensity[h_alpha_mask], norm=norm, cmap=cmap, s=30)

        if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
            anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                         loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
            ax_1[row_index, col_index].add_artist(anchored_left)
        else:
            anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                         loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
            ax_1[row_index, col_index].add_artist(anchored_left)

        ax_1[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                               direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0

    if y_color == 'ub':
        ax_1[0, 0].set_ylim(1.25, -2.8)
    elif y_color == 'bv':
        ax_1[0, 0].set_ylim(1.25, -0.8)
    ax_1[0, 0].set_xlim(-1.0, 2.3)
    fig_1.text(0.5, 0.08, '%s' % x_color.upper(), ha='center', fontsize=fontsize)
    fig_1.text(0.08, 0.5, '%s' % y_color.upper(), va='center', rotation='vertical', fontsize=fontsize)
    fig_1.text(0.5, 0.89, r'%s %s H$\alpha$ > %.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify.upper(), cluster_class.upper(), h_alpha_cut),
               ha='center', fontsize=fontsize)

    fig_1.subplots_adjust(wspace=0, hspace=0)

    fig_name_str = '%s_%s_panel_1_%s_c%s_h%s' % (x_color, y_color, classify, cluster_class[5:], h_alpha_cut)
    fig_1.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_1.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)

    row_index = 0
    col_index = 0
    for index in range(20, 39):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)

        color_x = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_y = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_x_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify='ml')
        color_y_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify='ml')
        cluster_id = catalog_access.get_hst_cc_phangs_id(target=target, classify=classify, cluster_class=cluster_class)

        h_alpha_intensity = np.zeros(len(cluster_id))
        hii_mask = np.zeros(len(cluster_id))
        if target == 'ngc0685':
            target_ext = 'ngc685'
        elif target == 'ngc0628c':
            target_ext = 'ngc628c'
        elif target == 'ngc0628e':
            target_ext = 'ngc628e'
        else:
            target_ext = target
        extra_file = extra_file_path + '%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
        if os.path.isfile(extra_file):
            extra_tab_hdu = fits.open(extra_file)
            extra_data_table = extra_tab_hdu[1].data
            extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
            extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
            extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev%i' % h_alpha_cut]

            for running_index, cluster_id in enumerate(cluster_id):
                index_extra_table = np.where(extra_cluster_id == cluster_id)
                h_alpha_intensity[running_index] = extra_h_alpha_intensity[index_extra_table]
                hii_mask[running_index] = extra_hii_mask[index_extra_table]
        else:
            print(extra_file, ' does not exist')

        h_alpha_mask = (h_alpha_intensity > h_alpha_cut) & (hii_mask > 0)

        # plot contours, arrow and model
        good_colors_ml_12 = (color_x_ml_12 > -5) & (color_x_ml_12 < 5) & (color_y_ml_12 > -5) & (color_y_ml_12 < 5)
        contours(ax=ax_2[row_index, col_index], x=color_x_ml_12[good_colors_ml_12], y=color_y_ml_12[good_colors_ml_12],
                 levels=None)
        # arrow
        ax_2[row_index, col_index].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub),
                                            xycoords='data', xytext=(vi_int, ub_int), fontsize=fontsize,
                                            textcoords='data',
                                            arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
        # model
        model_x = globals()['mag_%s_sol' % x_color[0]] - globals()['mag_%s_sol' % x_color[1]]
        model_y = globals()['mag_%s_sol' % y_color[0]] - globals()['mag_%s_sol' % y_color[1]]
        ax_2[row_index, col_index].plot(model_x, model_y, color='tab:red', linewidth=2, zorder=10)
        # data points
        ax_2[row_index, col_index].scatter(color_x[~h_alpha_mask], color_y[~h_alpha_mask],
                                           color='silver', s=10)
        ax_2[row_index, col_index].scatter(color_x[h_alpha_mask], color_y[h_alpha_mask],
                                           c=h_alpha_intensity[h_alpha_mask], norm=norm, cmap=cmap, s=30)

        if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
            anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                         loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
            ax_2[row_index, col_index].add_artist(anchored_left)
        else:
            anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                         loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
            ax_2[row_index, col_index].add_artist(anchored_left)

        ax_2[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                               direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0


    if y_color == 'ub':
        ax_2[0, 0].set_ylim(1.25, -2.8)
    elif y_color == 'bv':
        ax_2[0, 0].set_ylim(1.25, -0.8)
    ax_2[0, 0].set_xlim(-1.0, 2.3)
    fig_2.text(0.5, 0.08, '%s' % x_color.upper(), ha='center', fontsize=fontsize)
    fig_2.text(0.08, 0.5, '%s' % y_color.upper(), va='center', rotation='vertical', fontsize=fontsize)
    fig_2.text(0.5, 0.89, r'%s %s H$\alpha$ > %.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify.upper(), cluster_class.upper(), h_alpha_cut),
               ha='center', fontsize=fontsize)

    ax_2[4, 3].plot([], [], color='k', label='ML Class 1|2')
    ax_2[4, 3].legend(frameon=False, fontsize=fontsize)
    ax_2[4, 3].axis('off')

    fig_2.subplots_adjust(wspace=0, hspace=0)

    ax_cbar = fig_2.add_axes([0.73, 0.16, 0.15, 0.015])
    ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
    ax_cbar.set_xlabel(r'$\Sigma$ H$\alpha$ 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=0, fontsize=fontsize-3)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    fig_name_str = '%s_%s_panel_2_%s_c%s_h%s' % (x_color, y_color, classify, cluster_class[5:], h_alpha_cut)
    fig_2.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_2.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)


plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=2, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=3, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=5, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=10, save_pdf=False)

plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=2, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=3, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=5, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=10, save_pdf=False)



plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=2, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=3, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=5, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=10, save_pdf=False)

plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=2, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=3, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=5, save_pdf=False)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=10, save_pdf=False)






