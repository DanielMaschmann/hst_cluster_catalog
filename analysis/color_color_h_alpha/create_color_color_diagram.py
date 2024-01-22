import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
from matplotlib.colorbar import ColorbarBase
import matplotlib
from matplotlib.lines import Line2D


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
delta_ms_list = []
mass_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
    delta_ms_list.append(catalog_access.get_target_delta_ms(target=target))
    mass_list.append(catalog_access.get_target_mstar(target=target))
sort = np.argsort(delta_ms_list)[::-1]
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]
delta_ms_list = np.array(delta_ms_list)[sort]
mass_list = np.array(mass_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

vi_int = 1.2
ub_int = -1.7
nuvb_int = -3.0
av_value = 1

x_lim_vi = (-1.0, 2.3)
y_lim_ub = (1.25, -2.8)
y_lim_nuvb = (3.2, -4.3)

color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'tab:gray'

cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=5, vmax=100)


def plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=1,
                   save_pdf=False):

    fig_1, ax_1 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fig_2, ax_2 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fontsize = 18

    row_index = 0
    col_index = 0
    for index in range(0, 20):
        target = target_list[index]
        dist = dist_list[index]
        delta_ms = delta_ms_list[index]
        mass = mass_list[index]
        print('target ', target, 'dist ', dist)
        color_x = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_y = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_x_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify='ml')
        color_y_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify='ml')

        h_alpha_medsub = catalog_access.get_h_alpha_medsub(target=target, classify=classify,
                                                               cluster_class=cluster_class)
        hii_mask = catalog_access.get_h_alpha_threshold_mask(target=target, classify=classify,
                                                             cluster_class=cluster_class, h_alpha_cut=h_alpha_cut)

        h_alpha_mask = (h_alpha_medsub > h_alpha_cut) & (hii_mask > 0)

        # plot contours, arrow and model
        good_colors_ml_12 = (color_x_ml_12 > -5) & (color_x_ml_12 < 5) & (color_y_ml_12 > -5) & (color_y_ml_12 < 5)
        contours(ax=ax_1[row_index, col_index], x=color_x_ml_12[good_colors_ml_12], y=color_y_ml_12[good_colors_ml_12],
                 levels=None)

        if (row_index == 0) & (col_index == 0):
            text_flag = True
        else:
            text_flag = False
        hf.plot_reddening_vect(ax=ax_1[row_index, col_index],
                               x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                               x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                               linewidth=2, line_color='k', text=text_flag, fontsize=fontsize-4,
                               x_text_offset=-0.1, y_text_offset=-0.2)

        # model
        model_x = globals()['mag_%s_sol' % x_color[0]] - globals()['mag_%s_sol' % x_color[1]]
        model_y = globals()['mag_%s_sol' % y_color[0]] - globals()['mag_%s_sol' % y_color[1]]
        ax_1[row_index, col_index].plot(model_x, model_y, color='tab:red', linewidth=2, zorder=10)

        # data points
        ax_1[row_index, col_index].scatter(color_x[~h_alpha_mask], color_y[~h_alpha_mask],
                                           color='silver', s=10)
        color_x_above_threshold = color_x[h_alpha_mask]
        color_y_above_threshold = color_y[h_alpha_mask]
        h_alpha_medsub_above_threshold = h_alpha_medsub[h_alpha_mask]
        h_alpha_sort = np.argsort(h_alpha_medsub_above_threshold)
        color_x_above_threshold = color_x_above_threshold[h_alpha_sort]
        color_y_above_threshold = color_y_above_threshold[h_alpha_sort]
        h_alpha_medsub_above_threshold = h_alpha_medsub_above_threshold[h_alpha_sort]

        ax_1[row_index, col_index].scatter(color_x_above_threshold, color_y_above_threshold,
                                           c=h_alpha_medsub_above_threshold, norm=norm, cmap=cmap, s=30)

        anchored_left = AnchoredText(target.upper() +
                                     ' ($\Delta$MS=%.2f)' % delta_ms +
                                     '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + '%.1f Mpc' % dist,
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_1[row_index, col_index].add_artist(anchored_left)

        ax_1[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                               direction='in', labelsize=fontsize)

        dummy_vi_points = np.linspace(-1, 1.3)
        dummy_ub_points = 0.64 * dummy_vi_points - 1.0
        ax_1[row_index, col_index].plot(dummy_vi_points, dummy_ub_points, linewidth=3, linestyle='--', color='dodgerblue')
        ax_1[row_index, col_index].plot([1.3, 1.3], [0.64 * 1.3 - 1.0, 1.5], linewidth=3, linestyle='--', color='dodgerblue')

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0

    fig_1.subplots_adjust(left=0.055, bottom=0.035, right=0.995, top=0.93, wspace=0.0, hspace=0.0)

    ax_1[0, 0].set_ylim(y_lim_ub)
    ax_1[0, 0].set_xlim(x_lim_vi)
    x_label = 'V-I [mag]'
    y_label = 'U-B [mag]'
    fig_1.text(0.5, 0.01, x_label, ha='center', fontsize=fontsize)
    fig_1.text(0.01, 0.5, y_label, va='center', ha='left', rotation='vertical', fontsize=fontsize)

    if cluster_class == 'class12':
        classify_str = '%s Class 1+2 Clusters ' % classify.upper()
    elif cluster_class == 'class3':
        classify_str = '%s Class 3 Compact Association' % classify.upper()
    else:
        classify_str = ''
    fig_1.text(0.055, 0.935, r'%s, H$\alpha$>%.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify_str, h_alpha_cut),
               ha='left', va='bottom', fontsize=fontsize + 4)

    ax_cbar = fig_1.add_axes([0.78, 0.95, 0.2, 0.012])
    ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
    ax_cbar.set_xlabel(r'S(H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=3, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    fig_name_str = '%s_%s_panel_1_%s_c%s_h%s' % (x_color, y_color, classify, cluster_class[5:], h_alpha_cut)
    fig_1.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_1.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)

    #
    #
    #

    row_index = 0
    col_index = 0
    for index in range(20, 39):
        target = target_list[index]
        dist = dist_list[index]
        delta_ms = delta_ms_list[index]
        mass = mass_list[index]
        print('target ', target, 'dist ', dist)

        color_x = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_y = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify=classify,
                                                                             cluster_class=cluster_class)
        color_x_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % x_color)(target=target, classify='ml')
        color_y_ml_12 = getattr(catalog_access, 'get_hst_color_%s_vega' % y_color)(target=target, classify='ml')

        h_alpha_medsub = catalog_access.get_h_alpha_medsub(target=target, classify=classify,
                                                               cluster_class=cluster_class)
        hii_mask = catalog_access.get_h_alpha_threshold_mask(target=target, classify=classify,
                                                             cluster_class=cluster_class, h_alpha_cut=h_alpha_cut)

        h_alpha_mask = (h_alpha_medsub > h_alpha_cut) & (hii_mask > 0)

        # plot contours, arrow and model
        good_colors_ml_12 = (color_x_ml_12 > -5) & (color_x_ml_12 < 5) & (color_y_ml_12 > -5) & (color_y_ml_12 < 5)
        contours(ax=ax_2[row_index, col_index], x=color_x_ml_12[good_colors_ml_12], y=color_y_ml_12[good_colors_ml_12],
                 levels=None)
        # arrow
        if (row_index == 0) & (col_index == 0):
            text_flag = True
        else:
            text_flag = False
        hf.plot_reddening_vect(ax=ax_2[row_index, col_index],
                               x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                               x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                               linewidth=2, line_color='k', text=text_flag, fontsize=fontsize-4,
                               x_text_offset=-0.1, y_text_offset=-0.2)

        model_x = globals()['mag_%s_sol' % x_color[0]] - globals()['mag_%s_sol' % x_color[1]]
        model_y = globals()['mag_%s_sol' % y_color[0]] - globals()['mag_%s_sol' % y_color[1]]
        ax_2[row_index, col_index].plot(model_x, model_y, color='tab:red', linewidth=2, zorder=10)
        # data points
        ax_2[row_index, col_index].scatter(color_x[~h_alpha_mask], color_y[~h_alpha_mask],
                                           color='silver', s=10)
        color_x_above_threshold = color_x[h_alpha_mask]
        color_y_above_threshold = color_y[h_alpha_mask]
        h_alpha_medsub_above_threshold = h_alpha_medsub[h_alpha_mask]
        h_alpha_sort = np.argsort(h_alpha_medsub_above_threshold)
        color_x_above_threshold = color_x_above_threshold[h_alpha_sort]
        color_y_above_threshold = color_y_above_threshold[h_alpha_sort]
        h_alpha_medsub_above_threshold = h_alpha_medsub_above_threshold[h_alpha_sort]

        ax_2[row_index, col_index].scatter(color_x_above_threshold, color_y_above_threshold,
                                           c=h_alpha_medsub_above_threshold, norm=norm, cmap=cmap, s=30)

        anchored_left = AnchoredText(target.upper() +
                                     ' ($\Delta$MS=%.2f)' % delta_ms +
                                     '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + '%.1f Mpc' % dist,
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_2[row_index, col_index].add_artist(anchored_left)

        ax_2[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                               direction='in', labelsize=fontsize)

        dummy_vi_points = np.linspace(-1, 1.3)
        dummy_ub_points = 0.64 * dummy_vi_points - 1.0
        ax_2[row_index, col_index].plot(dummy_vi_points, dummy_ub_points, linewidth=3, linestyle='--', color='dodgerblue')
        ax_2[row_index, col_index].plot([1.3, 1.3], [0.64 * 1.3 - 1.0, 1.5], linewidth=3, linestyle='--', color='dodgerblue')

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0

    fig_2.subplots_adjust(left=0.055, bottom=0.035, right=0.995, top=0.93, wspace=0.0, hspace=0.0)

    ax_2[0, 0].set_ylim(y_lim_ub)
    ax_2[0, 0].set_xlim(x_lim_vi)
    x_label = 'V-I [mag]'
    y_label = 'U-B [mag]'
    fig_2.text(0.5, 0.01, x_label, ha='center', fontsize=fontsize)
    fig_2.text(0.01, 0.5, y_label, va='center', ha='left', rotation='vertical', fontsize=fontsize)

    if cluster_class == 'class12':
        classify_str = '%s Class 1+2 Clusters ' % classify.upper()
    elif cluster_class == 'class3':
        classify_str = '%s Class 3 Compact Association' % classify.upper()
    else:
        classify_str = ''
    fig_2.text(0.055, 0.935, r'%s, H$\alpha$>%.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify_str, h_alpha_cut),
               ha='left', va='bottom', fontsize=fontsize + 4)

    # ax_cbar = fig_2.add_axes([0.795, 0.14, 0.17, 0.012])
    # ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
    # ax_cbar.set_xlabel(r'S(H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=0, fontsize=fontsize)
    # ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
    #                     labeltop=True, labelsize=fontsize)

    ax_cbar = fig_2.add_axes([0.78, 0.95, 0.2, 0.012])
    ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
    ax_cbar.set_xlabel(r'S(H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=3, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    # handles, labels = ax_2[4, 3].get_legend_handles_labels()
    # point = Line2D([0], [0], label='ML Class 1|2', marker='s', markersize=30,
    #          markeredgecolor='k', markerfacecolor='None', linestyle='')
    # # add manual symbols to auto legend
    # handles.extend([point])
    # ax_2[4, 3].legend(handles=handles, frameon=False, fontsize=fontsize)

    # ax_2[4, 3].plot([], [], color='k', label='ML Class 1|2')
    # ax_2[4, 3].hist([], color='k', histtype='step', label='ML Class 1|2')
    # ax_2[4, 3].legend(frameon=False, fontsize=fontsize)
    ax_2[4, 3].axis('off')


    fig_name_str = '%s_%s_panel_2_%s_h%s_c%s' % (x_color, y_color, classify, h_alpha_cut,cluster_class[5:])
    fig_2.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_2.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)


plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=1, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=2, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=3, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=5, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class12', h_alpha_cut=10, save_pdf=True)

plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=1, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=2, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=3, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=5, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class12', h_alpha_cut=10, save_pdf=True)


plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=1, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=2, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=3, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=5, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='human', cluster_class='class3', h_alpha_cut=10, save_pdf=True)

plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=1, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=2, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=3, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=5, save_pdf=True)
plot_cc_panels(x_color='vi', y_color='ub', classify='ml', cluster_class='class3', h_alpha_cut=10, save_pdf=True)






