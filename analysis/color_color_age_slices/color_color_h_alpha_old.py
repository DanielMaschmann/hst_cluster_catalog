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


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access_ir4 = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                                hst_cc_ver='IR4')
catalog_access_fix = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                                hst_cc_ver='SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass')

target_list = catalog_access_ir4.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access_ir4.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access_ir4.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access_fix.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

extra_file_path = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/'

target_name_hum = np.array([], dtype=str)
clcl_color_hum = np.array([])
color_vi_hum = np.array([])
color_ub_hum = np.array([])
detect_u_hum = np.array([], dtype=bool)
detect_b_hum = np.array([], dtype=bool)
detect_v_hum = np.array([], dtype=bool)
detect_i_hum = np.array([], dtype=bool)
age_ir4_hum = np.array([])
age_fix_hum = np.array([])
age_fix_youngest_hum = np.array([])
age_fix_likeliest_hum = np.array([])
ebv_ir4_hum = np.array([])
ebv_fix_hum = np.array([])
h_alpha_intensity_hum = np.array([])
hii_mask_hum = np.array([])

target_name_ml = np.array([], dtype=str)
clcl_color_ml = np.array([])
color_vi_ml = np.array([])
color_ub_ml = np.array([])
detect_u_ml = np.array([], dtype=bool)
detect_b_ml = np.array([], dtype=bool)
detect_v_ml = np.array([], dtype=bool)
detect_i_ml = np.array([], dtype=bool)
age_ir4_ml = np.array([])
age_fix_ml = np.array([])
age_fix_youngest_ml = np.array([])
age_fix_likeliest_ml = np.array([])
ebv_ir4_ml = np.array([])
ebv_fix_ml = np.array([])
h_alpha_intensity_ml = np.array([])
hii_mask_ml = np.array([])


for index in range(len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    if 'F438W' in catalog_access_ir4.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    cluster_class_hum_12 = catalog_access_ir4.get_hst_cc_class_human(target=target)
    color_vi_hum_12 = catalog_access_ir4.get_hst_color_vi_vega(target=target)
    color_ub_hum_12 = catalog_access_ir4.get_hst_color_ub_vega(target=target)

    detect_u_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F336W') > 0
    detect_b_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band=b_band) > 0
    detect_v_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F555W') > 0
    detect_i_hum_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, band='F814W') > 0

    age_ir4_hum_12 = catalog_access_ir4.get_hst_cc_age(target=target)
    age_fix_hum_12 = catalog_access_fix.get_hst_cc_age(target=target)
    ebv_ir4_hum_12 = catalog_access_ir4.get_hst_cc_ebv(target=target)
    ebv_fix_hum_12 = catalog_access_fix.get_hst_cc_ebv(target=target)
    cluster_id_hum_12 = catalog_access_ir4.get_hst_cc_phangs_id(target=target)

    cluster_class_hum_3 = catalog_access_ir4.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access_ir4.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access_ir4.get_hst_color_ub_vega(target=target, cluster_class='class3')

    detect_u_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0
    detect_b_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0
    detect_v_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0
    detect_i_hum_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0

    age_ir4_hum_3 = catalog_access_ir4.get_hst_cc_age(target=target, cluster_class='class3')
    age_fix_hum_3 = catalog_access_fix.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_ir4_hum_3 = catalog_access_ir4.get_hst_cc_ebv(target=target, cluster_class='class3')
    ebv_fix_hum_3 = catalog_access_fix.get_hst_cc_ebv(target=target, cluster_class='class3')
    cluster_id_hum_3 = catalog_access_ir4.get_hst_cc_phangs_id(target=target, cluster_class='class3')


    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    detect_u_hum = np.concatenate([detect_u_hum, detect_u_hum_12, detect_u_hum_3])
    detect_b_hum = np.concatenate([detect_b_hum, detect_b_hum_12, detect_b_hum_3])
    detect_v_hum = np.concatenate([detect_v_hum, detect_v_hum_12, detect_v_hum_3])
    detect_i_hum = np.concatenate([detect_i_hum, detect_i_hum_12, detect_i_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])
    target_name_hum_12 = np.array([target]*len(cluster_class_hum_12))
    target_name_hum_3 = np.array([target]*len(cluster_class_hum_3))
    target_name_hum = np.concatenate([target_name_hum, target_name_hum_12, target_name_hum_3])
    age_ir4_hum = np.concatenate([age_ir4_hum, age_ir4_hum_12, age_ir4_hum_3])
    age_fix_hum = np.concatenate([age_fix_hum, age_fix_hum_12, age_fix_hum_3])
    ebv_ir4_hum = np.concatenate([ebv_ir4_hum, ebv_ir4_hum_12, ebv_ir4_hum_3])
    ebv_fix_hum = np.concatenate([ebv_fix_hum, ebv_fix_hum_12, ebv_fix_hum_3])
    cluster_id_hum = np.concatenate([cluster_id_hum_12, cluster_id_hum_3])


    cluster_class_ml_12 = catalog_access_ir4.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    color_vi_ml_12 = catalog_access_ir4.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_ml_12 = catalog_access_ir4.get_hst_color_ub_vega(target=target, classify='ml')

    detect_u_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0
    detect_b_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0
    detect_v_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0
    detect_i_ml_12 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0

    age_ir4_ml_12 = catalog_access_ir4.get_hst_cc_age(target=target, classify='ml')
    age_fix_ml_12 = catalog_access_fix.get_hst_cc_age(target=target, classify='ml')
    ebv_ir4_ml_12 = catalog_access_ir4.get_hst_cc_ebv(target=target, classify='ml')
    ebv_fix_ml_12 = catalog_access_fix.get_hst_cc_ebv(target=target, classify='ml')
    cluster_id_ml_12 = catalog_access_ir4.get_hst_cc_phangs_id(target=target, classify='ml')

    cluster_class_ml_3 = catalog_access_ir4.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access_ir4.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access_ir4.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')

    detect_u_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0
    detect_b_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0
    detect_v_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0
    detect_i_ml_3 = catalog_access_ir4.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0

    age_ir4_ml_3 = catalog_access_ir4.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    age_fix_ml_3 = catalog_access_fix.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_ir4_ml_3 = catalog_access_ir4.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    ebv_fix_ml_3 = catalog_access_fix.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    cluster_id_ml_3 = catalog_access_ir4.get_hst_cc_phangs_id(target=target, classify='ml', cluster_class='class3')


    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    detect_u_ml = np.concatenate([detect_u_ml, detect_u_ml_12, detect_u_ml_3])
    detect_b_ml = np.concatenate([detect_b_ml, detect_b_ml_12, detect_b_ml_3])
    detect_v_ml = np.concatenate([detect_v_ml, detect_v_ml_12, detect_v_ml_3])
    detect_i_ml = np.concatenate([detect_i_ml, detect_i_ml_12, detect_i_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    target_name_ml_12 = np.array([target]*len(cluster_class_ml_12))
    target_name_ml_3 = np.array([target]*len(cluster_class_ml_3))
    target_name_ml = np.concatenate([target_name_ml, target_name_ml_12, target_name_ml_3])
    age_ir4_ml = np.concatenate([age_ir4_ml, age_ir4_ml_12, age_ir4_ml_3])
    age_fix_ml = np.concatenate([age_fix_ml, age_fix_ml_12, age_fix_ml_3])
    ebv_ir4_ml = np.concatenate([ebv_ir4_ml, ebv_ir4_ml_12, ebv_ir4_ml_3])
    ebv_fix_ml = np.concatenate([ebv_fix_ml, ebv_fix_ml_12, ebv_fix_ml_3])
    cluster_id_ml = np.concatenate([cluster_id_ml_12, cluster_id_ml_3])


    # get h_alpha now
    h_alpha_intensity_hum_array = np.zeros(len(cluster_id_hum))
    hii_mask_hum_array = np.zeros(len(cluster_id_hum))
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
        extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev1']

        for running_index, cluster_id in enumerate(cluster_id_hum):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_array[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_mask_hum_array[running_index] = extra_hii_mask[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    hii_mask_hum = np.concatenate([hii_mask_hum, hii_mask_hum_array])
    h_alpha_intensity_hum = np.concatenate([h_alpha_intensity_hum, h_alpha_intensity_hum_array])


    # get h_alpha now
    h_alpha_intensity_ml_array = np.zeros(len(cluster_id_ml))
    hii_mask_ml_array = np.zeros(len(cluster_id_ml))
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
        extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev1']

        for running_index, cluster_id in enumerate(cluster_id_ml):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_ml_array[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_mask_ml_array[running_index] = extra_hii_mask[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    hii_mask_ml = np.concatenate([hii_mask_ml, hii_mask_ml_array])
    h_alpha_intensity_ml = np.concatenate([h_alpha_intensity_ml, h_alpha_intensity_ml_array])



age_mask_1 = age_fix_ml == 1
age_mask_2 = age_fix_ml == 2
age_mask_3 = age_fix_ml == 3
age_mask_4 = age_fix_ml == 4
age_mask_100 = age_fix_ml == 1000


plt.hist(h_alpha_intensity_ml[age_mask_1], bins=np.linspace(0, 30, 30), density=True, histtype='step')
plt.hist(h_alpha_intensity_ml[age_mask_2], bins=np.linspace(0, 30, 30), density=True, histtype='step')
plt.hist(h_alpha_intensity_ml[age_mask_3], bins=np.linspace(0, 30, 30), density=True, histtype='step')
plt.hist(h_alpha_intensity_ml[age_mask_4], bins=np.linspace(0, 30, 30), density=True, histtype='step')
plt.hist(h_alpha_intensity_ml[age_mask_100], bins=np.linspace(0, 30, 30), density=True, histtype='step')

plt.show()

exit()


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

    fig_name_str = '%s_%s_panel_2_%s_h%s_c%s' % (x_color, y_color, classify, h_alpha_cut,cluster_class[5:])
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






