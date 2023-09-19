import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colorbar import ColorbarBase
from photometry_tools import helper_func, data_access


# color range limitations
x_lim_ubvi = (-0.6, 1.9)
y_lim_ubvi = (0.9, -1.9)
n_bins_ubvi = 50

threshold_hum = 5
threshold_ml = 5

model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')

model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

for catalog_version in ['SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha1_inclusiveGCcc_restrictiveGCclass',
                        'SEDfix_Ha2_inclusiveGCcc_restrictiveGCclass',
                        'SEDfix_Ha2_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha3_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha3_inclusiveGCcc_restrictiveGCclass',
                        'IR4']:

    catalog_access = data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                               hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                               morph_mask_path=morph_mask_path,
                                               sample_table_path=sample_table_path,
                                               hst_cc_ver=catalog_version)

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


    target_name_hum = np.array([], dtype=str)
    color_vi_hum = np.array([])
    color_ub_hum = np.array([])
    detect_u_hum = np.array([], dtype=bool)
    detect_b_hum = np.array([], dtype=bool)
    detect_v_hum = np.array([], dtype=bool)
    detect_i_hum = np.array([], dtype=bool)
    clcl_color_hum = np.array([])
    age_hum = np.array([])
    age_err_hum = np.array([])
    ebv_hum = np.array([])
    ebv_err_hum = np.array([])

    target_name_ml = np.array([], dtype=str)
    color_vi_ml = np.array([])
    color_ub_ml = np.array([])
    detect_u_ml = np.array([], dtype=bool)
    detect_b_ml = np.array([], dtype=bool)
    detect_v_ml = np.array([], dtype=bool)
    detect_i_ml = np.array([], dtype=bool)
    clcl_color_ml = np.array([])
    age_ml = np.array([])
    age_err_ml = np.array([])
    ebv_ml = np.array([])
    ebv_err_ml = np.array([])

    for index in range(len(target_list)):
        target = target_list[index]
        print('target ', target)
        if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
            b_band = 'F438W'
        else:
            b_band = 'F435W'

        cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
        color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
        color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)

        detect_u_hum_12 = catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0
        detect_b_hum_12 = catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0
        detect_v_hum_12 = catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0
        detect_i_hum_12 = catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0

        age_hum_12 = catalog_access.get_hst_cc_age(target=target)
        age_err_hum_12 = catalog_access.get_hst_cc_age_err(target=target)
        ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)
        ebv_err_hum_12 = catalog_access.get_hst_cc_ebv_err(target=target)

        cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
        color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
        color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')

        detect_u_hum_3 = catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0
        detect_b_hum_3 = catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0
        detect_v_hum_3 = catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0
        detect_i_hum_3 = catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0

        age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
        age_err_hum_3 = catalog_access.get_hst_cc_age_err(target=target, cluster_class='class3')
        ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
        ebv_err_hum_3 = catalog_access.get_hst_cc_ebv_err(target=target, cluster_class='class3')

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

        age_hum = np.concatenate([age_hum, age_hum_12, age_hum_3])
        age_err_hum = np.concatenate([age_err_hum, age_err_hum_12, age_err_hum_3])
        ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])
        ebv_err_hum = np.concatenate([ebv_err_hum, ebv_err_hum_12, ebv_err_hum_3])


        cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
        color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
        color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
        detect_u_ml_12 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0
        detect_b_ml_12 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0
        detect_v_ml_12 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0
        detect_i_ml_12 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0

        age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
        age_err_ml_12 = catalog_access.get_hst_cc_age_err(target=target, classify='ml')
        ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
        ebv_err_ml_12 = catalog_access.get_hst_cc_ebv_err(target=target, classify='ml')

        cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
        color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
        color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
        detect_u_ml_3 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0
        detect_b_ml_3 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0
        detect_v_ml_3 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0
        detect_i_ml_3 = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0
        age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
        age_err_ml_3 = catalog_access.get_hst_cc_age_err(target=target, classify='ml', cluster_class='class3')
        ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
        ebv_err_ml_3 = catalog_access.get_hst_cc_ebv_err(target=target, classify='ml', cluster_class='class3')

        color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
        color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
        detect_u_ml = np.concatenate([detect_u_ml, detect_u_ml_12, detect_u_ml_3])
        detect_b_ml = np.concatenate([detect_b_ml, detect_b_ml_12, detect_b_ml_3])
        detect_v_ml = np.concatenate([detect_v_ml, detect_v_ml_12, detect_v_ml_3])
        detect_i_ml = np.concatenate([detect_i_ml, detect_i_ml_12, detect_i_ml_3])
        clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
        age_ml = np.concatenate([age_ml, age_ml_12, age_ml_3])
        age_err_ml = np.concatenate([age_err_ml, age_err_ml_12, age_err_ml_3])
        ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])
        ebv_err_ml = np.concatenate([ebv_err_ml, ebv_err_ml_12, ebv_err_ml_3])


    mask_class_1_hum = clcl_color_hum == 1
    mask_class_2_hum = clcl_color_hum == 2
    mask_class_3_hum = clcl_color_hum == 3

    mask_class_1_ml = clcl_color_ml == 1
    mask_class_2_ml = clcl_color_ml == 2
    mask_class_3_ml = clcl_color_ml == 3

    mask_detect_ubvi_hum = detect_v_hum * detect_i_hum * detect_u_hum * detect_b_hum
    mask_detect_ubvi_ml = detect_v_ml * detect_i_ml * detect_u_ml * detect_b_ml

    mask_good_colors_ubvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                                 (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
    mask_good_colors_ubvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                                (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml

    # bins
    x_bins = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], n_bins_ubvi)
    y_bins = np.linspace(y_lim_ubvi[1], y_lim_ubvi[0], n_bins_ubvi)

    age_map_hum_1 = np.zeros((len(x_bins), len(y_bins))) * np.nan
    age_map_hum_2 = np.zeros((len(x_bins), len(y_bins))) * np.nan
    age_map_hum_3 = np.zeros((len(x_bins), len(y_bins))) * np.nan
    age_map_ml_1 = np.zeros((len(x_bins), len(y_bins))) * np.nan
    age_map_ml_2 = np.zeros((len(x_bins), len(y_bins))) * np.nan
    age_map_ml_3 = np.zeros((len(x_bins), len(y_bins))) * np.nan

    for x_index in range(len(x_bins)-1):
        for y_index in range(len(y_bins)-1):
            mask_in_bin_hum = ((color_vi_hum > x_bins[x_index]) & (color_vi_hum < x_bins[x_index + 1]) &
                           (color_ub_hum > y_bins[y_index]) & (color_ub_hum < y_bins[y_index + 1]))
            mask_in_bin_ml = ((color_vi_ml > x_bins[x_index]) & (color_vi_ml < x_bins[x_index + 1]) &
                           (color_ub_ml > y_bins[y_index]) & (color_ub_ml < y_bins[y_index + 1]))

            mask_selected_obj_hum_1 = mask_in_bin_hum * mask_class_1_hum * mask_good_colors_ubvi_hum
            mask_selected_obj_hum_2 = mask_in_bin_hum * mask_class_2_hum * mask_good_colors_ubvi_hum
            mask_selected_obj_hum_3 = mask_in_bin_hum * mask_class_3_hum * mask_good_colors_ubvi_hum
            mask_selected_obj_ml_1 = mask_in_bin_ml * mask_class_1_ml * mask_good_colors_ubvi_ml
            mask_selected_obj_ml_2 = mask_in_bin_ml * mask_class_2_ml * mask_good_colors_ubvi_ml
            mask_selected_obj_ml_3 = mask_in_bin_ml * mask_class_3_ml * mask_good_colors_ubvi_ml

            if sum(mask_selected_obj_hum_1) > threshold_hum:
                age_map_hum_1[x_index, y_index] = np.nanmedian(np.log10(age_hum[mask_selected_obj_hum_1]) + 6)
            if sum(mask_selected_obj_hum_2) > threshold_hum:
                age_map_hum_2[x_index, y_index] = np.nanmedian(np.log10(age_hum[mask_selected_obj_hum_2]) + 6)
            if sum(mask_selected_obj_hum_3) > threshold_hum:
                age_map_hum_3[x_index, y_index] = np.nanmedian(np.log10(age_hum[mask_selected_obj_hum_3]) + 6)

            if sum(mask_selected_obj_ml_1) > threshold_ml:
                age_map_ml_1[x_index, y_index] = np.nanmedian(np.log10(age_ml[mask_selected_obj_ml_1]) + 6)
            if sum(mask_selected_obj_ml_2) > threshold_ml:
                age_map_ml_2[x_index, y_index] = np.nanmedian(np.log10(age_ml[mask_selected_obj_ml_2]) + 6)
            if sum(mask_selected_obj_ml_3) > threshold_ml:
                age_map_ml_3[x_index, y_index] = np.nanmedian(np.log10(age_ml[mask_selected_obj_ml_3]) + 6)



    figure = plt.figure(figsize=(29, 20))
    fontsize = 28

    cmap_age = matplotlib.cm.get_cmap('rainbow')
    norm_age = matplotlib.colors.Normalize(vmin=6, vmax=10.5)



    ax_age_hum_1 = figure.add_axes([0.04, 0.52, 0.32, 0.46])
    ax_age_ml_1 = figure.add_axes([0.04, 0.05, 0.32, 0.46])

    ax_age_hum_2 = figure.add_axes([0.33, 0.52, 0.32, 0.46])
    ax_age_ml_2 = figure.add_axes([0.33, 0.05, 0.32, 0.46])

    ax_age_hum_3 = figure.add_axes([0.62, 0.52, 0.32, 0.46])
    ax_age_ml_3 = figure.add_axes([0.62, 0.05, 0.32, 0.46])

    ax_cbar_age = figure.add_axes([0.94, 0.2, 0.015, 0.6])


    ax_age_hum_1.imshow(age_map_hum_1.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)
    ax_age_hum_2.imshow(age_map_hum_2.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)
    ax_age_hum_3.imshow(age_map_hum_3.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)

    ax_age_ml_1.imshow(age_map_ml_1.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)
    ax_age_ml_2.imshow(age_map_ml_2.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)
    ax_age_ml_3.imshow(age_map_ml_3.T, origin='lower', extent=(x_lim_ubvi[0], x_lim_ubvi[1], y_lim_ubvi[1], y_lim_ubvi[0]), cmap=cmap_age, norm=norm_age)




    ax_age_hum_1.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
    ax_age_hum_1.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')
    ax_age_hum_2.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
    ax_age_hum_2.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

    ax_age_hum_3.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
    ax_age_hum_3.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
    ax_age_ml_1.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
    ax_age_ml_1.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

    ax_age_ml_2.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
    ax_age_ml_2.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')
    ax_age_ml_3.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='-')
    ax_age_ml_3.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')



    vi_int = 1.1
    ub_int = -1.4
    bv_int = -0.2
    av_value = 1

    helper_func.plot_reddening_vect(ax=ax_age_hum_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.1)
    helper_func.plot_reddening_vect(ax=ax_age_hum_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax_age_hum_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)

    helper_func.plot_reddening_vect(ax=ax_age_ml_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
    helper_func.plot_reddening_vect(ax=ax_age_ml_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)
    helper_func.plot_reddening_vect(ax=ax_age_ml_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                           x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                           linewidth=4, line_color='k', text=False, fontsize=fontsize)


    ColorbarBase(ax_cbar_age, orientation='vertical', cmap=cmap_age, norm=norm_age, extend='neither', ticks=None)
    ax_cbar_age.set_ylabel(r'log(Age)', labelpad=-4, fontsize=fontsize + 5)
    ax_cbar_age.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)


    ax_age_hum_1.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Class 1 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax_age_ml_1.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Class 1 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax_age_hum_2.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Class 2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax_age_ml_2.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Class 2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax_age_hum_3.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Compact associations (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax_age_ml_3.text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
                     'Compact associations (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

    ax_age_hum_1.set_xlim(x_lim_ubvi)
    ax_age_hum_2.set_xlim(x_lim_ubvi)
    ax_age_hum_3.set_xlim(x_lim_ubvi)
    ax_age_ml_1.set_xlim(x_lim_ubvi)
    ax_age_ml_2.set_xlim(x_lim_ubvi)
    ax_age_ml_3.set_xlim(x_lim_ubvi)

    ax_age_hum_1.set_ylim(y_lim_ubvi)
    ax_age_hum_2.set_ylim(y_lim_ubvi)
    ax_age_hum_3.set_ylim(y_lim_ubvi)
    ax_age_ml_1.set_ylim(y_lim_ubvi)
    ax_age_ml_2.set_ylim(y_lim_ubvi)
    ax_age_ml_3.set_ylim(y_lim_ubvi)

    ax_age_hum_2.set_yticklabels([])
    ax_age_hum_3.set_yticklabels([])
    ax_age_ml_2.set_yticklabels([])
    ax_age_ml_3.set_yticklabels([])

    ax_age_hum_1.set_xticklabels([])
    ax_age_hum_2.set_xticklabels([])
    ax_age_hum_3.set_xticklabels([])


    ax_age_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax_age_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

    ax_age_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax_age_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
    ax_age_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

    ax_age_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

    ax_age_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_age_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_age_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_age_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_age_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_age_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig('plot_output/color_color_age_%s.png' % catalog_version)
    plt.savefig('plot_output/color_color_age_%s.pdf' % catalog_version)
    plt.clf()
    plt.cla()

