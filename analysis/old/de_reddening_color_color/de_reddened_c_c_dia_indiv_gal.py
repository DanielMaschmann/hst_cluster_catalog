import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits


def scale_reddening_vector(cluster_ebv, x_comp, y_comp):

    """This function scales the reddening vector for the given E(B-V) value of a star cluster"""
    comparison_scale = 1.0 / 3.2
    scale_factor = cluster_ebv / comparison_scale

    return x_comp * scale_factor, y_comp * scale_factor


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)

# get model
hdu_a = fits.open('../cigale_model/sfh2exp/out/models-block-0.fits')
data = hdu_a[1].data
age = data['sfh.age']
flux_f555w = data['F555W_UVIS_CHIP2']
flux_f814w = data['F814W_UVIS_CHIP2']
flux_f336w = data['F336W_UVIS_CHIP2']
flux_f438w = data['F438W_UVIS_CHIP2']

mag_v = hf.conv_mjy2vega(flux=flux_f555w, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                         vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i = hf.conv_mjy2vega(flux=flux_f814w, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                         vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u = hf.conv_mjy2vega(flux=flux_f336w, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                         vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b = hf.conv_mjy2vega(flux=flux_f438w, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                         vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))

model_vi = mag_v - mag_i
model_ub = mag_u - mag_b


target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')

catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

for target in target_list:

    fig_hum, ax_hum = plt.subplots(1, 3, sharex=True, sharey=True)
    fig_hum.set_size_inches(13, 6)
    fig_ml, ax_ml = plt.subplots(1, 3, sharex=True, sharey=True)
    fig_ml.set_size_inches(13, 6)
    fontsize = 13

    print('target ', target)
    cluster_class_hum_c12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_c12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_c12 = catalog_access.get_hst_color_vi(target=target)
    ebv_hum_c12 = catalog_access.get_hst_cc_ebv(target=target)
    vi_mod_hum_c12, ub_mod_hum_c12 = scale_reddening_vector(ebv_hum_c12, 0.43, 0.3)

    cluster_class_hum_c3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_c3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_c3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    ebv_hum_c3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    vi_mod_hum_c3, ub_mod_hum_c3 = scale_reddening_vector(ebv_hum_c3, 0.43, 0.3)

    cluster_class_hum = np.concatenate([cluster_class_hum_c12, cluster_class_hum_c3])
    color_ub_hum = np.concatenate([color_ub_hum_c12, color_ub_hum_c3])
    color_vi_hum = np.concatenate([color_vi_hum_c12, color_vi_hum_c3])
    ebv_hum = np.concatenate([ebv_hum_c12, ebv_hum_c3])
    vi_mod_hum = np.concatenate([vi_mod_hum_c12, vi_mod_hum_c3])
    ub_mod_hum = np.concatenate([ub_mod_hum_c12, ub_mod_hum_c3])

    cluster_class_ml_c12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    color_ub_ml_c12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_c12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    ebv_ml_c12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    vi_mod_ml_c12, ub_mod_ml_c12 = scale_reddening_vector(ebv_ml_c12, 0.43, 0.3)

    cluster_class_ml_c3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_c3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_c3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    ebv_ml_c3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    vi_mod_ml_c3, ub_mod_ml_c3 = scale_reddening_vector(ebv_ml_c3, 0.43, 0.3)

    cluster_class_ml = np.concatenate([cluster_class_ml_c12, cluster_class_ml_c3])
    color_ub_ml = np.concatenate([color_ub_ml_c12, color_ub_ml_c3])
    color_vi_ml = np.concatenate([color_vi_ml_c12, color_vi_ml_c3])
    ebv_ml = np.concatenate([ebv_ml_c12, ebv_ml_c3])
    vi_mod_ml = np.concatenate([vi_mod_ml_c12, vi_mod_ml_c3])
    ub_mod_ml = np.concatenate([ub_mod_ml_c12, ub_mod_ml_c3])

    ax_hum[0].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
    ax_hum[1].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
    ax_hum[2].plot(model_vi, model_ub, color='salmon', linewidth=1.2)

    ax_ml[0].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
    ax_ml[1].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
    ax_ml[2].plot(model_vi, model_ub, color='salmon', linewidth=1.2)

    for index in range(int(len(color_vi_hum)*0.2)):
        if ebv_hum[index] == 0:
            continue
        ax_hum[0].scatter(color_vi_hum[index], color_ub_hum[index], c='r', s=1)
        ax_hum[0].scatter((color_vi_hum - vi_mod_hum)[index], (color_ub_hum - ub_mod_hum)[index], c='gray', s=1)
        ax_hum[0].plot([color_vi_hum[index], (color_vi_hum - vi_mod_hum)[index]],
                       [color_ub_hum[index], (color_ub_hum - ub_mod_hum)[index]],
                       color='k', linestyle='--')

    for index in range(int(len(color_vi_ml)*0.05)):
        if ebv_ml[index] == 0:
            continue
        ax_ml[0].scatter(color_vi_ml[index], color_ub_ml[index], c='r', s=1)
        ax_ml[0].scatter((color_vi_ml - vi_mod_ml)[index], (color_ub_ml - ub_mod_ml)[index], c='gray', s=1)
        ax_ml[0].plot([color_vi_ml[index], (color_vi_ml - vi_mod_ml)[index]],
                      [color_ub_ml[index], (color_ub_ml - ub_mod_ml)[index]],
                      color='k', linestyle='--')

    ax_hum[1].scatter(color_vi_hum, color_ub_hum, c='r', s=1)
    ax_hum[2].scatter((color_vi_hum - vi_mod_hum), (color_ub_hum - ub_mod_hum), c='gray', s=1)

    ax_ml[1].scatter(color_vi_ml, color_ub_ml, c='r', s=1)
    ax_ml[2].scatter((color_vi_ml - vi_mod_ml), (color_ub_ml - ub_mod_ml), c='gray', s=1)

    ax_hum[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_hum[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_hum[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    ax_ml[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_ml[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_ml[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

    ax_hum[0].set_title(target.upper() + '  Random 20% BCW Class 1/2/3', fontsize=fontsize)
    ax_hum[1].set_title('All BCW Class 1/2/3', fontsize=fontsize)
    ax_hum[2].set_title('De-reddened', fontsize=fontsize)
    ax_hum[0].set_ylim(1.25, -2.2)
    ax_hum[0].set_xlim(-1.0, 2.3)
    ax_hum[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax_hum[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

    ax_ml[0].set_title(target.upper() + '  Random 5% BCW Class 1/2/3', fontsize=fontsize)
    ax_ml[1].set_title('All BCW Class 1/2/3', fontsize=fontsize)
    ax_ml[2].set_title('De-reddened', fontsize=fontsize)
    ax_ml[0].set_ylim(1.25, -2.2)
    ax_ml[0].set_xlim(-1.0, 2.3)
    ax_ml[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
    ax_ml[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

    fig_hum.subplots_adjust(wspace=0, hspace=0)
    fig_hum.savefig('plot_output/individual_gal_hum/ub_vi_dereddening_%s.png' % target, bbox_inches='tight', dpi=300)
    # fig_hum.close()

    fig_ml.subplots_adjust(wspace=0, hspace=0)
    fig_ml.savefig('plot_output/individual_gal_ml/ub_vi_dereddening_%s.png' % target, bbox_inches='tight', dpi=300)
    # fig_ml.close()

