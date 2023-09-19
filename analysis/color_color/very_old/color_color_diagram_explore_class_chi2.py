import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from photometry_tools.plotting_tools import DensityContours


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)


# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/out/models-block-0.fits')
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
# sort = np.argsort(dist_list)
# target_list = np.array(target_list)[sort]
# dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


color_ub_array_hum = np.array([])
color_vi_array_hum = np.array([])
cluster_class_array_hum = np.array([])
chi2_array_hum = np.array([])

color_ub_array_ml = np.array([])
color_vi_array_ml = np.array([])
cluster_class_array_ml = np.array([])
class_qual_array_ml = np.array([])
chi2_array_ml = np.array([])


for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    color_ub_12_hum = catalog_access.get_hst_color_ub(target=target)
    color_vi_12_hum = catalog_access.get_hst_color_vi(target=target)
    color_ub_3_hum = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_3_hum = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    chi2_12_hum = catalog_access.get_hst_cc_min_chi2(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    chi2_3_hum = catalog_access.get_hst_cc_min_chi2(target=target, cluster_class='class3')


    color_ub_hum = np.concatenate([color_ub_12_hum, color_ub_3_hum])
    color_vi_hum = np.concatenate([color_vi_12_hum, color_vi_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    m_star_hum = np.concatenate([m_star_12_hum, m_star_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])
    chi2_hum = np.concatenate([chi2_12_hum, chi2_3_hum])


    color_ub_12_ml = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_12_ml = catalog_access.get_hst_color_vi(target=target, classify='ml')
    color_ub_3_ml = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_3_ml = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    chi2_12_ml = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    chi2_3_ml = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml', cluster_class='class3')
    color_ub_ml = np.concatenate([color_ub_12_ml, color_ub_3_ml])
    color_vi_ml = np.concatenate([color_vi_12_ml, color_vi_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    m_star_ml = np.concatenate([m_star_12_ml, m_star_3_ml])
    cluster_class_ml = np.concatenate([cluster_class_12_ml, cluster_class_3_ml])
    cluster_class_qual_ml = np.concatenate([cluster_class_qual_12_ml, cluster_class_qual_3_ml])
    chi2_ml = np.concatenate([chi2_12_ml, chi2_3_ml])


    color_ub_array_hum = np.concatenate([color_ub_array_hum, color_ub_hum])
    color_vi_array_hum = np.concatenate([color_vi_array_hum, color_vi_hum])
    cluster_class_array_hum = np.concatenate([cluster_class_array_hum, cluster_class_hum])
    chi2_array_hum = np.concatenate([chi2_array_hum, chi2_hum])

    color_ub_array_ml = np.concatenate([color_ub_array_ml, color_ub_ml])
    color_vi_array_ml = np.concatenate([color_vi_array_ml, color_vi_ml])
    cluster_class_array_ml = np.concatenate([cluster_class_array_ml, cluster_class_ml])
    class_qual_array_ml = np.concatenate([class_qual_array_ml, cluster_class_qual_ml])
    chi2_array_ml = np.concatenate([chi2_array_ml, chi2_ml])



good_color_hum = (color_vi_array_hum > -1.0) & (color_vi_array_hum < 1.8) & (color_ub_array_hum > -2.2) & (color_ub_array_hum < 1.25)
good_color_ml = (color_vi_array_ml > -1.0) & (color_vi_array_ml < 1.8) & (color_ub_array_ml > -2.2) & (color_ub_array_ml < 1.25) & (class_qual_array_ml >= 0.9)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
fontsize = 15

ax[0].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[0], x_data=color_vi_array_hum[(chi2_array_hum<0.5)*good_color_hum], y_data=color_ub_array_hum[(chi2_array_hum<0.5)*good_color_hum], color='black', percent=False)
ax[0].set_title('All Hum Clusters', fontsize=14)
ax[0].set_ylim(1.25, -2.2)
ax[0].set_xlim(-1.0, 1.8)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)
ax[0].set_title('Chi2 < 0.5 Hum')

ax[1].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[1], x_data=color_vi_array_hum[(chi2_array_hum>1.5)*good_color_hum], y_data=color_ub_array_hum[(chi2_array_hum>1.5)*good_color_hum], color='black', percent=False)
ax[1].set_title('All Hum Clusters', fontsize=14)
ax[1].set_ylim(1.25, -2.2)
ax[1].set_xlim(-1.0, 1.8)
# ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)
ax[1].set_title('Chi2 > 1.5 Hum')


# plt.show()
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('plot_output/color_color_chi2_hum.png')
plt.clf()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
fontsize = 15

ax[0].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[0], x_data=color_vi_array_ml[(chi2_array_ml<0.5)*good_color_ml], y_data=color_ub_array_ml[(chi2_array_ml<0.5)*good_color_ml], color='black', percent=False)
ax[0].set_title('All ML Clusters', fontsize=14)
ax[0].set_ylim(1.25, -2.2)
ax[0].set_xlim(-1.0, 1.8)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)
ax[0].set_title('Chi2 < 0.5 ML')

ax[1].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[1], x_data=color_vi_array_ml[(chi2_array_ml>1.5)*good_color_ml], y_data=color_ub_array_ml[(chi2_array_ml>1.5)*good_color_ml], color='black', percent=False)
ax[1].set_title('All ML Clusters', fontsize=14)
ax[1].set_ylim(1.25, -2.2)
ax[1].set_xlim(-1.0, 1.8)
# ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)
ax[1].set_title('Chi2 > 1.5 ML')


# plt.show()
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('plot_output/color_color_chi2_ml.png')
plt.clf()






fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(23, 8))
fontsize = 17

ax[0].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[0], x_data=color_vi_array_hum[(cluster_class_array_hum==1)*good_color_hum], y_data=color_ub_array_hum[(cluster_class_array_hum==1)*good_color_hum], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[0].set_title('class 1 Hum', fontsize=fontsize)
ax[0].set_ylim(1.25, -2.2)
ax[0].set_xlim(-1.0, 1.8)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)


ax[1].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[1], x_data=color_vi_array_hum[(cluster_class_array_hum==2)*good_color_hum], y_data=color_ub_array_hum[(cluster_class_array_hum==2)*good_color_hum], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[1].set_ylim(1.25, -2.2)
ax[1].set_xlim(-1.0, 1.8)
# ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_title('Class 2 Hum', fontsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)

ax[2].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[2], x_data=color_vi_array_hum[((cluster_class_array_hum==1) | (cluster_class_array_hum==2))*good_color_hum], y_data=color_ub_array_hum[((cluster_class_array_hum==1) | (cluster_class_array_hum==2))*good_color_hum], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[2].set_ylim(1.25, -2.2)
ax[2].set_xlim(-1.0, 1.8)
# ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_title('Class 1+2 Hum', fontsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)

ax[3].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[3], x_data=color_vi_array_hum[(cluster_class_array_hum==3)*good_color_hum], y_data=color_ub_array_hum[(cluster_class_array_hum==3)*good_color_hum], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[3].set_ylim(1.25, -2.2)
ax[3].set_xlim(-1.0, 1.8)
# ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3].set_title('Class 3 Hum', fontsize=fontsize)
ax[3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)


# plt.show()
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('plot_output/color_color_classes_hum.png')
plt.clf()


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(23, 8))
fontsize = 17

ax[0].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[0], x_data=color_vi_array_ml[(cluster_class_array_ml==1)*good_color_ml], y_data=color_ub_array_ml[(cluster_class_array_ml==1)*good_color_ml], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[0].set_title('class 1 ML', fontsize=fontsize)
ax[0].set_ylim(1.25, -2.2)
ax[0].set_xlim(-1.0, 1.8)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)


ax[1].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[1], x_data=color_vi_array_ml[(cluster_class_array_ml==2)*good_color_ml], y_data=color_ub_array_ml[(cluster_class_array_ml==2)*good_color_ml], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[1].set_ylim(1.25, -2.2)
ax[1].set_xlim(-1.0, 1.8)
# ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_title('Class 2 ML', fontsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)

ax[2].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[2], x_data=color_vi_array_ml[((cluster_class_array_ml==1) | (cluster_class_array_ml==2))*good_color_ml], y_data=color_ub_array_ml[((cluster_class_array_ml==1) | (cluster_class_array_ml==2))*good_color_ml], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[2].set_ylim(1.25, -2.2)
ax[2].set_xlim(-1.0, 1.8)
# ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_title('Class 1+2 ML', fontsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)

ax[3].plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax[3], x_data=color_vi_array_ml[(cluster_class_array_ml==3)*good_color_ml], y_data=color_ub_array_ml[(cluster_class_array_ml==3)*good_color_ml], color='black', percent=True, contour_levels=[0, 0.05, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99])
ax[3].set_ylim(1.25, -2.2)
ax[3].set_xlim(-1.0, 1.8)
# ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3].set_title('Class 3 ML', fontsize=fontsize)
ax[3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)


# plt.show()
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('plot_output/color_color_classes_ml.png')
plt.clf()

