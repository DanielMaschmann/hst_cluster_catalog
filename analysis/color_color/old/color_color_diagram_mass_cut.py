import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools


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
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


# sfr = []
# sfr_err = []
# mstar = []
# mstar_err = []
# for index in range(0, 39):
#     target = target_list[index]
#     dist = dist_list[index]
#     if (target == 'ngc0628c') | (target == 'ngc0628e'):
#         target = 'ngc0628'
#     sfr.append(catalog_access.get_target_sfr(target=target))
#     sfr_err.append(catalog_access.get_target_sfr_err(target=target))
#     mstar.append(catalog_access.get_target_mstar(target=target))
#     mstar_err.append(catalog_access.get_target_mstar_err(target=target))
#
# empty_access = analysis_tools.AnalysisTools(object_type='empty')
# delta_ms_phangs = np.log10(sfr) - empty_access.main_sequence_sf(redshift=0, log_stellar_mass=np.log10(mstar), ref='Whitaker+12')
#
# sort = np.argsort(delta_ms_phangs)
# target_list = np.array(target_list)[sort]
# dist_list = np.array(dist_list)[sort]


color_class_1 = 'royalblue'
color_class_2 = 'forestgreen'
color_class_3 = 'darkorange'

fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
fig_hum.set_size_inches(16, 18)
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
fig_ml.set_size_inches(16, 18)
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
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
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    color_ub_hum = np.concatenate([color_ub_12_hum, color_ub_3_hum])
    color_vi_hum = np.concatenate([color_vi_12_hum, color_vi_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    m_star_hum = np.concatenate([m_star_12_hum, m_star_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])


    color_ub_12_ml = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_12_ml = catalog_access.get_hst_color_vi(target=target, classify='ml')
    color_ub_3_ml = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_3_ml = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    color_ub_ml = np.concatenate([color_ub_12_ml, color_ub_3_ml])
    color_vi_ml = np.concatenate([color_vi_12_ml, color_vi_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    m_star_ml = np.concatenate([m_star_12_ml, m_star_3_ml])
    cluster_class_ml = np.concatenate([cluster_class_12_ml, cluster_class_3_ml])
    cluster_class_qual_ml = np.concatenate([cluster_class_qual_12_ml, cluster_class_qual_3_ml])

    class_1_hum = cluster_class_hum == 1
    class_2_hum = cluster_class_hum == 2
    class_3_hum = cluster_class_hum == 3

    class_1_ml = cluster_class_ml == 1
    class_2_ml = cluster_class_ml == 2
    class_3_ml = cluster_class_ml == 3

    mask_complete_hum = (m_star_hum > 1e4) & (age_hum < 1e3)
    mask_complete_ml = (m_star_ml > 1e4) & (age_ml < 1e3)
    mask_qual_ml = cluster_class_qual_ml >= 0.9


    ax_hum[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=2)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_3_hum*mask_complete_hum], color_ub_hum[class_3_hum*mask_complete_hum], c=color_class_3, s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_2_hum*mask_complete_hum], color_ub_hum[class_2_hum*mask_complete_hum], c=color_class_2, s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_1_hum*mask_complete_hum], color_ub_hum[class_1_hum*mask_complete_hum], c=color_class_1, s=1)

    ax_ml[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=2)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_3_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_3_ml*mask_complete_ml*mask_qual_ml], c=color_class_3, s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_2_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_2_ml*mask_complete_ml*mask_qual_ml], c=color_class_2, s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_1_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_1_ml*mask_complete_ml*mask_qual_ml], c=color_class_1, s=1)


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

ax_ml[0, 0].set_ylim(1.25, -2.2)
ax_ml[0, 0].set_xlim(-1.0, 2.3)
fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/ub_vi_hum_mass_cut_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_mass_cut_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_mass_cut_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_mass_cut_1.pdf', bbox_inches='tight', dpi=300)


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

    color_ub_12_hum = catalog_access.get_hst_color_ub(target=target)
    color_vi_12_hum = catalog_access.get_hst_color_vi(target=target)
    color_ub_3_hum = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_3_hum = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    color_ub_hum = np.concatenate([color_ub_12_hum, color_ub_3_hum])
    color_vi_hum = np.concatenate([color_vi_12_hum, color_vi_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    m_star_hum = np.concatenate([m_star_12_hum, m_star_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])


    color_ub_12_ml = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_12_ml = catalog_access.get_hst_color_vi(target=target, classify='ml')
    color_ub_3_ml = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_3_ml = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    color_ub_ml = np.concatenate([color_ub_12_ml, color_ub_3_ml])
    color_vi_ml = np.concatenate([color_vi_12_ml, color_vi_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    m_star_ml = np.concatenate([m_star_12_ml, m_star_3_ml])
    cluster_class_ml = np.concatenate([cluster_class_12_ml, cluster_class_3_ml])
    cluster_class_qual_ml = np.concatenate([cluster_class_qual_12_ml, cluster_class_qual_3_ml])

    class_1_hum = cluster_class_hum == 1
    class_2_hum = cluster_class_hum == 2
    class_3_hum = cluster_class_hum == 3

    class_1_ml = cluster_class_ml == 1
    class_2_ml = cluster_class_ml == 2
    class_3_ml = cluster_class_ml == 3

    mask_complete_hum = (m_star_hum > 1e4) & (age_hum < 1e3)
    mask_complete_ml = (m_star_ml > 1e4) & (age_ml < 1e3)
    mask_qual_ml = cluster_class_qual_ml >= 0.9


    ax_hum[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=2)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_3_hum*mask_complete_hum], color_ub_hum[class_3_hum*mask_complete_hum], c=color_class_3, s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_2_hum*mask_complete_hum], color_ub_hum[class_2_hum*mask_complete_hum], c=color_class_2, s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum[class_1_hum*mask_complete_hum], color_ub_hum[class_1_hum*mask_complete_hum], c=color_class_1, s=1)

    ax_ml[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=2)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_3_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_3_ml*mask_complete_ml*mask_qual_ml], c=color_class_3, s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_2_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_2_ml*mask_complete_ml*mask_qual_ml], c=color_class_2, s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml[class_1_ml*mask_complete_ml*mask_qual_ml], color_ub_ml[class_1_ml*mask_complete_ml*mask_qual_ml], c=color_class_1, s=1)


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
fig_hum.savefig('plot_output/ub_vi_hum_mass_cut_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_mass_cut_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_mass_cut_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_mass_cut_2.pdf', bbox_inches='tight', dpi=300)

