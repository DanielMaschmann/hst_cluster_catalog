import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits

np.random.seed(1234)


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
hst_cc_ver = 'phangs_hst_cc_dr4_cr3_hst_ha'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                            hst_cc_ver=hst_cc_ver)

# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod = hdu_a[1].data
age_mod = data_mod['sfh.age']
m_star_mod = data_mod['stellar.m_star']
flux_f555w_mod = data_mod['F555W_UVIS_CHIP2']

ABmag_F555W = - 6
f_mJy_f555w = 1.e3 * 1.e23 * 10.**((ABmag_F555W + 48.6) / -2.5)
lower_lim_m_star = np.log10(f_mJy_f555w * m_star_mod / flux_f555w_mod)


target_list = catalog_access.target_hst_cc_hst_ha
dist_list = []
delta_ms_list = []
mass_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
    delta_ms_list.append(catalog_access.get_target_delta_ms(target=target))
    mass_list.append(catalog_access.get_target_mstar(target=target))

# sort = np.argsort(delta_ms_list)[::-1]
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]
delta_ms_list = np.array(delta_ms_list)[sort]
mass_list = np.array(mass_list)[sort]


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')



color_c1 = 'tab:green'
color_c2 = 'mediumblue'
color_c3 = 'darkorange'


x_lim_age = (5.7, 10.3)
y_lim_mass = (1.9, 8.7)

fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
fig_hum.set_size_inches(16, 18)
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
fig_ml.set_size_inches(16, 18)
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    delta_ms = delta_ms_list[index]
    mass = mass_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    non_det_flag_12_hum = catalog_access.get_hst_cc_det_flag(target=target)
    cov_flag_12_hum = catalog_access.get_hst_cc_cov_flag(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    non_det_flag_3_hum = catalog_access.get_hst_cc_det_flag(target=target, cluster_class='class3')
    cov_flag_3_hum = catalog_access.get_hst_cc_cov_flag(target=target, cluster_class='class3')

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    non_det_flag_12_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml')
    cov_flag_12_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    non_det_flag_3_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml', cluster_class='class3')
    cov_flag_3_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml', cluster_class='class3')

    clean_mask_12_hum = (non_det_flag_12_hum < 2) & (cov_flag_12_hum < 2)
    clean_mask_3_hum = (non_det_flag_3_hum < 2) & (cov_flag_3_hum < 2)
    clean_mask_12_ml = (non_det_flag_12_ml < 2) & (cov_flag_12_ml < 2)
    clean_mask_3_ml = (non_det_flag_3_ml < 2) & (cov_flag_3_ml < 2)

    ax_hum[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
    ax_hum[row_index, col_index].scatter((np.log10(age_3_hum) + 6)[(cluster_class_3_hum == 3)*clean_mask_3_hum],
                                         np.log10(m_star_3_hum[(cluster_class_3_hum == 3)*clean_mask_3_hum]), c=color_c3, s=20, alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 2)*clean_mask_12_hum],
                                         np.log10(m_star_12_hum)[(cluster_class_12_hum == 2)*clean_mask_12_hum], c=color_c2, s=20,
                                         alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 1)*clean_mask_12_hum],
                                         np.log10(m_star_12_hum)[(cluster_class_12_hum == 1)*clean_mask_12_hum], c=color_c1, s=20,
                                         alpha=0.7)

    ax_ml[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
    ax_ml[row_index, col_index].scatter((np.log10(age_3_ml) + 6)[(cluster_class_3_ml == 3)*clean_mask_3_ml],
                                         np.log10(m_star_3_ml[(cluster_class_3_ml == 3)*clean_mask_3_ml]), c=color_c3, s=20, alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[(cluster_class_12_ml == 2)*clean_mask_12_ml],
                                         np.log10(m_star_12_ml)[(cluster_class_12_ml == 2)*clean_mask_12_ml], c=color_c2, s=20,
                                         alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[(cluster_class_12_ml == 1)*clean_mask_12_ml],
                                         np.log10(m_star_12_ml)[(cluster_class_12_ml == 1)*clean_mask_12_ml], c=color_c1, s=20,
                                         alpha=0.7)
    anchored_left = AnchoredText(target.upper() +
                                 ' ($\Delta$MS=%.2f)' % delta_ms +
                                 '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_hum[row_index, col_index].add_artist(anchored_left)
    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)


    anchored_left = AnchoredText(target.upper() +
                                 ' ($\Delta$MS=%.2f)' % delta_ms +
                                 '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_ml[row_index, col_index].add_artist(anchored_left)
    ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)


    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

ax_hum[0, 0].set_xlim(x_lim_age)
ax_hum[0, 0].set_ylim(y_lim_mass)
# fig_hum.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_hum.text(0.55, 0.087, 'log(Age/yr)', ha='center', va='center', fontsize=fontsize)
fig_hum.text(0.08, 0.55, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

ax_ml[0, 0].set_xlim(x_lim_age)
ax_ml[0, 0].set_ylim(y_lim_mass)
# fig_ml.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_ml.text(0.55, 0.087, 'log(Age/yr)', ha='center', va='center', fontsize=fontsize)
fig_ml.text(0.08, 0.55, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

ax_hum[4, 3].axis('off')
ax_ml[4, 3].axis('off')
ax_hum[4, 2].axis('off')
ax_ml[4, 2].axis('off')

ax_hum[0, 0].scatter([], [], c=color_c1, s=30, label='C1 cluster (Hum)')
ax_hum[0, 0].scatter([], [], c=color_c2, s=30, label='C2 cluster (Hum)')
ax_hum[0, 0].scatter([], [], c=color_c3, s=30, label='C3 compact association (Hum)')
ax_hum[0, 0].legend(frameon=True, ncols=3, loc="upper center", bbox_to_anchor=[2.005, 1.16], fontsize=fontsize-4)
# fig_hum.text(0.55, 1.01, 'Class 1+2 Clusters', ha='center', va='center', fontsize=fontsize)

ax_ml[0, 0].scatter([], [], c=color_c1, s=30, label='C1 cluster (ML)')
ax_ml[0, 0].scatter([], [], c=color_c2, s=30, label='C2 cluster (ML)')
ax_ml[0, 0].scatter([], [], c=color_c3, s=30, label='C3 compact association (ML)')
ax_ml[0, 0].legend(frameon=True, ncols=3, loc="upper center", bbox_to_anchor=[2.005, 1.16], fontsize=fontsize-4)
# fig_ml.text(0.55, 1.01, 'Class 1+2 Clusters', ha='center', va='center', fontsize=fontsize)


fig_hum.subplots_adjust(left=0.11, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_hum.savefig('plot_output/age_m_star_hum_hst_ha_c123.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_m_star_hum_hst_ha_c123.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(left=0.11, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_ml.savefig('plot_output/age_m_star_ml_hst_ha_c123.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_m_star_ml_hst_ha_c123.pdf', bbox_inches='tight', dpi=300)

exit()
