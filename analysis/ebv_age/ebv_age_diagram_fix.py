import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

np.random.seed(1234)

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

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


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

line_84p_age = [0.30, 0.47, 0.68, 0.81, 0.91, 0.95, 0.99, 1.01, 1.06, 1.12, 1.17, 1.22, 1.29, 1.37, 1.49, 1.64, 1.71,
                1.81, 1.88, 2.06, 2.18, 2.31, 2.38, 2.52, 2.66, 2.75, 2.81, 2.90, 2.98, 3.11, 3.20, 3.30]

line_84p_ebv = [1.196, 1.176, 1.091, 1.033, 0.937, 0.847, 0.791, 0.780, 0.788, 0.788, 0.774, 0.727, 0.649, 0.570, 0.526,
                0.488, 0.480, 0.485, 0.485, 0.474, 0.439, 0.398, 0.375, 0.351, 0.340, 0.317, 0.282, 0.209, 0.162, 0.124,
                0.116, 0.113]
line_84p_age = np.array(line_84p_age) + 6
line_84p_ebv = np.array(line_84p_ebv)

color_c1 = 'tab:green'
color_c2 = 'mediumblue'
color_c3 = 'darkorange'

fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):

    target = target_list[index]
    dist = dist_list[index]
    delta_ms = delta_ms_list[index]
    mass = mass_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    non_det_flag_12_hum = catalog_access.get_hst_cc_det_flag(target=target)
    cov_flag_12_hum = catalog_access.get_hst_cc_cov_flag(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    non_det_flag_3_hum = catalog_access.get_hst_cc_det_flag(target=target, cluster_class='class3')
    cov_flag_3_hum = catalog_access.get_hst_cc_cov_flag(target=target, cluster_class='class3')
    non_det_flag_hum = np.concatenate([non_det_flag_12_hum, non_det_flag_3_hum])
    cov_flag_hum = np.concatenate([cov_flag_12_hum, cov_flag_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    ebv_hum = np.concatenate([ebv_12_hum, ebv_3_hum])

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    non_det_flag_12_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml')
    cov_flag_12_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    non_det_flag_3_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml', cluster_class='class3')
    cov_flag_3_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml', cluster_class='class3')
    non_det_flag_ml = np.concatenate([non_det_flag_12_ml, non_det_flag_3_ml])
    cov_flag_ml = np.concatenate([cov_flag_12_ml, cov_flag_3_ml])
    cluster_class_ml = np.concatenate([cluster_class_12_ml, cluster_class_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    ebv_ml = np.concatenate([ebv_12_ml, ebv_3_ml])

    class_1_hum = cluster_class_hum == 1
    class_2_hum = cluster_class_hum == 2
    class_3_hum = cluster_class_hum == 3

    class_1_ml = cluster_class_ml == 1
    class_2_ml = cluster_class_ml == 2
    class_3_ml = cluster_class_ml == 3

    clean_mask_hum = (non_det_flag_hum < 2) & (cov_flag_hum < 2)
    clean_mask_ml = (non_det_flag_ml < 2) & (cov_flag_ml < 2)


    # random dots
    random_x_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_hum))
    random_y_hum = np.random.uniform(low=-0.05, high=0.05, size=len(age_hum))

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_3_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_3_hum * clean_mask_hum], c=color_c3, s=20, alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_2_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_2_hum * clean_mask_hum], c=color_c2, s=20, alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_1_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_1_hum * clean_mask_hum], c=color_c1, s=20, alpha=0.7)
    ax_hum[row_index, col_index].plot(line_84p_age, line_84p_ebv, linewidth=1.2, color='red')
    if target == 'ngc1365':
        ax_hum[row_index, col_index].plot([6, 9], [1, 0.1], linewidth=1.6, linestyle='--', color='red')

    # random dots
    random_x_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_ml))
    random_y_ml = np.random.uniform(low=-0.05, high=0.05, size=len(age_ml))

    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_3_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_3_ml * clean_mask_ml], c=color_c3, s=20, alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_2_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_2_ml * clean_mask_ml], c=color_c2, s=20, alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_1_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_1_ml * clean_mask_ml], c=color_c1, s=20, alpha=0.7)

    ax_ml[row_index, col_index].plot(line_84p_age, line_84p_ebv, linewidth=1.2, color='red')
    if target == 'ngc1365':
        ax_ml[row_index, col_index].plot([6, 9], [1, 0.1], linewidth=1.6, linestyle='--', color='red')

    # anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
    #                              frameon=False, prop=dict(size=fontsize-4))
    anchored_left = AnchoredText(target.upper() +
                                 ' ($\Delta$MS=%.2f)' % delta_ms +
                                 '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_hum[row_index, col_index].add_artist(anchored_left)
    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    # anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
    #                              frameon=False, prop=dict(size=fontsize-4))
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


ax_hum[0, 0].set_xlim(5.7, 10.3)
ax_hum[0, 0].set_ylim(-0.1, 2.1)
fig_hum.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_hum.text(0.08, 0.55, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(-0.1, 2.1)
fig_ml.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_ml.text(0.08, 0.55, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)

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


# plt.tight_layout()
fig_hum.subplots_adjust(left=0.13, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_hum.savefig('plot_output/age_ebv_hum_1_c123.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_ebv_hum_1_c123.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(left=0.13, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_ml.savefig('plot_output/age_ebv_ml_1_c123.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_ebv_ml_1_c123.pdf', bbox_inches='tight', dpi=300)


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
    delta_ms = delta_ms_list[index]
    mass = mass_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    non_det_flag_12_hum = catalog_access.get_hst_cc_det_flag(target=target)
    cov_flag_12_hum = catalog_access.get_hst_cc_cov_flag(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    non_det_flag_3_hum = catalog_access.get_hst_cc_det_flag(target=target, cluster_class='class3')
    cov_flag_3_hum = catalog_access.get_hst_cc_cov_flag(target=target, cluster_class='class3')
    non_det_flag_hum = np.concatenate([non_det_flag_12_hum, non_det_flag_3_hum])
    cov_flag_hum = np.concatenate([cov_flag_12_hum, cov_flag_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    ebv_hum = np.concatenate([ebv_12_hum, ebv_3_hum])

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    non_det_flag_12_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml')
    cov_flag_12_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    non_det_flag_3_ml = catalog_access.get_hst_cc_det_flag(target=target, classify='ml', cluster_class='class3')
    cov_flag_3_ml = catalog_access.get_hst_cc_cov_flag(target=target, classify='ml', cluster_class='class3')
    non_det_flag_ml = np.concatenate([non_det_flag_12_ml, non_det_flag_3_ml])
    cov_flag_ml = np.concatenate([cov_flag_12_ml, cov_flag_3_ml])
    cluster_class_ml = np.concatenate([cluster_class_12_ml, cluster_class_3_ml])
    age_ml = np.concatenate([age_12_ml, age_3_ml])
    ebv_ml = np.concatenate([ebv_12_ml, ebv_3_ml])

    class_1_hum = cluster_class_hum == 1
    class_2_hum = cluster_class_hum == 2
    class_3_hum = cluster_class_hum == 3

    class_1_ml = cluster_class_ml == 1
    class_2_ml = cluster_class_ml == 2
    class_3_ml = cluster_class_ml == 3

    clean_mask_hum = (non_det_flag_hum < 2) & (cov_flag_hum < 2)
    clean_mask_ml = (non_det_flag_ml < 2) & (cov_flag_ml < 2)


    # random dots
    random_x_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_hum))
    random_y_hum = np.random.uniform(low=-0.05, high=0.05, size=len(age_hum))

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_3_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_3_hum * clean_mask_hum], c=color_c3, s=20, alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_2_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_2_hum * clean_mask_hum], c=color_c2, s=20, alpha=0.7)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + random_x_hum + 6)[class_1_hum * clean_mask_hum],
                                         (ebv_hum + random_y_hum)[class_1_hum * clean_mask_hum], c=color_c1, s=20, alpha=0.7)
    ax_hum[row_index, col_index].plot(line_84p_age, line_84p_ebv, linewidth=1.2, color='red')
    if target == 'ngc1365':
        ax_hum[row_index, col_index].plot([6, 9], [1, 0.1], linewidth=1.6, linestyle='--', color='red')

    # random dots
    random_x_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_ml))
    random_y_ml = np.random.uniform(low=-0.05, high=0.05, size=len(age_ml))

    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_3_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_3_ml * clean_mask_ml], c=color_c3, s=20, alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_2_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_2_ml * clean_mask_ml], c=color_c2, s=20, alpha=0.7)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + random_x_ml + 6)[class_1_ml * clean_mask_ml],
                                         (ebv_ml + random_y_ml)[class_1_ml * clean_mask_ml], c=color_c1, s=20, alpha=0.7)

    ax_ml[row_index, col_index].plot(line_84p_age, line_84p_ebv, linewidth=1.2, color='red')
    if target == 'ngc1365':
        ax_ml[row_index, col_index].plot([6, 9], [1, 0.1], linewidth=1.6, linestyle='--', color='red')

    # anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
    #                              frameon=False, prop=dict(size=fontsize-4))
    anchored_left = AnchoredText(target.upper() +
                                 ' ($\Delta$MS=%.2f)' % delta_ms +
                                 '\nlog(M$_{*}$/M$_{\odot})$=%.1f, d=' % np.log10(mass) + str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_hum[row_index, col_index].add_artist(anchored_left)
    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    # anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
    #                              frameon=False, prop=dict(size=fontsize-4))
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




ax_hum[0, 0].set_xlim(5.7, 10.3)
ax_hum[0, 0].set_ylim(-0.1, 2.1)
fig_hum.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_hum.text(0.08, 0.55, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(-0.1, 2.1)
fig_ml.text(0.55, 0.087, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', va='center', fontsize=fontsize)
fig_ml.text(0.08, 0.55, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)

ax_hum[4, 3].scatter([], [], c=color_c1, s=30, label='C1 (Hum)')
ax_hum[4, 3].scatter([], [], c=color_c2, s=30, label='C2 (Hum)')
ax_hum[4, 3].scatter([], [], c=color_c3, s=30, label='C3 (Hum)')
ax_hum[4, 3].legend(frameon=False, fontsize=fontsize-3)
ax_hum[4, 3].axis('off')

ax_ml[4, 3].scatter([], [], c=color_c1, s=30, label='C1 (ML)')
ax_ml[4, 3].scatter([], [], c=color_c2, s=30, label='C2 (ML)')
ax_ml[4, 3].scatter([], [], c=color_c3, s=30, label='C3 (ML)')
ax_ml[4, 3].legend(frameon=False, fontsize=fontsize-3)
ax_ml[4, 3].axis('off')

# plt.tight_layout()
fig_hum.subplots_adjust(left=0.13, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_hum.savefig('plot_output/age_ebv_hum_2_c123.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_ebv_hum_2_c123.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(left=0.13, bottom=0.11, right=0.995, top=0.995, wspace=0.01, hspace=0.01)
fig_ml.savefig('plot_output/age_ebv_ml_2_c123.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_ebv_ml_2_c123.pdf', bbox_inches='tight', dpi=300)


exit()




ax_hum[0, 0].set_xlim(5.7, 10.3)
ax_hum[0, 0].set_ylim(-0.1, 2.1)
fig_hum.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)
ax_hum[4, 3].scatter([], [], c=color_c1, s=30, label='Class 1')
ax_hum[4, 3].scatter([], [], c=color_c2, s=30, label='Class 2')
ax_hum[4, 3].scatter([], [], c=color_c3, s=30, label='Compact Associations')
ax_hum[4, 3].legend(frameon=False, fontsize=fontsize-3)
ax_hum[4, 3].axis('off')


ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(-0.1, 2.1)
fig_ml.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'E(B-V) + random.uniform(-0.05, 0.05)', va='center', rotation='vertical', fontsize=fontsize)
ax_ml[4, 3].scatter([], [], c='darkorange', s=30, label='Class 1')
ax_ml[4, 3].scatter([], [], c='forestgreen', s=30, label='Class 2')
ax_ml[4, 3].scatter([], [], c='royalblue', s=30, label='Compact Associations')
ax_ml[4, 3].legend(frameon=False, fontsize=fontsize-3)
ax_ml[4, 3].axis('off')
# plt.tight_layout()
fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/age_ebv_hum_2_c12.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_ebv_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/age_ebv_ml_2_c12.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_ebv_ml_2.pdf', bbox_inches='tight', dpi=300)



