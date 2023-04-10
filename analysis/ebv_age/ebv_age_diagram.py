import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)


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
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
    class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
    class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

    ax_hum[row_index, col_index].scatter((np.log10(age_3_hum[cluster_class_3_hum == 3]) + 6),
                                         ebv_3_hum[cluster_class_3_hum == 3], c='royalblue', s=1)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[cluster_class_12_hum == 1],
                                         ebv_12_hum[cluster_class_12_hum == 1], c='forestgreen', s=1)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[cluster_class_12_hum == 2],
                                         ebv_12_hum[cluster_class_12_hum == 2], c='darkorange', s=1)

    ax_ml[row_index, col_index].scatter((np.log10(age_3_ml[class_3_ml]) + 6), ebv_3_ml[class_3_ml], c='royalblue', s=1)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[class_1_ml], ebv_12_ml[class_1_ml], c='forestgreen', s=1)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[class_2_ml], ebv_12_ml[class_2_ml], c='darkorange', s=1)

    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_hum[row_index, col_index].add_artist(anchored_left)
    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
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
fig_hum.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(-0.1, 2.1)
fig_ml.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)

# plt.tight_layout()
fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/age_ebv_hum_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_ebv_hum_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/age_ebv_ml_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_ebv_ml_1.pdf', bbox_inches='tight', dpi=300)



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
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
    class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
    class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

    ax_hum[row_index, col_index].scatter((np.log10(age_3_hum[cluster_class_3_hum == 3]) + 6),
                                         ebv_3_hum[cluster_class_3_hum == 3], c='royalblue', s=1)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[cluster_class_12_hum == 1],
                                         ebv_12_hum[cluster_class_12_hum == 1], c='forestgreen', s=1)
    ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[cluster_class_12_hum == 2],
                                         ebv_12_hum[cluster_class_12_hum == 2], c='darkorange', s=1)

    ax_ml[row_index, col_index].scatter((np.log10(age_3_ml[class_3_ml]) + 6), ebv_3_ml[class_3_ml], c='royalblue', s=1)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[class_1_ml], ebv_12_ml[class_1_ml], c='forestgreen', s=1)
    ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + 6)[class_2_ml], ebv_12_ml[class_2_ml], c='darkorange', s=1)

    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax_hum[row_index, col_index].add_artist(anchored_left)
    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
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
fig_hum.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)
ax_hum[4, 3].axis('off')

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(-0.1, 2.1)
fig_ml.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)
ax_ml[4, 3].axis('off')

# plt.tight_layout()
fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/age_ebv_hum_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_ebv_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/age_ebv_ml_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_ebv_ml_2.pdf', bbox_inches='tight', dpi=300)



