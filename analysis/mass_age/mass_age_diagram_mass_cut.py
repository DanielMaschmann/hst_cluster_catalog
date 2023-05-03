import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)

# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod = hdu_a[1].data
age_mod = data_mod['sfh.age']
m_star_mod = data_mod['stellar.m_star']
flux_f555w_mod = data_mod['F555W_UVIS_CHIP2']

ABmag_F555W = - 6
f_mJy_f555w = 1.e3 * 1.e23 * 10.**((ABmag_F555W + 48.6) / -2.5)
lower_lim_m_star = np.log10(f_mJy_f555w * m_star_mod / flux_f555w_mod)


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


color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'darkorange'

m_star_cut = 3*1e4
age_cut = 7*1e2


fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    m_star_hum = np.concatenate([m_star_12_hum, m_star_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
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

    mask_complete_hum = (m_star_hum > m_star_cut) & (age_hum < age_cut)
    mask_complete_ml = (m_star_ml > m_star_cut) & (age_ml < age_cut)
    mask_qual_ml = cluster_class_qual_ml >= 0.9

    # plotting
    ax_hum[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=2)
    ax_hum[row_index, col_index].plot([np.log10(age_cut) + 1, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut)], color='k', linestyle='--', linewidth=1)
    ax_hum[row_index, col_index].plot([np.log10(age_cut) + 6, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut) + 5], color='k', linestyle='--', linewidth=1)

    ax_ml[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=2)
    ax_ml[row_index, col_index].plot([np.log10(age_cut) + 1, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut)], color='k', linestyle='--', linewidth=1)
    ax_ml[row_index, col_index].plot([np.log10(age_cut) + 6, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut) + 5], color='k', linestyle='--', linewidth=1)


    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_1_hum*mask_complete_hum], np.log10(m_star_hum)[class_1_hum*mask_complete_hum], c=color_c1, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_1_hum*~mask_complete_hum], np.log10(m_star_hum)[class_1_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_2_hum*mask_complete_hum], np.log10(m_star_hum)[class_2_hum*mask_complete_hum], c=color_c2, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_2_hum*~mask_complete_hum], np.log10(m_star_hum)[class_2_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_3_hum*mask_complete_hum], np.log10(m_star_hum)[class_3_hum*mask_complete_hum], c=color_c3, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_3_hum*~mask_complete_hum], np.log10(m_star_hum)[class_3_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)


    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_2_ml*mask_complete_ml], np.log10(m_star_ml)[class_2_ml*mask_complete_ml], c=color_c2, s=20, alpha=0.5)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_2_ml*~mask_complete_ml], np.log10(m_star_ml)[class_2_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)

    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_1_ml*mask_complete_ml], np.log10(m_star_ml)[class_1_ml*mask_complete_ml], c=color_c1, s=20, alpha=0.5)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_1_ml*~mask_complete_ml], np.log10(m_star_ml)[class_1_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)


    # ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_3_ml*mask_complete_ml], np.log10(m_star_ml)[class_3_ml*mask_complete_ml], c=color_c3, s=20, alpha=0.5)
    # ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_3_ml*~mask_complete_ml], np.log10(m_star_ml)[class_3_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)


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
ax_hum[0, 0].set_ylim(1.9, 8.7)
fig_hum.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(1.9, 8.7)
fig_ml.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

# plt.tight_layout()
fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/age_m_star_hum_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_m_star_hum_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/age_m_star_ml_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_m_star_ml_1.pdf', bbox_inches='tight', dpi=300)



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
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    age_hum = np.concatenate([age_12_hum, age_3_hum])
    m_star_hum = np.concatenate([m_star_12_hum, m_star_3_hum])
    cluster_class_hum = np.concatenate([cluster_class_12_hum, cluster_class_3_hum])

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
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

    mask_complete_hum = (m_star_hum > m_star_cut) & (age_hum < age_cut)
    mask_complete_ml = (m_star_ml > m_star_cut) & (age_ml < age_cut)
    mask_qual_ml = cluster_class_qual_ml >= 0.9

    # plotting
    ax_hum[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=2)
    ax_hum[row_index, col_index].plot([np.log10(age_cut) + 1, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut)], color='k', linestyle='--', linewidth=1)
    ax_hum[row_index, col_index].plot([np.log10(age_cut) + 6, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut) + 5], color='k', linestyle='--', linewidth=1)

    ax_ml[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=2)
    ax_ml[row_index, col_index].plot([np.log10(age_cut) + 1, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut)], color='k', linestyle='--', linewidth=1)
    ax_ml[row_index, col_index].plot([np.log10(age_cut) + 6, np.log10(age_cut) + 6], [np.log10(m_star_cut), np.log10(m_star_cut) + 5], color='k', linestyle='--', linewidth=1)

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_1_hum*mask_complete_hum], np.log10(m_star_hum)[class_1_hum*mask_complete_hum], c=color_c1, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_1_hum*~mask_complete_hum], np.log10(m_star_hum)[class_1_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_2_hum*mask_complete_hum], np.log10(m_star_hum)[class_2_hum*mask_complete_hum], c=color_c2, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_2_hum*~mask_complete_hum], np.log10(m_star_hum)[class_2_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)

    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_3_hum*mask_complete_hum], np.log10(m_star_hum)[class_3_hum*mask_complete_hum], c=color_c3, s=20, alpha=0.5)
    ax_hum[row_index, col_index].scatter((np.log10(age_hum) + 6)[class_3_hum*~mask_complete_hum], np.log10(m_star_hum)[class_3_hum*~mask_complete_hum], c='grey', s=20, alpha=0.5)

    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_2_ml*mask_complete_ml], np.log10(m_star_ml)[class_2_ml*mask_complete_ml], c=color_c2, s=20, alpha=0.5)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_2_ml*~mask_complete_ml], np.log10(m_star_ml)[class_2_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)

    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_1_ml*mask_complete_ml], np.log10(m_star_ml)[class_1_ml*mask_complete_ml], c=color_c1, s=20, alpha=0.5)
    ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_1_ml*~mask_complete_ml], np.log10(m_star_ml)[class_1_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)


    # ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_3_ml*mask_complete_ml], np.log10(m_star_ml)[class_3_ml*mask_complete_ml], c=color_c3, s=20, alpha=0.5)
    # ax_ml[row_index, col_index].scatter((np.log10(age_ml) + 6)[class_3_ml*~mask_complete_ml], np.log10(m_star_ml)[class_3_ml*~mask_complete_ml], c='grey', s=20, alpha=0.5)


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
ax_hum[0, 0].set_ylim(1.9, 8.7)
fig_hum.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)
ax_hum[4, 3].scatter([], [], c=color_c1, s=30, label='Class 1')
ax_hum[4, 3].scatter([], [], c=color_c2, s=30, label='Class 2')
ax_hum[4, 3].legend(frameon=False, fontsize=fontsize)
ax_hum[4, 3].axis('off')

ax_ml[0, 0].set_xlim(5.7, 10.3)
ax_ml[0, 0].set_ylim(1.9, 8.7)
fig_ml.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)
ax_ml[4, 3].scatter([], [], c=color_c1, s=30, label='Class 1')
ax_ml[4, 3].scatter([], [], c=color_c2, s=30, label='Class 2')
ax_ml[4, 3].legend(frameon=False, fontsize=fontsize)
ax_ml[4, 3].axis('off')

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/age_m_star_hum_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/age_m_star_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/age_m_star_ml_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/age_m_star_ml_2.pdf', bbox_inches='tight', dpi=300)

