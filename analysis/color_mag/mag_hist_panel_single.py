import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


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
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')



abs_v_mag_bins = np.linspace(19, 26, 15)
fontsize = 21


fig, ax = plt.subplots(8, 5, sharex=True, sharey=True, figsize=(21, 26))
row_index = 0
col_index = 0
dim_targets = []
for index in range(0, 39):
    target = target_list[index]
    dist = dist_list[index]

    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)
    v_mag_hum = np.concatenate([v_mag_hum_12, v_mag_hum_3])

    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml')
    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml', cluster_class='class3')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)
    v_mag_ml = np.concatenate([v_mag_ml_12, v_mag_ml_3])

    ax[row_index, col_index].hist(v_mag_ml, bins=abs_v_mag_bins, density=True, histtype='step', color='tab:grey', linewidth=3)
    ax[row_index, col_index].hist(v_mag_hum, bins=abs_v_mag_bins, density=True, histtype='step', color='tab:red', linewidth=3)

    median_v_mag_ml = np.nanmedian(v_mag_ml[(v_mag_ml > 18) & (v_mag_ml < 26)])
    faintest_v_band_mag_hum = np.nanmax(v_mag_hum[(v_mag_hum > 18) & (v_mag_hum < 26)])
    faintest_v_band_mag_ml = np.nanmax(v_mag_ml[(v_mag_ml > 18) & (v_mag_ml < 26)])
    print('median_v_mag_ml ', median_v_mag_ml,  ' faintest_v_band_mag_hum ', faintest_v_band_mag_hum)
    # get the apparent mag limit of M_v = -6
    mag_lim = -6 + 25 + 5*np.log10(dist)
    ax[row_index, col_index].plot([median_v_mag_ml, median_v_mag_ml], [0, 0.9], color='tab:grey', linestyle='--', linewidth=3)
    ax[row_index, col_index].plot([mag_lim, mag_lim], [0, 0.9], color='k', linestyle='-', linewidth=3)

    if median_v_mag_ml > faintest_v_band_mag_hum:
        dim_targets.append(target)
        description_str = (target.upper() + '$^{*}$' + ('\nd= %.1f Mpc' % dist) +
                           ('\nMax' + r'$_{\rm Hum}$= %.1f' % faintest_v_band_mag_hum) +
                           ('\nMax' + r'$_{\rm ML}$= %.1f' % faintest_v_band_mag_ml))
    else:
        description_str = (target.upper() + ('\nd= %.1f Mpc' % dist) +
                           ('\nMax' + r'$_{\rm Hum}$= %.1f' % faintest_v_band_mag_hum) +
                           ('\nMax' + r'$_{\rm ML}$= %.1f' % faintest_v_band_mag_ml))
    anchored_left = AnchoredText(description_str, loc='upper right', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
    ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 5:
        row_index += 1
        col_index = 0

fig.text(0.5, 0.09, 'V [mag]', ha='center', fontsize=fontsize)
fig.text(0.085, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=fontsize)

ax[7, 4].plot([], [], color='tab:red', linewidth=2, label='Human')
ax[7, 4].plot([], [], color='tab:grey', linewidth=2, label='ML')
ax[7, 4].legend(frameon=False, fontsize=fontsize)

ax[0, 0].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[0, 0].set_xticks([24, 22, 20])
ax[0, 0].set_xlim(25.9, 15.5)

ax[7, 4].axis('off')

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/v_mag.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/v_mag.pdf', bbox_inches='tight', dpi=300)

print(dim_targets)

exit()


bins = np.linspace(-14, -3, 25)

fig, ax = plt.subplots(nrows=4)

ax[0].hist(abs_v_band_mag_hum, bins=np.linspace(-12, -4, 25), density=True, color='tab:blue', histtype='step')
ax[0].hist(abs_v_band_mag_ml, bins=np.linspace(-12, -4, 25), density=True, color='tab:orange', histtype='step')
ax[1].hist(abs_v_band_mag_hum[cluster_class_hum == 1], bins=np.linspace(-12, -4, 25), density=True, color='tab:blue', linestyle='--', histtype='step')
ax[1].hist(abs_v_band_mag_ml[(cluster_class_ml == 1) & (cluster_class_qual_ml >= 0.9)], bins=np.linspace(-12, -4, 25), density=True, color='tab:orange', linestyle='--', histtype='step')
ax[2].hist(abs_v_band_mag_hum[cluster_class_hum == 2], bins=np.linspace(-12, -4, 25), density=True, color='tab:blue', linestyle=':', histtype='step')
ax[2].hist(abs_v_band_mag_ml[(cluster_class_ml == 2) & (cluster_class_qual_ml >= 0.9)], bins=np.linspace(-12, -4, 25), density=True, color='tab:orange', linestyle=':', histtype='step')
ax[3].hist(abs_v_band_mag_hum[cluster_class_hum == 3], bins=np.linspace(-12, -4, 25), density=True, color='tab:blue', linestyle='dashdot', histtype='step')
ax[3].hist(abs_v_band_mag_ml[(cluster_class_ml == 3) & (cluster_class_qual_ml >= 0.9)], bins=np.linspace(-12, -4, 25), density=True, color='tab:orange', linestyle='dashdot', histtype='step')




plt.show()


exit()



    #
    #
    # ax_hum[row_index, col_index].scatter(color_ub_hum_3[cluster_class_hum_3 == 3],
    #                                      abs_v_mag_hum_3[cluster_class_hum_3 == 3],
    #                                      c='royalblue', s=1)
    # ax_hum[row_index, col_index].scatter(color_ub_hum_12[cluster_class_hum_12 == 1],
    #                                      abs_v_mag_hum_12[cluster_class_hum_12 == 1],
    #                                      c='forestgreen', s=1)
    # ax_hum[row_index, col_index].scatter(color_ub_hum_12[cluster_class_hum_12 == 2],
    #                                      abs_v_mag_hum_12[cluster_class_hum_12 == 2],
    #                                      c='darkorange', s=1)
    #
    # ax_ml[row_index, col_index].scatter(color_ub_ml_3[cluster_class_ml_3 == 3],
    #                                     abs_v_mag_ml_3[cluster_class_ml_3 == 3],
    #                                      c='royalblue', s=1)
    # ax_ml[row_index, col_index].scatter(color_ub_ml_12[cluster_class_ml_12 == 1],
    #                                     abs_v_mag_ml_12[cluster_class_ml_12 == 1],
    #                                      c='forestgreen', s=1)
    # ax_ml[row_index, col_index].scatter(color_ub_ml_12[cluster_class_ml_12 == 2],
    #                                     abs_v_mag_ml_12[cluster_class_ml_12 == 2],
    #                                      c='darkorange', s=1)
    #
    #
    # if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
    #     anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
    #                                  loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
    #     ax_hum[row_index, col_index].add_artist(anchored_left)
    #     anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
    #                                  loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
    #     ax_ml[row_index, col_index].add_artist(anchored_left)
    # else:
    #     anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
    #                                  loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
    #     ax_hum[row_index, col_index].add_artist(anchored_left)
    #     anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
    #                                  loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
    #     ax_ml[row_index, col_index].add_artist(anchored_left)
    #
    # ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
    #                                          direction='in', labelsize=fontsize)
    # ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
    #                                         direction='in', labelsize=fontsize)
    #
    #
    # col_index += 1
    # if col_index == 4:
    #     row_index += 1
    #     col_index = 0


ax_hum[0, 0].set_xlim(-1.0, 2.3)
ax_hum[0, 0].set_ylim(-4.5, -11.1)
# fig_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
# fig_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_hum.text(0.5, 0.89, 'Class 1|2|3 Human', ha='center', fontsize=fontsize)

ax_ml[0, 0].set_xlim(-1.0, 2.3)
ax_ml[0, 0].set_ylim(-4.5, -11.1)

# ax_ml[0, 0].set_ylim(1.25, -2.2)
# fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
# fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/vi_vmag_hum_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/vi_vmag_hum_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/vi_vmag_ml_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/vi_vmag_ml_1.pdf', bbox_inches='tight', dpi=300)


exit()

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
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_ml_12 == 1) & (cluster_class_qual_ml_12 >= 0.9)
    class_2_ml = (cluster_class_ml_12 == 2) & (cluster_class_qual_ml_12 >= 0.9)
    class_3_ml = (cluster_class_ml_3 == 3) & (cluster_class_qual_ml_3 >= 0.9)

    ax_hum[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_hum[row_index, col_index].scatter(color_vi_hum_3[cluster_class_hum_3 == 3],
                                         color_ub_hum_3[cluster_class_hum_3 == 3], c='royalblue', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 1],
                                         color_ub_hum_12[cluster_class_hum_12 == 1], c='forestgreen', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 2],
                                         color_ub_hum_12[cluster_class_hum_12 == 2], c='darkorange', s=1)

    ax_ml[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_ml[row_index, col_index].scatter(color_vi_ml_3[class_3_ml], color_ub_ml_3[class_3_ml], c='royalblue', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_1_ml], color_ub_ml_12[class_1_ml], c='forestgreen', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_2_ml], color_ub_ml_12[class_2_ml], c='darkorange', s=1)

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
fig_hum.savefig('plot_output/ub_vi_hum_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_2.pdf', bbox_inches='tight', dpi=300)

