import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                            #hst_cc_ver='IR4'
                                                            )

target_list = catalog_access.target_hst_cc

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


apparent_v_band_mag_hum = np.array([])
apparent_v_band_mag_ml = np.array([])

abs_v_band_mag_hum = np.array([])
abs_v_band_mag_ml = np.array([])

mstar_mag_hum = np.array([])
mstar_mag_ml = np.array([])

hum_class_hum = np.array([])
hum_class_ml = np.array([])
vgg_class_ml = np.array([])
vgg_class_qual_ml = np.array([])


dist_array_ml = np.array([])

for target in target_list:

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    dist = catalog_access.dist_dict[galaxy_name]['dist']
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)
    hum_class_hum = np.concatenate([hum_class_hum, cluster_class_hum_12, cluster_class_hum_3])
    abs_v_band_mag_hum = np.concatenate([abs_v_band_mag_hum, abs_v_mag_hum_12, abs_v_mag_hum_3])
    apparent_v_band_mag_hum = np.concatenate([apparent_v_band_mag_hum, v_mag_hum_12, v_mag_hum_3])

    vgg_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    vgg_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    hum_class_ml_12 = catalog_access.get_hst_cc_class_human(target=target, classify='ml')
    hum_class_ml_3 = catalog_access.get_hst_cc_class_human(target=target, classify='ml', cluster_class='class3')
    vgg_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    vgg_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')


    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml')
    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml', cluster_class='class3')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)

    vgg_class_ml = np.concatenate([vgg_class_ml, vgg_class_ml_12, vgg_class_ml_3])
    hum_class_ml = np.concatenate([hum_class_ml, hum_class_ml_12, hum_class_ml_3])
    vgg_class_qual_ml = np.concatenate([vgg_class_qual_ml, vgg_class_qual_ml_12, vgg_class_qual_ml_3])
    abs_v_band_mag_ml = np.concatenate([abs_v_band_mag_ml, abs_v_mag_ml_12, abs_v_mag_ml_3])
    apparent_v_band_mag_ml = np.concatenate([apparent_v_band_mag_ml, v_mag_ml_12, v_mag_ml_3])


    dist_array_ml = np.concatenate([dist_array_ml, np.ones(len(abs_v_mag_ml_12)+len(abs_v_mag_ml_3))*dist])


mask_hum_classified = (hum_class_ml > 0)
mask_vgg_cl1_ml = vgg_class_ml == 1
mask_vgg_cl2_ml = vgg_class_ml == 2
mask_vgg_cl3_ml = vgg_class_ml == 3

mask_hum_cl1_hum = hum_class_hum == 1
mask_hum_cl2_hum = hum_class_hum == 2
mask_hum_cl3_hum = hum_class_hum == 3



fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(13, 17))
fontsize = 25
bins = np.linspace(-14.5, -3.5, 30)


ax[0].hist(abs_v_band_mag_hum, bins=bins, linewidth=3, linestyle=':', color='tab:red', histtype='step')
ax[0].hist(abs_v_band_mag_ml, bins=bins, linewidth=3, linestyle=':', color='tab:gray', histtype='step')

ax[1].hist(abs_v_band_mag_hum, bins=bins, linewidth=3, linestyle=':', color='tab:red', histtype='step')
ax[1].hist(abs_v_band_mag_ml, bins=bins, linewidth=3, linestyle=':', color='tab:gray', histtype='step')

ax[2].hist(abs_v_band_mag_hum, bins=bins, linewidth=3, linestyle=':', color='tab:red', histtype='step')
ax[2].hist(abs_v_band_mag_ml, bins=bins, linewidth=3, linestyle=':', color='tab:gray', histtype='step')

ax[0].hist(abs_v_band_mag_hum[mask_hum_cl1_hum], bins=bins, linewidth=3, linestyle='-', color='tab:red', histtype='step')
ax[0].hist(abs_v_band_mag_ml[mask_vgg_cl1_ml], bins=bins, linewidth=3, linestyle='-', color='tab:gray', histtype='step')

ax[1].hist(abs_v_band_mag_hum[mask_hum_cl2_hum], bins=bins, linewidth=3, linestyle='-', color='tab:red', histtype='step')
ax[1].hist(abs_v_band_mag_ml[mask_vgg_cl2_ml], bins=bins, linewidth=3, linestyle='-', color='tab:gray', histtype='step')

ax[2].hist(abs_v_band_mag_hum[mask_hum_cl3_hum], bins=bins, linewidth=3, linestyle='-', color='tab:red', histtype='step')
ax[2].hist(abs_v_band_mag_ml[mask_vgg_cl3_ml], bins=bins, linewidth=3, linestyle='-', color='tab:gray', histtype='step')


ax[0].plot([], [], color='tab:red', linewidth=3, linestyle=':', label='Hum (N = %i)' % len(abs_v_band_mag_hum))
ax[0].plot([], [], color='tab:grey', linewidth=3, linestyle=':', label='ML (N = %i)' % len(abs_v_band_mag_ml))

ax[0].plot([], [], color='tab:red', linewidth=3, linestyle='-', label='Hum Class 1 (N = %i)' % np.sum(mask_hum_cl1_hum))
ax[0].plot([], [], color='tab:grey', linewidth=3, linestyle='-', label='ML Class 1 (N = %i)' % np.sum(mask_vgg_cl1_ml))

ax[1].plot([], [], color='tab:red', linewidth=3, linestyle='-', label='Hum Class 2 (N = %i)' % np.sum(mask_hum_cl2_hum))
ax[1].plot([], [], color='tab:grey', linewidth=3, linestyle='-', label='ML Class 2 (N = %i)' % np.sum(mask_vgg_cl2_ml))

ax[2].plot([], [], color='tab:red', linewidth=3, linestyle='-', label='Hum Class 3 (N = %i)' % np.sum(mask_hum_cl3_hum))
ax[2].plot([], [], color='tab:grey', linewidth=3, linestyle='-', label='ML Class 3 (N = %i)' % np.sum(mask_vgg_cl3_ml))

ax[0].legend(frameon=False, loc='lower center', fontsize=fontsize)
ax[1].legend(frameon=False, loc='lower center', fontsize=fontsize)
ax[2].legend(frameon=False, loc='lower center', fontsize=fontsize)


ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')

ax[0].set_xlim(-3.4, -14.9)

ax[0].set_ylabel('# Clusters', fontsize=fontsize)
ax[1].set_ylabel('# Clusters', fontsize=fontsize)
ax[2].set_ylabel('# Clusters', fontsize=fontsize)

ax[2].set_xlabel(r'M$_{\rm V}$ [mag]', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0.01)
fig.savefig('plot_output/abs_v_mag_hist_classes.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/abs_v_mag_hist_classes.pdf', bbox_inches='tight', dpi=300)


print(sum(abs_v_band_mag_ml < -10))
print(sum(abs_v_band_mag_hum < -10))

print('diff ', sum(abs_v_band_mag_hum < -10) - sum(abs_v_band_mag_ml < -10))

print('median ML', np.nanmedian(abs_v_band_mag_ml))
print('median Human', np.nanmedian(abs_v_band_mag_hum))

print('percentile 99 ML', np.nanpercentile(abs_v_band_mag_ml, 99))
print('percentile 99 Human', np.nanpercentile(abs_v_band_mag_hum, 99))

print(np.nanmedian(abs_v_band_mag_ml[dist_array_ml < 14]))
print(np.nanmedian(abs_v_band_mag_ml[dist_array_ml > 14]))




exit()



fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(20, 10))
fontsize = 25
bins = np.linspace(-14.5, -3.5, 30)

ax[0].hist(abs_v_band_mag_ml[cluster_class_ml==1], bins=bins, linewidth=2, color='tab:grey', histtype='step')
ax[0].hist(abs_v_band_mag_hum[cluster_class_hum==1], bins=bins, linewidth=2, color='tab:red', histtype='step')

ax[1].hist(abs_v_band_mag_ml[cluster_class_ml==2], bins=bins, linewidth=2, color='tab:grey', histtype='step')
ax[1].hist(abs_v_band_mag_hum[cluster_class_hum==2], bins=bins, linewidth=2, color='tab:red', histtype='step')

ax[2].hist(abs_v_band_mag_ml[cluster_class_ml==3], bins=bins, linewidth=2, color='tab:grey', histtype='step')
ax[2].hist(abs_v_band_mag_hum[cluster_class_hum==3], bins=bins, linewidth=2, color='tab:red', histtype='step')


ax[0].plot([], [], color='tab:red', linewidth=2, label='Human (N = %i)' % sum(cluster_class_hum==1))
ax[0].plot([], [], color='tab:grey', linewidth=2, label='ML (N = %i)' % sum(cluster_class_ml==1))

ax[1].plot([], [], color='tab:red', linewidth=2, label='Human (N = %i)' % sum(cluster_class_hum==2))
ax[1].plot([], [], color='tab:grey', linewidth=2, label='ML (N = %i)' % sum(cluster_class_ml==2))

ax[2].plot([], [], color='tab:red', linewidth=2, label='Human (N = %i)' % sum(cluster_class_hum==3))
ax[2].plot([], [], color='tab:grey', linewidth=2, label='ML (N = %i)' % sum(cluster_class_ml==3))

# percentile_position = np.percentile(abs_v_band_mag_ex_ml, len(abs_v_band_mag_ex_hum)/len(abs_v_band_mag_ex_ml) * 100)
# ax[1].plot([percentile_position, percentile_position], [0, 400], color='k', linewidth=2, linestyle='--',
#            label=r'ML Percentile (N$_{\rm Hum}$/N$_{\rm ML}$) = %.1f' % percentile_position)

ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')
# ax[1].set_yscale('log')


ax[0].legend(frameon=False, fontsize=fontsize)
ax[1].legend(frameon=False, fontsize=fontsize)
ax[2].legend(frameon=False, fontsize=fontsize)


ax[0].set_title('Class 1', fontsize=fontsize)
ax[1].set_title('Class 2', fontsize=fontsize)
ax[2].set_title('Class 3', fontsize=fontsize)


ax[0].set_xlim(-3.4, -14.9)
ax[1].set_xlim(-3.4, -14.9)
ax[2].set_xlim(-3.4, -14.9)
ax[0].set_ylabel('# Clusters', fontsize=fontsize)
ax[0].set_xlabel(r'M$_{\rm V}$ [mag]', fontsize=fontsize)
ax[1].set_xlabel(r'M$_{\rm V}$ [mag]', fontsize=fontsize)
ax[2].set_xlabel(r'M$_{\rm V}$ [mag]', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/abs_v_mag_hist_class.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/abs_v_mag_hist_class.pdf', bbox_inches='tight', dpi=300)


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

