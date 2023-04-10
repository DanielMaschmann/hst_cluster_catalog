import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits


def get_slop_inter(x1, x2, y1, y2):
    slope = (y2 - y1) / (x2 - x1)
    intersect = y1 - x1 * (y2-y1) / (x2 - x1)
    return slope, intersect


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)

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


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')



color_ub_1 = np.array([])
color_vi_1 = np.array([])
ebv = np.array([])
age = np.array([])
chi2 = np.array([])

for index in range(0, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    ebv_12 = catalog_access.get_hst_cc_ebv(target=target)
    age_12 = catalog_access.get_hst_cc_age(target=target)
    chi2_12 = catalog_access.get_hst_cc_min_chi2(target=target)

    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    ebv_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    age_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')

    color_ub_1 = np.concatenate([color_ub_1, color_ub_hum_12[cluster_class_hum_12 == 1]])
    color_vi_1 = np.concatenate([color_vi_1, color_vi_hum_12[cluster_class_hum_12 == 1]])
    ebv = np.concatenate([ebv, ebv_12[cluster_class_hum_12 == 1]])
    age = np.concatenate([age, age_12[cluster_class_hum_12 == 1]])
    chi2 = np.concatenate([chi2, chi2_12[cluster_class_hum_12 == 1]])

mask_globe = (color_vi_1 > 0.95) & (color_vi_1 < 1.5) & (color_ub_1 > -0.6) & (color_ub_1 < 0.5)

slope, intersect = get_slop_inter(x1=6, x2=9, y1=1.0, y2=0.1)
mask_outliers = (ebv > slope * np.log10(age*1e6) + intersect) & (ebv > 0.1)

print(mask_outliers)
print(sum(mask_outliers))

mask_chi2 = chi2 < 1.0
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].plot(model_vi, model_ub, color='red', linewidth=1.2)
ax[1].plot(model_vi, model_ub, color='red', linewidth=1.2)
ax[0].scatter(color_vi_1[~mask_chi2], color_ub_1[~mask_chi2], marker='.')
ax[1].scatter(color_vi_1[mask_chi2], color_ub_1[mask_chi2], marker='.')
ax[0].set_ylim(1.25, -2.2)
ax[0].set_xlim(-1.0, 2.3)
plt.show()

exit()

# plt.hist(chi2[mask_outliers], bins=np.linspace(0, 5, 30), density=True, histtype='step', label='over')
# plt.hist(chi2[~mask_outliers], bins=np.linspace(0, 5, 30), density=True, histtype='step', label='other')
# plt.legend()
# plt.show()




exit()


fig = plt.figure(figsize=(20, 13))
fontsize = 19
ax_cc = fig.add_axes([0.07, 0.05, 0.43, 0.9])
ax_age_ebv = fig.add_axes([0.55, 0.05, 0.43, 0.9])


ax_cc.plot(model_vi, model_ub, color='red', linewidth=1.2)
ax_cc.plot([0.95, 0.95], [-0.6, 0.5], linewidth=2, color='tab:orange')
ax_cc.plot([1.5, 1.5], [-0.6, 0.5], linewidth=2, color='tab:orange')
ax_cc.plot([0.95, 1.5], [-0.6, -0.6], linewidth=2, color='tab:orange')
ax_cc.plot([0.95, 1.5], [0.5, 0.5], linewidth=2, color='tab:orange')
ax_cc.scatter(color_vi_1, color_ub_1, marker='.')
ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-1.0, 2.3)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)




ax_age_ebv.scatter(age[mask_globe] * 1e6, ebv[mask_globe])
ax_age_ebv.set_xscale('log')
ax_age_ebv.plot([1e6, 1e9], [1.0, 0.1], linewidth=2, color='tab:red')
ax_age_ebv.set_xlabel('log(Age/yr)', fontsize=fontsize)
ax_age_ebv.set_ylabel('E(B-V)', fontsize=fontsize)
ax_age_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.savefig('plot_output/age_ebc_globular.png')

plt.clf()


fig = plt.figure(figsize=(20, 13))
fontsize = 19
ax_age_ebv = fig.add_axes([0.07, 0.05, 0.43, 0.9])
ax_cc = fig.add_axes([0.55, 0.05, 0.43, 0.9])




ax_age_ebv.scatter(age * 1e6, ebv)
ax_age_ebv.scatter(age[mask_outliers] * 1e6, ebv[mask_outliers])
ax_age_ebv.set_xscale('log')
ax_age_ebv.plot([1e6, 1e9], [1.0, 0.1], linewidth=2, color='tab:red')
ax_age_ebv.set_xlabel('log(Age/yr)', fontsize=fontsize)
ax_age_ebv.set_ylabel('E(B-V)', fontsize=fontsize)
ax_age_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)







ax_cc.plot(model_vi, model_ub, color='red', linewidth=1.2)
ax_cc.plot([0.95, 0.95], [-0.6, 0.5], linewidth=2, color='tab:orange')
ax_cc.plot([1.5, 1.5], [-0.6, 0.5], linewidth=2, color='tab:orange')
ax_cc.plot([0.95, 1.5], [-0.6, -0.6], linewidth=2, color='tab:orange')
ax_cc.plot([0.95, 1.5], [0.5, 0.5], linewidth=2, color='tab:orange')
ax_cc.scatter(color_vi_1, color_ub_1, marker='.')
ax_cc.scatter(color_vi_1[mask_outliers], color_ub_1[mask_outliers], marker='.')
ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-1.0, 2.3)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


plt.savefig('plot_output/age_ebc_globular_reverse.png')

plt.clf()







exit()




fig_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_hum.text(0.5, 0.89, 'Class 1|2|3 Human', ha='center', fontsize=fontsize)

ax_ml[0, 0].set_ylim(1.25, -2.2)
ax_ml[0, 0].set_xlim(-1.0, 2.3)
fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/ub_vi_hum_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_1.pdf', bbox_inches='tight', dpi=300)


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

