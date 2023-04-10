
import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
import ast
from astropy.io import ascii
from astropy.table import QTable




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



color_ub = np.array([])
color_vi = np.array([])
ebv = np.array([])
age = np.array([])
cluster_class = np.array([])
modality = np.array([])
det = np.array([])
likelihood_norm = np.array([])
chi2 = np.array([])

for index in range(0, 39):
    target = target_list[index]
    dist = dist_list[index]

    if (target[0:3] == 'ngc') & (target[3] == '0'):
        target_string = target[0:3] + target[4:]
    else:
        target_string = target

    print('target ', target, 'dist ', dist)

    try:
        table = QTable.read('/home/benutzer/data/PHANGS_products/HST_catalog_multi_modal/%s_modified_input_temp.csv' % target_string)
    except FileNotFoundError:
        print('NOOOOOOOOOOOOOOO')
        continue

    # print(table.colnames)
    # print(table['peak_likelihoods'])
    # print(table['peak_likelihoods_norm'])
    # print()
    # exit()

    age = np.concatenate([age, table['PHANGS_AGE_MINCHISQ']])
    ebv = np.concatenate([ebv, table['PHANGS_EBV_MINCHISQ']])
    cluster_class = np.concatenate([cluster_class, table['PHANGS_CLUSTER_CLASS']])
    modality = np.concatenate([modality, table['num_kde_age_ebv_joint_pd_peaks']])
    det = np.concatenate([det, table['PHANGS_NON_DETECTION_FLAG']])
    likelihood_norm = np.concatenate([likelihood_norm, table['peak_likelihoods_norm']])
    chi2 = np.concatenate([chi2, table['PHANGS_REDUCED_MINCHISQ']])

    color_u = table['PHANGS_F336W_VEGA_TOT']
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        color_b = table['PHANGS_F438W_VEGA_TOT']
    else:
        color_b = table['PHANGS_F435W_VEGA_TOT']
    color_v = table['PHANGS_F555W_VEGA_TOT']
    color_i = table['PHANGS_F814W_VEGA_TOT']

    color_ub = np.concatenate([color_ub, color_u - color_b])
    color_vi = np.concatenate([color_vi, color_v - color_i])


plt.hist(chi2[modality == 1], bins=np.linspace(0, 4, 20), density=True, histtype='step', label='1 Mode')
plt.hist(chi2[modality == 2], bins=np.linspace(0, 4, 20), density=True, histtype='step', label='2 Modes')
plt.hist(chi2[modality == 3], bins=np.linspace(0, 4, 20), density=True, histtype='step', label='4 Modes')
plt.legend()
plt.xlabel('Chi2')
plt.savefig('plot_output/hist_chi2.png')
plt.clf()

mode_2_low = []
mode_2_high = []
for index in range(len(likelihood_norm[modality == 2])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 2][index]))
    where_low = np.where(norm_array != 1.0)
    where_high = np.where(norm_array == 1.0)
    mode_2_low.append(norm_array[where_low])
    mode_2_high.append(norm_array[where_high])

mode_3_low = []
mode_3_second = []
mode_3_high = []

for index in range(len(likelihood_norm[modality == 3])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 3][index]))
    low = np.where(norm_array == np.min(norm_array))
    second = np.where((norm_array != 1.0) & (norm_array != np.min(norm_array)))
    high = np.where(norm_array == 1.0)

    mode_3_low.append(norm_array[low])
    mode_3_second.append(norm_array[second])
    mode_3_high.append(norm_array[high])

fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
bins = np.linspace(0, 1.0, 11)
ax[0].hist(np.array(mode_2_low), bins=bins, color='tab:blue', linewidth=4, histtype='step', label='second peak')
ax[1].hist(np.array(mode_3_low), bins=bins, color='tab:blue', linewidth=4, histtype='step', label='lowest peak')
ax[1].hist(np.array(mode_3_second), bins=bins, color='tab:orange', linewidth=4, histtype='step', label='second highest peak')

ax[0].legend(fontsize=15)
ax[1].legend(fontsize=15)
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=15)

ax[0].set_ylabel('counts', fontsize=15)
ax[1].set_ylabel('counts', fontsize=15)
ax[1].set_xlabel('likelihood ratio', fontsize=15)

plt.savefig('plot_output/likelihood_ratio.png')
plt.clf()



fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(28, 35))
fontsize = 19

threshold_1 = 0.4
threshold_2 = 0.8



ax[0, 0].plot(model_vi, model_ub, color='r', linewidth=3)
ax[0, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[0, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[1, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[1, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[2, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[2, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[0, 0].scatter(color_vi[modality == 1], color_ub[modality == 1], color='tab:blue', marker='.', s=20)
ax[1, 0].axis('off')
ax[2, 0].axis('off')
ax[0, 1].scatter(color_vi[modality == 2], color_ub[modality == 2], color='tab:blue', marker='.', s=20)
ax[0, 2].scatter(color_vi[modality == 3], color_ub[modality == 3], color='tab:blue', marker='.', s=20)

ax[0, 0].set_title('1 Mode', fontsize=fontsize)
ax[0, 1].set_title('2 Modes', fontsize=fontsize)
ax[0, 2].set_title('3 Modes', fontsize=fontsize)

for index in range(len(likelihood_norm[modality == 2])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 2][index]))
    if sum(norm_array < threshold_1) == 1:
        ax[1, 1].scatter(color_vi[modality == 2][index], color_ub[modality == 2][index], color='tab:blue', marker='.', s=20)
    if sum((norm_array > threshold_2) & (norm_array != 1)) == 1:
        ax[2, 1].scatter(color_vi[modality == 2][index], color_ub[modality == 2][index], color='tab:blue', marker='.', s=20)

for index in range(len(likelihood_norm[modality == 3])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 3][index]))
    if sum(norm_array < threshold_1) == 1:
        ax[1, 2].scatter(color_vi[modality == 3][index], color_ub[modality == 3][index], color='tab:blue', marker='.', s=20)
    if sum((norm_array > threshold_2) & (norm_array != 1)) == 1:
        ax[2, 2].scatter(color_vi[modality == 3][index], color_ub[modality == 3][index], color='tab:blue', marker='.', s=20)


ax[0, 0].set_ylim(1.25, -2.2)
ax[0, 0].set_xlim(-1.0, 2.3)

ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 1].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[2, 1].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax[1, 1].scatter([], [], color='tab:blue', marker='.', s=20, label='LR < 0.4')
ax[2, 1].scatter([], [], color='tab:blue', marker='.', s=20, label='LR > 0.8')
ax[1, 2].scatter([], [], color='tab:blue', marker='.', s=20, label='LR < 0.4')
ax[2, 2].scatter([], [], color='tab:blue', marker='.', s=20, label='LR > 0.8')

ax[1, 1].legend(frameon=False, fontsize=fontsize)
ax[2, 1].legend(frameon=False, fontsize=fontsize)
ax[1, 2].legend(frameon=False, fontsize=fontsize)
ax[2, 2].legend(frameon=False, fontsize=fontsize)

plt.savefig('plot_output/multi_modals_likelihood_ratio.png')
plt.clf()




fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(28, 35))
fontsize = 19


ax[0, 0].plot(model_vi, model_ub, color='r', linewidth=3)
ax[0, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[0, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[1, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[1, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[2, 1].plot(model_vi, model_ub, color='r', linewidth=3)
ax[2, 2].plot(model_vi, model_ub, color='r', linewidth=3)

ax[0, 0].scatter(color_vi[modality == 1], color_ub[modality == 1], color='tab:blue', marker='.', s=20)
ax[1, 0].axis('off')
ax[2, 0].axis('off')

ax[0, 1].scatter(color_vi[modality == 2], color_ub[modality == 2], color='tab:blue', marker='.', s=20)

ax[1, 1].scatter(color_vi[(modality == 2) & (chi2 < 0.2)], color_ub[(modality == 2) & (chi2 < 0.2)], color='tab:blue', marker='.', s=20, label='chi2 < 0.2')
ax[2, 1].scatter(color_vi[(modality == 2) & (chi2 > 1.5)], color_ub[(modality == 2) & (chi2 > 1.5)], color='tab:blue', marker='.', s=20, label='chi2 > 1.5')

ax[0, 2].scatter(color_vi[modality == 3], color_ub[modality == 3], color='tab:blue', marker='.', s=20)
ax[1, 2].scatter(color_vi[(modality == 3) & (chi2 < 0.2)], color_ub[(modality == 3) & (chi2 < 0.2)], color='tab:blue', marker='.', s=20, label='chi2 < 0.2')
ax[2, 2].scatter(color_vi[(modality == 3) & (chi2 > 1.5)], color_ub[(modality == 3) & (chi2 > 1.5)], color='tab:blue', marker='.', s=20, label='chi2 > 1.5')

ax[0, 0].set_title('1 Mode', fontsize=fontsize)
ax[0, 1].set_title('2 Modes', fontsize=fontsize)
ax[0, 2].set_title('3 Modes', fontsize=fontsize)



ax[0, 0].set_ylim(1.25, -2.2)
ax[0, 0].set_xlim(-1.0, 2.3)

ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 1].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[2, 1].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax[1, 1].legend(frameon=False, fontsize=fontsize)
ax[2, 1].legend(frameon=False, fontsize=fontsize)
ax[1, 2].legend(frameon=False, fontsize=fontsize)
ax[2, 2].legend(frameon=False, fontsize=fontsize)


plt.savefig('plot_output/multi_modals_chi2.png')
plt.clf()




fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(28, 35))
fontsize = 19

threshold_1 = 0.4
threshold_2 = 0.8



ax[0, 0].scatter(age, ebv, marker='.', label='1 MODE')
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 0].legend(frameon=False, fontsize=fontsize)

ax[0, 1].scatter(age[modality==2], ebv[modality==2], marker='.', color='tab:orange', label='2 MODEs')
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].legend(frameon=False, fontsize=fontsize)


ax[0, 2].scatter(age[modality==3], ebv[modality==3], marker='.',  color='tab:green', label='3 MODEs')
ax[0, 2].set_xscale('log')
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].legend(frameon=False, fontsize=fontsize)

ax[1, 0].axis('off')
ax[2, 0].axis('off')

ax[0, 0].set_title('1 Mode', fontsize=fontsize)
ax[0, 1].set_title('2 Modes', fontsize=fontsize)
ax[0, 2].set_title('3 Modes', fontsize=fontsize)

for index in range(len(likelihood_norm[modality == 2])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 2][index]))
    if sum(norm_array < threshold_1) == 1:
        ax[1, 1].scatter(age[modality == 2][index], ebv[modality == 2][index], color='tab:blue', marker='.', s=20)
    if sum((norm_array > threshold_2) & (norm_array != 1)) == 1:
        ax[2, 1].scatter(age[modality == 2][index], ebv[modality == 2][index], color='tab:blue', marker='.', s=20)

for index in range(len(likelihood_norm[modality == 3])):
    print(index)
    norm_array = np.array(ast.literal_eval(likelihood_norm[modality == 3][index]))
    if sum(norm_array < threshold_1) == 1:
        ax[1, 2].scatter(age[modality == 3][index], ebv[modality == 3][index], color='tab:blue', marker='.', s=20)
    if sum((norm_array > threshold_2) & (norm_array != 1)) == 1:
        ax[2, 2].scatter(age[modality == 3][index], ebv[modality == 3][index], color='tab:blue', marker='.', s=20)


ax[1, 1].set_xscale('log')
ax[1, 2].set_xscale('log')
ax[2, 1].set_xscale('log')
ax[2, 2].set_xscale('log')

ax[0, 0].set_ylabel('E(B-V)', fontsize=fontsize)

ax[2, 0].set_xlabel('Age', fontsize=fontsize)
ax[2, 1].set_xlabel('Age', fontsize=fontsize)
ax[2, 2].set_xlabel('Age', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax[1, 1].scatter([], [], color='tab:blue', marker='.', s=20, label='LR < 0.4')
ax[2, 1].scatter([], [], color='tab:blue', marker='.', s=20, label='LR > 0.8')
ax[1, 2].scatter([], [], color='tab:blue', marker='.', s=20, label='LR < 0.4')
ax[2, 2].scatter([], [], color='tab:blue', marker='.', s=20, label='LR > 0.8')

ax[1, 1].legend(frameon=False, fontsize=fontsize)
ax[2, 1].legend(frameon=False, fontsize=fontsize)
ax[1, 2].legend(frameon=False, fontsize=fontsize)
ax[2, 2].legend(frameon=False, fontsize=fontsize)


plt.savefig('plot_output/age_red_multimode.png')
plt.clf()





fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(28, 35))
fontsize = 19

threshold_1 = 0.4
threshold_2 = 0.8



ax[0, 0].scatter(age, ebv, marker='.', label='1 MODE')
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 0].legend(frameon=False, fontsize=fontsize)

ax[0, 1].scatter(age[modality==2], ebv[modality==2], marker='.', color='tab:orange', label='2 MODEs')
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].legend(frameon=False, fontsize=fontsize)


ax[0, 2].scatter(age[modality==3], ebv[modality==3], marker='.',  color='tab:green', label='3 MODEs')
ax[0, 2].set_xscale('log')
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].legend(frameon=False, fontsize=fontsize)

ax[1, 1].scatter(age[(modality == 2) & (chi2 < 0.2)], ebv[(modality == 2) & (chi2 < 0.2)], color='tab:blue', marker='.', s=20, label='chi2 < 0.2')
ax[2, 1].scatter(age[(modality == 2) & (chi2 > 1.5)], ebv[(modality == 2) & (chi2 > 1.5)], color='tab:blue', marker='.', s=20, label='chi2 > 1.5')

ax[0, 2].scatter(age[modality == 3], ebv[modality == 3], color='tab:blue', marker='.', s=20)
ax[1, 2].scatter(age[(modality == 3) & (chi2 < 0.2)], ebv[(modality == 3) & (chi2 < 0.2)], color='tab:blue', marker='.', s=20, label='chi2 < 0.2')
ax[2, 2].scatter(age[(modality == 3) & (chi2 > 1.5)], ebv[(modality == 3) & (chi2 > 1.5)], color='tab:blue', marker='.', s=20, label='chi2 > 1.5')



ax[1, 0].axis('off')
ax[2, 0].axis('off')

ax[0, 0].set_title('1 Mode', fontsize=fontsize)
ax[0, 1].set_title('2 Modes', fontsize=fontsize)
ax[0, 2].set_title('3 Modes', fontsize=fontsize)



ax[1, 1].set_xscale('log')
ax[1, 2].set_xscale('log')
ax[2, 1].set_xscale('log')
ax[2, 2].set_xscale('log')

ax[0, 0].set_ylabel('E(B-V)', fontsize=fontsize)

ax[2, 0].set_xlabel('Age', fontsize=fontsize)
ax[2, 1].set_xlabel('Age', fontsize=fontsize)
ax[2, 2].set_xlabel('Age', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax[1, 1].legend(frameon=False, fontsize=fontsize)
ax[2, 1].legend(frameon=False, fontsize=fontsize)
ax[1, 2].legend(frameon=False, fontsize=fontsize)
ax[2, 2].legend(frameon=False, fontsize=fontsize)


plt.savefig('plot_output/age_red_multimode_chi2.png')
plt.clf()




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

mask_globe = (color_vi_1 > 0.95) & (color_vi_1 < 1.5) & (color_ub_1 > -0.6) & (color_ub_1 < 0.5)
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


slope, intersect = get_slop_inter(x1=6, x2=9, y1=1.0, y2=0.1)
mask_outliers = (ebv > slope * np.log10(age*1e6) + intersect) & (ebv > 0.1)

print(mask_outliers)
print(sum(mask_outliers))

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



