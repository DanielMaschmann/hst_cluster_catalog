import os.path

import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from scipy.stats import gaussian_kde
import dust_tools.extinction_tools
from matplotlib.colorbar import ColorbarBase
import matplotlib

def contours(ax, x, y, levels=None, axis_offse=(-0.2, 0.1, -0.55, 0.6)):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]

    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min()+axis_offse[0]:x.max()+axis_offse[1]:x.size**0.5*1j,
             y.min()+axis_offse[2]:y.max()+axis_offse[3]:y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    # ax[0].scatter(xi.flatten(), yi.flatten(), c=zi)
    cs = ax.contour(xi, yi, zi, levels=levels,
                    colors='k',
                    linewidths=(1,),
                    origin='lower')


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)


# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol

# get model
hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
flux_f814w_sol50 = data_mod_sol50['F814W_UVIS_CHIP2']
flux_f336w_sol50 = data_mod_sol50['F336W_UVIS_CHIP2']
flux_f438w_sol50 = data_mod_sol50['F438W_UVIS_CHIP2']
mag_v_sol50 = hf.conv_mjy2vega(flux=flux_f555w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol50 = hf.conv_mjy2vega(flux=flux_f814w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol50 = hf.conv_mjy2vega(flux=flux_f336w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol50 = hf.conv_mjy2vega(flux=flux_f438w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol50 = mag_v_sol50 - mag_i_sol50
model_ub_sol50 = mag_u_sol50 - mag_b_sol50



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


color_c1 = 'darkorange'
color_c2 = 'tab:green'

vi_int = 0.8
ub_int = -2.2
max_av = 2
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_vi_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av+0.1)
max_color_ext_ub_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av+0.1)

cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=5, vmax=100)

fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18


row_index = 0
col_index = 0
age_array = np.array([])
n_h_alpha_array = np.array([])
class_array = np.array([])
hii_array = np.array([])

for index in range(len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')
    age_hum_12 = catalog_access.get_hst_cc_age(target)
    age_hum_3 = catalog_access.get_hst_cc_age(target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    cluster_id_ml_12 = catalog_access.get_hst_cc_phangs_id(target, classify='ml')
    cluster_id_ml_3 = catalog_access.get_hst_cc_phangs_id(target, classify='ml', cluster_class='class3')
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')

    class_1_ml = (cluster_class_ml_12 == 1) # & (cluster_class_qual_ml_12 >= 0.9)
    class_2_ml = (cluster_class_ml_12 == 2) # & (cluster_class_qual_ml_12 >= 0.9)
    class_3_ml = (cluster_class_ml_3 == 3) # & (cluster_class_qual_ml_3 >= 0.9)

    color_ub_ml = np.concatenate([color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = np.concatenate([color_vi_ml_12, color_vi_ml_3])

    h_alpha_intensity_hum_12 = np.zeros(len(color_vi_hum_12))
    h_alpha_intensity_hum_3 = np.zeros(len(color_vi_hum_3))
    hii_hum_12 = np.zeros(len(color_vi_hum_12))
    hii_hum_3 = np.zeros(len(color_vi_hum_3))
    if target == 'ngc0685':
        target_ext = 'ngc685'
    elif target == 'ngc0628c':
        target_ext = 'ngc628c'
    elif target == 'ngc0628e':
        target_ext = 'ngc628e'
    else:
        target_ext = target
    extra_file = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
    if os.path.isfile(extra_file):
        extra_tab_hdu = fits.open(extra_file)
        extra_data_table = extra_tab_hdu[1].data
        extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
        extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
        extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev3']

        for running_index, cluster_id in enumerate(cluster_id_hum_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_12[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_hum_12[running_index] = extra_hii_mask[index_extra_table]
        for running_index, cluster_id in enumerate(cluster_id_hum_3):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_3[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_hum_3[running_index] = extra_hii_mask[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    age_array = np.concatenate([age_array, age_hum_12, age_hum_3])
    n_h_alpha_array = np.concatenate([n_h_alpha_array, h_alpha_intensity_hum_12, h_alpha_intensity_hum_3])
    class_array = np.concatenate([class_array, cluster_class_hum_12, cluster_class_hum_3])
    hii_array = np.concatenate([hii_array, hii_hum_12, hii_hum_3])



fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey='col', figsize=(20, 10))
fontsize = 15
bins = np.logspace(-1, 3, 21)

ax[0, 0].hist(n_h_alpha_array[(class_array==1) & (age_array < 5)], linewidth=2, bins=bins, histtype='step')
ax[0, 0].hist(n_h_alpha_array[(class_array==1) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step')
ax[0, 0].hist(n_h_alpha_array[(class_array==1) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step')
ax[0, 0].hist(n_h_alpha_array[(class_array==1) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 0].hist(n_h_alpha_array[(class_array==1) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 0].set_xscale('log')

ax[0, 1].hist(n_h_alpha_array[(class_array==2) & (age_array < 5)], linewidth=2, bins=bins, histtype='step')
ax[0, 1].hist(n_h_alpha_array[(class_array==2) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step')
ax[0, 1].hist(n_h_alpha_array[(class_array==2) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step')
ax[0, 1].hist(n_h_alpha_array[(class_array==2) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 1].hist(n_h_alpha_array[(class_array==2) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 1].set_xscale('log')


ax[0, 2].hist(n_h_alpha_array[(class_array==3) & (age_array < 5)], linewidth=2, bins=bins, histtype='step', label='age < 5 Myr')
ax[0, 2].hist(n_h_alpha_array[(class_array==3) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step')
ax[0, 2].hist(n_h_alpha_array[(class_array==3) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step')
ax[0, 2].hist(n_h_alpha_array[(class_array==3) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 2].hist(n_h_alpha_array[(class_array==3) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step')
ax[0, 2].set_xscale('log')


ax[1, 0].hist(n_h_alpha_array[(class_array==1) & (hii_array > 0) & (age_array < 5)], linewidth=2, bins=bins, histtype='step')
ax[1, 0].hist(n_h_alpha_array[(class_array==1) & (hii_array > 0) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step')
ax[1, 0].hist(n_h_alpha_array[(class_array==1) & (hii_array > 0) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step')
ax[1, 0].hist(n_h_alpha_array[(class_array==1) & (hii_array > 0) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step')
ax[1, 0].hist(n_h_alpha_array[(class_array==1) & (hii_array > 0) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step')
ax[1, 0].set_xscale('log')

ax[1, 1].hist(n_h_alpha_array[(class_array==2) & (hii_array > 0) & (age_array < 5)], linewidth=2, bins=bins, histtype='step')
ax[1, 1].hist(n_h_alpha_array[(class_array==2) & (hii_array > 0) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step')
ax[1, 1].hist(n_h_alpha_array[(class_array==2) & (hii_array > 0) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step')
ax[1, 1].hist(n_h_alpha_array[(class_array==2) & (hii_array > 0) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step')
ax[1, 1].hist(n_h_alpha_array[(class_array==2) & (hii_array > 0) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step')
ax[1, 1].set_xscale('log')


ax[1, 2].hist(n_h_alpha_array[(class_array==3) & (hii_array > 0) & (age_array < 5)], linewidth=2, bins=bins, histtype='step', label='age < 5 Myr')
ax[1, 2].hist(n_h_alpha_array[(class_array==3) & (hii_array > 0) & (age_array >= 5) & (age_array < 10)], linewidth=2, bins=bins, histtype='step', label='5 Myr <= age < 10 Myr')
ax[1, 2].hist(n_h_alpha_array[(class_array==3) & (hii_array > 0) & (age_array >= 10) & (age_array < 20)], linewidth=2, bins=bins, histtype='step', label='10 Myr <= age < 20 Myr')
ax[1, 2].hist(n_h_alpha_array[(class_array==3) & (hii_array > 0) & (age_array >= 20) & (age_array < 100)], linewidth=2, bins=bins, histtype='step', label='20 Myr <= age < 100 Myr')
ax[1, 2].hist(n_h_alpha_array[(class_array==3) & (hii_array > 0) & (age_array >= 100)], linewidth=2, bins=bins, histtype='step', label='age >= 100 Myr')
ax[1, 2].set_xscale('log')


ax[0, 0].set_title('Class 1 Hum', fontsize=fontsize)
ax[0, 1].set_title('Class 2 Hum', fontsize=fontsize)
ax[0, 2].set_title('Class 3 Hum', fontsize=fontsize)

ax[1, 2].legend(frameon=False, fontsize=fontsize - 4)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)

ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                         direction='in', labelsize=fontsize)

ax[1, 0].set_xlabel(r'Flux (H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', fontsize=fontsize)
ax[1, 1].set_xlabel(r'Flux (H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', fontsize=fontsize)
ax[1, 2].set_xlabel(r'Flux (H$\alpha$) 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', fontsize=fontsize)

ax[0, 0].set_ylabel(r'# Counts', fontsize=fontsize)
ax[1, 0].set_ylabel(r'# Counts', fontsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/hist_h_alpha_total_hum.png', bbox_inches='tight', dpi=300)
# fig.savefig('plot_output/hist_h_alpha_total.pdf', bbox_inches='tight', dpi=300)


exit()

fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(20, 39):

    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target=target)
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    cluster_id_ml_12 = catalog_access.get_hst_cc_phangs_id(target, classify='ml')
    cluster_id_ml_3 = catalog_access.get_hst_cc_phangs_id(target, classify='ml', cluster_class='class3')
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')

    class_1_ml = (cluster_class_ml_12 == 1) # & (cluster_class_qual_ml_12 >= 0.9)
    class_2_ml = (cluster_class_ml_12 == 2) # & (cluster_class_qual_ml_12 >= 0.9)
    class_3_ml = (cluster_class_ml_3 == 3) # & (cluster_class_qual_ml_3 >= 0.9)

    color_ub_ml = np.concatenate([color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = np.concatenate([color_vi_ml_12, color_vi_ml_3])

    h_alpha_intensity_hum_12 = np.zeros(len(color_vi_hum_12))
    if target == 'ngc0685':
        target_ext = 'ngc685'
    elif target == 'ngc0628c':
        target_ext = 'ngc628c'
    elif target == 'ngc0628e':
        target_ext = 'ngc628e'
    else:
        target_ext = target
    extra_file = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
    if os.path.isfile(extra_file):
        extra_tab_hdu = fits.open(extra_file)
        extra_data_table = extra_tab_hdu[1].data
        extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
        extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']

        for running_index, cluster_id in enumerate(cluster_id_hum_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_12[running_index] = extra_h_alpha_intensity[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    bins = np.logspace(-1, 3, 21)
    ax[row_index, col_index].hist(h_alpha_intensity_hum_12[(cluster_class_hum_12==2) & (age_hum_12 < 5)], linewidth=2, bins=bins, histtype='step')
    ax[row_index, col_index].hist(h_alpha_intensity_hum_12[(cluster_class_hum_12==2) & (age_hum_12 >= 5) & (age_hum_12 < 10)], linewidth=2, bins=bins, histtype='step')
    ax[row_index, col_index].hist(h_alpha_intensity_hum_12[(cluster_class_hum_12==2) & (age_hum_12 >= 10) & (age_hum_12 < 20)], linewidth=2, bins=bins, histtype='step')
    ax[row_index, col_index].set_xscale('log')


    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax[row_index, col_index].add_artist(anchored_left)

    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)





    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

    # fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, figsize=(16, 6))
    # fontsize = 18
    # ax[0].hist(h_alpha_intensity_hum_12[cluster_class_hum_12==1], linewidth=2, bins=bins, histtype='step')
    # ax[1].hist(h_alpha_intensity_hum_12[cluster_class_hum_12==2], linewidth=2, bins=bins, histtype='step')
    # ax[2].hist(h_alpha_intensity_hum_12[(cluster_class_hum_12==2) & (age_hum_12 < 10)], linewidth=2, bins=bins, histtype='step')
    # ax[2].hist(h_alpha_intensity_hum_12[(cluster_class_hum_12==2) & (age_hum_12 < 5)], linewidth=2, bins=bins, histtype='step')
    # ax[0].set_xscale('log')
    # plt.show()


ax[4, 3].scatter([], [], c=color_c1, s=30, label='Human class 1')
ax[4, 3].scatter([], [], c=color_c2, s=30, label='Human class 2')
ax[4, 3].plot([], [], color='k', label='ML')
ax[4, 3].legend(frameon=False, fontsize=fontsize)
ax[4, 3].axis('off')

fig.subplots_adjust(wspace=0, hspace=0)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/hist_h_alpha_panel_2_medsub.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/hist_h_alpha_panel_2_medsub.pdf', bbox_inches='tight', dpi=300)
