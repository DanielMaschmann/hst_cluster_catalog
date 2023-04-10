import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits

from xgaltool import analysis_tools, plotting_tools


from photometry_tools.plotting_tools import DensityContours


# get access to HST cluster catalog
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
data_mod = hdu_a[1].data
age_mod = data_mod['sfh.age']
flux_f555w = data_mod['F555W_UVIS_CHIP2']
flux_f814w = data_mod['F814W_UVIS_CHIP2']
flux_f336w = data_mod['F336W_UVIS_CHIP2']
flux_f438w = data_mod['F438W_UVIS_CHIP2']

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



# target_list = catalog_access.target_hst_cc
target_list = catalog_access.phangs_galaxy_list
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]


catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='human')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='human', cluster_class='class3')

catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')




fig = plt.figure(figsize=(24, 20))


x_lim_low = 8.1
x_lim_high = 11.9
y_lim_low = -1.9
y_lim_high = 1.4

ms_x_pos = 0.07
ms_y_pos = 0.07
ms_x_len = 0.9
ms_y_len = 0.9

contour_width = 0.12
contour_hight = 0.12

ax_ms = fig.add_axes([ms_x_pos, ms_y_pos, ms_x_len, ms_y_len])

ax_ms.set_xlim(x_lim_low, x_lim_high)
ax_ms.set_ylim(y_lim_low, y_lim_high)


# plot contours
gswlc_access = analysis_tools.AnalysisTools(object_type='gswlc_d_v1', writable_table=False)
log_sfr = gswlc_access.get_log_sed_sfr()
log_stellar_mass = gswlc_access.get_log_sed_stellar_mass()
good_values = (log_sfr > -2.5) & (log_sfr < 3.2) & (log_stellar_mass > 6.5) & (log_stellar_mass < 12.5)

hist, x_bin, y_bin = np.histogram2d(log_stellar_mass[good_values], log_sfr[good_values],
                                    bins=(np.linspace(6.5, 12.5), np.linspace(-2.5, 3.2)))
hist[hist < 10] = np.nan
ax_ms.imshow(hist.T, extent=(6.5, 12.5, -2.5, 3.2), origin='lower')



for index in range(0, 38):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    if target == 'ngc0628':
        color_ub_hum_12_e = catalog_access.get_hst_color_ub(target='ngc0628e')
        color_vi_hum_12_e = catalog_access.get_hst_color_vi(target='ngc0628e')
        color_ub_hum_3_e = catalog_access.get_hst_color_ub(target='ngc0628e', cluster_class='class3')
        color_vi_hum_3_e = catalog_access.get_hst_color_vi(target='ngc0628e', cluster_class='class3')
        color_ub_hum_12_c = catalog_access.get_hst_color_ub(target='ngc0628c')
        color_vi_hum_12_c = catalog_access.get_hst_color_vi(target='ngc0628c')
        color_ub_hum_3_c = catalog_access.get_hst_color_ub(target='ngc0628c', cluster_class='class3')
        color_vi_hum_3_c = catalog_access.get_hst_color_vi(target='ngc0628c', cluster_class='class3')
        color_ub = np.concatenate([color_ub_hum_12_e, color_ub_hum_3_e, color_ub_hum_12_c, color_ub_hum_3_c])
        color_vi = np.concatenate([color_vi_hum_12_e, color_vi_hum_3_e, color_vi_hum_12_c, color_vi_hum_3_c])
    else:
        color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
        color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
        color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
        color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
        color_ub = np.concatenate([color_ub_hum_12, color_ub_hum_3])
        color_vi = np.concatenate([color_vi_hum_12, color_vi_hum_3])

    sfr = catalog_access.get_target_sfr(target=target)
    sfr_err = catalog_access.get_target_sfr_err(target=target)
    mstar = catalog_access.get_target_mstar(target=target)
    mstar_err = catalog_access.get_target_mstar_err(target=target)

    ax_ms.scatter(np.log10(mstar), np.log10(sfr))

    ms_x_pos_cc = ms_x_pos + (ms_x_len / (x_lim_high - x_lim_low)) * (np.log10(mstar) - x_lim_low) - contour_width/2
    ms_y_pos_cc = ms_y_pos + (ms_y_len / (y_lim_high - y_lim_low)) * (np.log10(sfr) - y_lim_low) - contour_hight / 2
    print('ms_x_pos_cc ', ms_x_pos_cc)
    print('ms_y_pos_cc ', ms_y_pos_cc)

    ax1 = fig.add_axes([ms_x_pos_cc, ms_y_pos_cc, contour_width, contour_hight])
    ax1.patch.set_alpha(0.5)
    good_colors = (color_ub < 2) & (color_ub > -5.0) & (color_vi > -1.5) & (color_vi < 2.6)


    # hist_cont, x_, y_ = np.histogram2d(color_vi[good_colors], color_ub[good_colors],
    #                            bins=(np.linspace(-1.5, 2.6, 10), np.linspace(-2.5, 2, 10)))
    # import scipy.ndimage
    # hist_cont = scipy.ndimage.zoom(hist_cont, 10)
    # hist_cont = scipy.ndimage.gaussian_filter(hist_cont, 0.01)
    DensityContours.get_contours_percentage(ax=ax1, x_data=color_vi[good_colors],
                                            y_data=color_ub[good_colors],
                                            contour_levels=[0.5, 0.7, 0.8, 0.95, 0.99],
                                            color='black', percent=False,
                                            linewidth=2)
    # ax1.contour(hist_cont)
    # ax1.scatter(color_vi[good_colors], color_ub[good_colors])
    ax1.plot(model_vi, model_ub, color='red', linewidth=2)
    ax1.set_xlim(-1.0, 2.3)
    ax1.set_ylim(1.25, -2.5)
    ax1.axis('off')

# plt.show()
#
# exit()

plt.savefig('plot_output/test_ms.png')


exit()




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

