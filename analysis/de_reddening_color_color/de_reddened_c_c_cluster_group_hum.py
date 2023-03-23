import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
from photometry_tools.plotting_tools import DensityContours
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


def scale_reddening_vector(cluster_ebv, x_comp, y_comp):

    """This function scales the reddening vector for the given E(B-V) value of a star cluster"""
    comparison_scale = 1.0 / 3.2
    scale_factor = cluster_ebv / comparison_scale

    return x_comp * scale_factor, y_comp * scale_factor


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)

# get model
hdu_a = fits.open('../cigale_model/sfh2exp/out/models-block-0.fits')
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

cluster_class = np.array([])
color_ub = np.array([])
color_vi = np.array([])
ebv = np.array([])
age = np.array([])
stellar_m = np.array([])
ci = np.array([])
vi_mod = np.array([])
ub_mod = np.array([])

for target in target_list:

    print('target ', target)
    cluster_class_c12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_c12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_c12 = catalog_access.get_hst_color_vi(target=target)
    ebv_c12 = catalog_access.get_hst_cc_ebv(target=target)
    age_c12 = catalog_access.get_hst_cc_age(target=target)
    stellar_m_c12 = catalog_access.get_hst_cc_stellar_m(target=target)
    ci_c12 = catalog_access.get_hst_cc_ci(target=target)
    vi_mod_c12, ub_mod_c12 = scale_reddening_vector(ebv_c12, 0.43, 0.3)

    cluster_class_c3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_c3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_c3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    ebv_c3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    age_c3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    stellar_m_c3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    ci_c3 = catalog_access.get_hst_cc_ci(target=target, cluster_class='class3')
    vi_mod_c3, ub_mod_c3 = scale_reddening_vector(ebv_c3, 0.43, 0.3)

    cluster_class = np.concatenate([cluster_class, cluster_class_c12, cluster_class_c3])
    color_ub = np.concatenate([color_ub, color_ub_c12, color_ub_c3])
    color_vi = np.concatenate([color_vi, color_vi_c12, color_vi_c3])
    ebv = np.concatenate([ebv, ebv_c12, ebv_c3])
    age = np.concatenate([age, age_c12, age_c3])
    stellar_m = np.concatenate([stellar_m, stellar_m_c12, stellar_m_c3])
    ci = np.concatenate([ci, ci_c12, ci_c3])
    vi_mod = np.concatenate([vi_mod, vi_mod_c12, vi_mod_c3])
    ub_mod = np.concatenate([ub_mod, ub_mod_c12, ub_mod_c3])

print('cluster_class ', len(cluster_class))
print('color_ub ', len(color_ub))
print('color_vi ', len(color_vi))
print('ebv ', len(ebv))
print('vi_mod ', len(vi_mod))
print('ub_mod ', len(ub_mod))

print('cluster_class 1 ', sum(cluster_class == 1))
print('cluster_class 2 ', sum(cluster_class == 2))
print('cluster_class 3 ', sum(cluster_class == 3))
class_1 = cluster_class == 1
class_2 = cluster_class == 2
class_3 = cluster_class == 3

good_data = (color_vi > -1) & (color_vi < 2.5) & (color_ub > -2.3) & (color_ub < 1.5)

x_bins = np.linspace(-1.0, 2.3, 30)
y_bins = np.linspace(-2.2, 1.25, 30)

age_hist = np.zeros((29, 29)) * np.nan
ebv_hist = np.zeros((29, 29)) * np.nan
stellar_m_hist = np.zeros((29, 29)) * np.nan
ci_hist = np.zeros((29, 29)) * np.nan

age_hist_1 = np.zeros((29, 29)) * np.nan
ebv_hist_1 = np.zeros((29, 29)) * np.nan
stellar_m_hist_1 = np.zeros((29, 29)) * np.nan
ci_hist_1 = np.zeros((29, 29)) * np.nan

age_hist_2 = np.zeros((29, 29)) * np.nan
ebv_hist_2 = np.zeros((29, 29)) * np.nan
stellar_m_hist_2 = np.zeros((29, 29)) * np.nan
ci_hist_2 = np.zeros((29, 29)) * np.nan

age_hist_3 = np.zeros((29, 29)) * np.nan
ebv_hist_3 = np.zeros((29, 29)) * np.nan
stellar_m_hist_3 = np.zeros((29, 29)) * np.nan
ci_hist_3 = np.zeros((29, 29)) * np.nan

for x_index in range(len(x_bins) - 1):
    for y_index in range(len(y_bins) - 1):
        mask = (((color_vi - vi_mod)[good_data] > x_bins[x_index]) &
                ((color_vi - vi_mod)[good_data] < x_bins[x_index+1]) &
                ((color_ub - ub_mod)[good_data] > y_bins[y_index]) &
                ((color_ub - ub_mod)[good_data] < y_bins[y_index+1]))
        mask_1 = (((color_vi - vi_mod)[good_data * class_1] > x_bins[x_index]) &
                  ((color_vi - vi_mod)[good_data * class_1] < x_bins[x_index+1]) &
                  ((color_ub - ub_mod)[good_data * class_1] > y_bins[y_index]) &
                  ((color_ub - ub_mod)[good_data * class_1] < y_bins[y_index+1]))

        mask_2 = (((color_vi - vi_mod)[good_data * class_2] > x_bins[x_index]) &
                  ((color_vi - vi_mod)[good_data * class_2] < x_bins[x_index+1]) &
                  ((color_ub - ub_mod)[good_data * class_2] > y_bins[y_index]) &
                  ((color_ub - ub_mod)[good_data * class_2] < y_bins[y_index+1]))

        mask_3 = (((color_vi - vi_mod)[good_data * class_3] > x_bins[x_index]) &
                  ((color_vi - vi_mod)[good_data * class_3] < x_bins[x_index+1]) &
                  ((color_ub - ub_mod)[good_data * class_3] > y_bins[y_index]) &
                  ((color_ub - ub_mod)[good_data * class_3] < y_bins[y_index+1]))

        if sum(mask) < 15:
            continue

        age_hist[x_index, y_index] = np.nanmean(age[good_data][mask])
        ebv_hist[x_index, y_index] = np.nanmean(ebv[good_data][mask])
        stellar_m_hist[x_index, y_index] = np.nanmean(stellar_m[good_data][mask])
        ci_hist[x_index, y_index] = np.nanmean(ci[good_data][mask])

        if sum(mask_1) > 15:
            age_hist_1[x_index, y_index] = np.nanmean(age[good_data*class_1][mask_1])
            ebv_hist_1[x_index, y_index] = np.nanmean(ebv[good_data*class_1][mask_1])
            stellar_m_hist_1[x_index, y_index] = np.nanmean(stellar_m[good_data*class_1][mask_1])
            ci_hist_1[x_index, y_index] = np.nanmean(ci[good_data*class_1][mask_1])
        if sum(mask_2) > 15:
            age_hist_2[x_index, y_index] = np.nanmean(age[good_data*class_2][mask_2])
            ebv_hist_2[x_index, y_index] = np.nanmean(ebv[good_data*class_2][mask_2])
            stellar_m_hist_2[x_index, y_index] = np.nanmean(stellar_m[good_data*class_2][mask_2])
            ci_hist_2[x_index, y_index] = np.nanmean(ci[good_data*class_2][mask_2])
        if sum(mask_3) > 15:
            age_hist_3[x_index, y_index] = np.nanmean(age[good_data*class_3][mask_3])
            ebv_hist_3[x_index, y_index] = np.nanmean(ebv[good_data*class_3][mask_3])
            stellar_m_hist_3[x_index, y_index] = np.nanmean(stellar_m[good_data*class_3][mask_3])
            ci_hist_3[x_index, y_index] = np.nanmean(ci[good_data*class_3][mask_3])

figure = plt.figure(figsize=(17, 28))
fontsize = 15

ax_explain = figure.add_axes([0.07, 0.80, 0.29, 0.15])
ax_all = figure.add_axes([0.38, 0.80, 0.29, 0.15])
ax_de_red = figure.add_axes([0.69, 0.80, 0.29, 0.15])

ax_c1 = figure.add_axes([0.07, 0.6, 0.29, 0.15])
ax_c2 = figure.add_axes([0.38, 0.6, 0.29, 0.15])
ax_c3 = figure.add_axes([0.69, 0.6, 0.29, 0.15])

ax_age_cbar = figure.add_axes([0.94, 0.4, 0.015, 0.11])
ax_ebv_cbar = figure.add_axes([0.94, 0.24, 0.015, 0.11])
ax_mass_cbar = figure.add_axes([0.94, 0.07, 0.015, 0.11])

ax_age_1 = figure.add_axes([0.06, 0.38, 0.29, 0.15])
ax_age_2 = figure.add_axes([0.37, 0.38, 0.29, 0.15])
ax_age_3 = figure.add_axes([0.68, 0.38, 0.29, 0.15])

ax_ebv_1 = figure.add_axes([0.06, 0.22, 0.29, 0.15])
ax_ebv_2 = figure.add_axes([0.37, 0.22, 0.29, 0.15])
ax_ebv_3 = figure.add_axes([0.68, 0.22, 0.29, 0.15])

ax_mass_1 = figure.add_axes([0.06, 0.05, 0.29, 0.15])
ax_mass_2 = figure.add_axes([0.37, 0.05, 0.29, 0.15])
ax_mass_3 = figure.add_axes([0.68, 0.05, 0.29, 0.15])

norm_age = Normalize(6, 10)
cmap_age = 'inferno'
norm_ebv = Normalize(0, 0.8)
cmap_ebv = 'cividis'
norm_mass = Normalize(4, 6)
cmap_mass = 'viridis'


ax_explain.plot(model_vi, model_ub, color='red', linewidth=3)
x_de_red, y_dered = scale_reddening_vector(0.5, 0.43, 0.3)
ax_explain.annotate('', xy=(1.7 - x_de_red, -1.5 - y_dered), xycoords='data', xytext=(1.7, -1.5), textcoords='data',
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=3, ls='-'))
ax_explain.text(1.1, -1.6, 'E(B-V) = 0.5', fontsize=fontsize, rotation=-29)


vi_1_my = model_vi[age_mod == 1][0]
ub_1_my = model_ub[age_mod == 1][0]

vi_4_my = model_vi[age_mod == 4][0]
ub_4_my = model_ub[age_mod == 4][0]

vi_5_my = model_vi[age_mod == 5][0]
ub_5_my = model_ub[age_mod == 5][0]

vi_10_my = model_vi[age_mod == 10][0]
ub_10_my = model_ub[age_mod == 10][0]

vi_100_my = model_vi[age_mod == 102][0]
ub_100_my = model_ub[age_mod == 102][0]

vi_1000_my = model_vi[age_mod == 1028][0]
ub_1000_my = model_ub[age_mod == 1028][0]

vi_10000_my = model_vi[age_mod == 10308][0]
ub_10000_my = model_ub[age_mod == 10308][0]

ax_explain.scatter(vi_1_my, ub_1_my, c='k', zorder=10)
ax_explain.scatter(vi_4_my, ub_4_my, c='k', zorder=10)
ax_explain.scatter(vi_5_my, ub_5_my, c='k', zorder=10)
ax_explain.scatter(vi_10_my, ub_10_my, c='k', zorder=10)
ax_explain.scatter(vi_100_my, ub_100_my, c='k', zorder=10)
ax_explain.scatter(vi_1000_my, ub_1000_my, c='k', zorder=10)
ax_explain.scatter(vi_10000_my, ub_10000_my, c='k', zorder=10)

ax_explain.annotate('1 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my + 0.5, ub_1_my),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my + 0.5, ub_10_my),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my, ub_1000_my - 0.5),
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my - 0.5, ub_10000_my + 0.5), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.set_ylim(1.25, -2.2)
ax_explain.set_xlim(-1.0, 1.8)
ax_explain.set_xticklabels([])
ax_explain.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_explain.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
# ax_ebv_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax_all.plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_all, x_data=color_vi[good_data], y_data=color_ub[good_data], color='black', percent=False)
ax_all.set_title('All Clusters', fontsize=fontsize)
ax_all.set_ylim(1.25, -2.2)
ax_all.set_xlim(-1.0, 1.8)
ax_all.set_xticklabels([])
ax_all.set_yticklabels([])
ax_all.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax_de_red.plot(model_vi, model_ub, color='red', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_de_red,
                                        x_data=(color_vi - vi_mod)[good_data], y_data=(color_ub - ub_mod)[good_data],
                                        color='black', percent=False)
ax_de_red.set_title('All clusters de-reddened', fontsize=fontsize)
ax_de_red.set_ylim(1.25, -2.2)
ax_de_red.set_xlim(-1.0, 1.8)
ax_de_red.set_xticklabels([])
ax_de_red.set_yticklabels([])
ax_de_red.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax_c1.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c1,
                                        x_data=(color_vi - vi_mod)[good_data * class_1],
                                        y_data=(color_ub - ub_mod)[good_data * class_1],
                                        color='forestgreen', percent=False, )
ax_c1.set_title('Class 1 de-reddened', fontsize=fontsize)
ax_c1.set_ylim(1.25, -2.2)
ax_c1.set_xlim(-1.0, 1.8)
# ax_c1.set_xticklabels([])
# ax_c1.set_yticklabels([])
ax_c1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_c1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_c1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_c2.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c2,
                                        x_data=(color_vi - vi_mod)[good_data * class_2],
                                        y_data=(color_ub - ub_mod)[good_data * class_2],
                                        color='darkorange', percent=False, )
ax_c2.set_title('Class 2 de-reddened', fontsize=fontsize)
ax_c2.set_ylim(1.25, -2.2)
ax_c2.set_xlim(-1.0, 1.8)
# ax_c2.set_xticklabels([])
ax_c2.set_yticklabels([])
ax_c2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_c2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_c3.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c3,
                                        x_data=(color_vi - vi_mod)[good_data * class_3],
                                        y_data=(color_ub - ub_mod)[good_data * class_3],
                                        color='royalblue', percent=False, )
ax_c3.set_title('Class 3 de-reddened', fontsize=fontsize)
ax_c3.set_ylim(1.25, -2.2)
ax_c3.set_xlim(-1.0, 1.8)
# ax_c3.set_xticklabels([])
ax_c3.set_yticklabels([])
ax_c3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_c3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)



ax_age_1.plot(model_vi, model_ub, color='red', linewidth=2)
ax_age_1.imshow(np.log10(age_hist_1.T) + 6, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_age, norm=norm_age)
ax_age_1.set_ylim(1.25, -2.2)
ax_age_1.set_xlim(-1.0, 1.8)
ax_age_1.set_xticklabels([])
# ax_age_1.set_yticklabels([])
ax_age_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_1.set_title('Class 1', fontsize=fontsize)
ax_age_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax_age_2.plot(model_vi, model_ub, color='red', linewidth=2)
ax_age_2.imshow(np.log10(age_hist_2.T) + 6, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_age, norm=norm_age)
ax_age_2.set_ylim(1.25, -2.2)
ax_age_2.set_xlim(-1.0, 1.8)
ax_age_2.set_xticklabels([])
ax_age_2.set_yticklabels([])
ax_age_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_2.set_title('Class 2', fontsize=fontsize)

ax_age_3.plot(model_vi, model_ub, color='red', linewidth=2)
ax_age_3.imshow(np.log10(age_hist_3.T) + 6, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_age, norm=norm_age)
ax_age_3.set_ylim(1.25, -2.2)
ax_age_3.set_xlim(-1.0, 1.8)
ax_age_3.set_xticklabels([])
ax_age_3.set_yticklabels([])
ax_age_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_3.set_title('Class 3', fontsize=fontsize)

ColorbarBase(ax_age_cbar, orientation='vertical', cmap=cmap_age, norm=norm_age)
ax_age_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)
ax_age_cbar.set_ylabel('log(Age/Yr)', labelpad=2, fontsize=fontsize)
ax_age_cbar.yaxis.set_label_position('right')


ax_ebv_1.plot(model_vi, model_ub, color='red', linewidth=2)
ax_ebv_1.imshow(ebv_hist_1.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_1.set_ylim(1.25, -2.2)
ax_ebv_1.set_xlim(-1.0, 1.8)
ax_ebv_1.set_xticklabels([])
# ax_ebv_1.set_yticklabels([])
ax_ebv_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
# ax_ebv_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax_ebv_2.plot(model_vi, model_ub, color='red', linewidth=2)
ax_ebv_2.imshow(ebv_hist_2.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_2.set_ylim(1.25, -2.2)
ax_ebv_2.set_xlim(-1.0, 1.8)
ax_ebv_2.set_xticklabels([])
ax_ebv_2.set_yticklabels([])
ax_ebv_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_ebv_3.plot(model_vi, model_ub, color='red', linewidth=2)
ax_ebv_3.imshow(ebv_hist_3.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_3.set_ylim(1.25, -2.2)
ax_ebv_3.set_xlim(-1.0, 1.8)
ax_ebv_3.set_xticklabels([])
ax_ebv_3.set_yticklabels([])
ax_ebv_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ColorbarBase(ax_ebv_cbar, orientation='vertical', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)
ax_ebv_cbar.set_ylabel('E(B-V)', labelpad=2, fontsize=fontsize)
ax_ebv_cbar.yaxis.set_label_position('right')


ax_mass_1.plot(model_vi, model_ub, color='red', linewidth=2)
ax_mass_1.imshow(np.log10(stellar_m_hist_1).T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_mass, norm=norm_mass)
ax_mass_1.set_ylim(1.25, -2.2)
ax_mass_1.set_xlim(-1.0, 1.8)
# ax_mass_1.set_xticklabels([])
# ax_mass_1.set_yticklabels([])
ax_mass_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_mass_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_mass_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax_mass_2.plot(model_vi, model_ub, color='red', linewidth=2)
ax_mass_2.imshow(np.log10(stellar_m_hist_2).T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_mass, norm=norm_mass)
ax_mass_2.set_ylim(1.25, -2.2)
ax_mass_2.set_xlim(-1.0, 1.8)
ax_mass_2.set_xticklabels([])
# ax_mass_2.set_yticklabels([])
ax_mass_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
# ax_mass_2.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_mass_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_mass_3.plot(model_vi, model_ub, color='red', linewidth=2)
ax_mass_3.imshow(np.log10(stellar_m_hist_3).T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_mass, norm=norm_mass)
ax_mass_3.set_ylim(1.25, -2.2)
ax_mass_3.set_xlim(-1.0, 1.8)
ax_mass_3.set_xticklabels([])
# ax_mass_3.set_yticklabels([])
ax_mass_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
# ax_mass_3.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_mass_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ColorbarBase(ax_mass_cbar, orientation='vertical', cmap=cmap_mass, norm=norm_mass)
ax_mass_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                         labeltop=True, labelsize=fontsize)
ax_mass_cbar.set_ylabel(r'log(M$_{*}$/M$_{\odot}$)', labelpad=2, fontsize=fontsize)
ax_mass_cbar.yaxis.set_label_position('right')


plt.savefig('plot_output/de_reddening_group_hum.png')
