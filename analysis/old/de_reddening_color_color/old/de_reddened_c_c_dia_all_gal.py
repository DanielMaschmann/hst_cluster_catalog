import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
from photometry_tools.plotting_tools import DensityContours
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from matplotlib.colors import Normalize, LogNorm
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
hdu_a = fits.open('out/models-block-0.fits')
data = hdu_a[1].data
age_mod = data['stellar.age']
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


catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class12')
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
    color_ub_c12 = catalog_access.get_hst_color_ub(target=target, cluster_class='class12')
    color_vi_c12 = catalog_access.get_hst_color_vi(target=target, cluster_class='class12')
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

good_data = (color_vi > -1) & (color_vi < 2.5) & (color_ub > -2.3) & (color_ub < 1.5) & (age > 0.4) & (age < 14000)

x_bins = np.linspace(-1.0, 2.3, 30)
y_bins = np.linspace(-2.2, 1.25, 30)

age_hist = np.zeros((29, 29)) * np.nan
ebv_hist = np.zeros((29, 29)) * np.nan
stellar_m_hist = np.zeros((29, 29)) * np.nan
ci_hist = np.zeros((29, 29)) * np.nan


for x_index in range(len(x_bins) - 1):
    for y_index in range(len(y_bins) - 1):
        mask = (((color_vi - vi_mod)[good_data] > x_bins[x_index]) &
                ((color_vi - vi_mod)[good_data] < x_bins[x_index+1]) &
                ((color_ub - ub_mod)[good_data] > y_bins[y_index]) &
                ((color_ub - ub_mod)[good_data] < y_bins[y_index+1]))

        if sum(mask) < 15:
            continue

        age_hist[x_index, y_index] = np.nanmean(age[good_data][mask])
        ebv_hist[x_index, y_index] = np.nanmean(ebv[good_data][mask])
        stellar_m_hist[x_index, y_index] = np.nanmean(stellar_m[good_data][mask])
        ci_hist[x_index, y_index] = np.nanmean(ci[good_data][mask])


figure = plt.figure(figsize=(17, 17))
fontsize = 15

ax_explain = figure.add_axes([0.07, 0.69, 0.29, 0.28])
ax_all = figure.add_axes([0.38, 0.69, 0.29, 0.28])
ax_de_red = figure.add_axes([0.69, 0.69, 0.29, 0.28])

ax_age = figure.add_axes([0.07, 0.355, 0.29, 0.28])
ax_ebv = figure.add_axes([0.38, 0.355, 0.29, 0.28])
ax_mass = figure.add_axes([0.69, 0.355, 0.29, 0.28])

ax_age_cbar = figure.add_axes([0.115, 0.64, 0.2, 0.015])
ax_ebv_cbar = figure.add_axes([0.425, 0.64, 0.2, 0.015])
ax_mass_cbar = figure.add_axes([0.735, 0.64, 0.2, 0.015])


ax_c1 = figure.add_axes([0.07, 0.05, 0.29, 0.28])
ax_c2 = figure.add_axes([0.38, 0.05, 0.29, 0.28])
ax_c3 = figure.add_axes([0.69, 0.05, 0.29, 0.28])


ax_explain.plot(model_vi, model_ub, color='red', linewidth=3)
x_de_red, y_dered = scale_reddening_vector(0.5, 0.43, 0.3)
ax_explain.annotate('', xy=(1.7 - x_de_red, -1.5 - y_dered),
             xycoords='data',
             xytext=(1.7, -1.5),
             textcoords='data',
             arrowprops=dict(arrowstyle='<|-',
                             color='k',
                             lw=3,
                             ls='-'))
ax_explain.text(1.1, -1.6, 'E(B-V) = 0.5', fontsize=fontsize, rotation=-29)


vi_1_my = model_vi[(age_mod > 0.9) & (age_mod < 1.1)][0]
ub_1_my = model_ub[(age_mod > 0.9) & (age_mod < 1.1)][0]

vi_3_my = model_vi[(age_mod > 2.9) & (age_mod < 3.1)][0]
ub_3_my = model_ub[(age_mod > 2.9) & (age_mod < 3.1)][0]

vi_5_my = model_vi[(age_mod > 4.9) & (age_mod < 5.1)][0]
ub_5_my = model_ub[(age_mod > 4.9) & (age_mod < 5.1)][0]

vi_10_my = model_vi[(age_mod > 9.9) & (age_mod < 10.1)][0]
ub_10_my = model_ub[(age_mod > 9.9) & (age_mod < 10.1)][0]

vi_100_my = model_vi[(age_mod > 90) & (age_mod < 110)][0]
ub_100_my = model_ub[(age_mod > 90) & (age_mod < 110)][0]

vi_1000_my = model_vi[(age_mod > 990) & (age_mod < 1100)][0]
ub_1000_my = model_ub[(age_mod > 990) & (age_mod < 1100)][0]

vi_10000_my = model_vi[(age_mod > 9900) & (age_mod < 11000)][0]
ub_10000_my = model_ub[(age_mod > 9900) & (age_mod < 11000)][0]

ax_explain.scatter(vi_1_my, ub_1_my, c='k', zorder=10)
ax_explain.scatter(vi_3_my, ub_3_my, c='k', zorder=10)
ax_explain.scatter(vi_5_my, ub_5_my, c='k', zorder=10)
ax_explain.scatter(vi_10_my, ub_10_my, c='k', zorder=10)
ax_explain.scatter(vi_100_my, ub_100_my, c='k', zorder=10)
ax_explain.scatter(vi_1000_my, ub_1000_my, c='k', zorder=10)
ax_explain.scatter(vi_10000_my, ub_10000_my, c='k', zorder=10)

ax_explain.annotate('1 Myr', xy=(vi_1_my, ub_1_my),
             xycoords='data',
             xytext=(vi_1_my + 0.5, ub_1_my),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('3 Myr', xy=(vi_3_my, ub_3_my),
             xycoords='data',
             xytext=(vi_3_my - 0.5, ub_3_my),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('5 Myr', xy=(vi_5_my, ub_5_my),
             xycoords='data',
             xytext=(vi_5_my - 0.5, ub_5_my + 0.5),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('10 Myr', xy=(vi_10_my, ub_10_my),
             xycoords='data',
             xytext=(vi_10_my + 0.5, ub_10_my),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('100 Myr', xy=(vi_100_my, ub_100_my),
             xycoords='data',
             xytext=(vi_100_my - 0.5, ub_100_my + 0.5),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my),
             xycoords='data',
             xytext=(vi_1000_my, ub_1000_my - 0.5),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))
ax_explain.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my),
             xycoords='data',
             xytext=(vi_10000_my - 0.5, ub_10000_my + 0.5),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>',
                             color='k',
                             lw=2,
                             ls='-'))

ax_explain.set_ylim(1.25, -2.2)
ax_explain.set_xlim(-1.0, 1.8)
ax_explain.set_xticklabels([])
ax_explain.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



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






norm_age = Normalize(6, 10)
cmap_age = 'plasma'

ax_age.plot(model_vi, model_ub, color='red', linewidth=2)
ax_age.imshow(np.log10(age_hist.T) + 6, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_age, norm=norm_age)
ax_age.set_ylim(1.25, -2.2)
ax_age.set_xlim(-1.0, 1.8)
ax_age.set_xticklabels([])
# ax_age.set_yticklabels([])
ax_age.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ColorbarBase(ax_age_cbar, orientation='horizontal', cmap=cmap_age, norm=norm_age)
ax_age_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)
ax_age_cbar.set_xlabel('log(Age/Yr)', labelpad=2, fontsize=fontsize)
ax_age_cbar.xaxis.set_label_position('top')


norm_ebv = Normalize(0, 0.8)
cmap_ebv = 'plasma'

ax_ebv.plot(model_vi, model_ub, color='red', linewidth=2)
ax_ebv.imshow(ebv_hist.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_ebv, norm=norm_ebv)
ax_ebv.set_ylim(1.25, -2.2)
ax_ebv.set_xlim(-1.0, 1.8)
ax_ebv.set_xticklabels([])
ax_ebv.set_yticklabels([])
ax_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ColorbarBase(ax_ebv_cbar, orientation='horizontal', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)
ax_ebv_cbar.set_xlabel('E(B-V)', labelpad=2, fontsize=fontsize)
ax_ebv_cbar.xaxis.set_label_position('top')


norm_mass = Normalize(4, 6)
cmap_mass = 'plasma'
ax_mass.plot(model_vi, model_ub, color='red', linewidth=2)
ax_mass.imshow(np.log10(stellar_m_hist).T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_mass, norm=norm_mass)
ax_mass.set_ylim(1.25, -2.2)
ax_mass.set_xlim(-1.0, 1.8)
ax_mass.set_xticklabels([])
ax_mass.set_yticklabels([])
ax_mass.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ColorbarBase(ax_mass_cbar, orientation='horizontal', cmap=cmap_mass, norm=norm_mass)
ax_mass_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                         labeltop=True, labelsize=fontsize)
ax_mass_cbar.set_xlabel(r'log(M$_{*}$/M$_{\odot}$)', labelpad=2, fontsize=fontsize)
ax_mass_cbar.xaxis.set_label_position('top')


ax_c1.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c1,
                                        x_data=(color_vi - vi_mod)[good_data * class_1],
                                        y_data=(color_ub - ub_mod)[good_data * class_1],
                                        color='darkslategray', percent=False, )
ax_c1.set_title('Class 1 de-reddened', fontsize=fontsize)
ax_c1.set_ylim(1.25, -2.2)
ax_c1.set_xlim(-1.0, 1.8)
#ax_c1.set_xticklabels([])
#ax_c1.set_yticklabels([])
ax_c1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



ax_c2.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c2,
                                        x_data=(color_vi - vi_mod)[good_data * class_2],
                                        y_data=(color_ub - ub_mod)[good_data * class_2],
                                        color='darkorange', percent=False, )
ax_c2.set_title('Class 2 de-reddened', fontsize=fontsize)
ax_c2.set_ylim(1.25, -2.2)
ax_c2.set_xlim(-1.0, 1.8)
#ax_c2.set_xticklabels([])
ax_c2.set_yticklabels([])
ax_c2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


ax_c3.plot(model_vi, model_ub, color='r', linewidth=2)
DensityContours.get_contours_percentage(ax=ax_c3,
                                        x_data=(color_vi - vi_mod)[good_data * class_3],
                                        y_data=(color_ub - ub_mod)[good_data * class_3],
                                        color='royalblue', percent=False, )
ax_c3.set_title('Class 3 de-reddened', fontsize=fontsize)
ax_c3.set_ylim(1.25, -2.2)
ax_c3.set_xlim(-1.0, 1.8)
#ax_c3.set_xticklabels([])
ax_c3.set_yticklabels([])
ax_c3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



plt.savefig('plot_output/test.png')

exit()


ax[0, 2].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
DensityContours.get_contours_percentage(ax=ax[0, 2],
                                        x_data=(color_vi - vi_mod)[good_data * class_1],
                                        y_data=(color_ub - ub_mod)[good_data * class_1],
                                        color='darkslategray', percent=False, )
ax[0, 2].set_title('Class 1 de-reddened', fontsize=fontsize)

axax.plot(model_vi, model_ub, color='salmon', linewidth=1.2)
DensityContours.get_contours_percentage(ax=ax[0, 3],
                                        x_data=(color_vi - vi_mod)[good_data * class_2],
                                        y_data=(color_ub - ub_mod)[good_data * class_2],
                                        color='darkorange', percent=False, )
ax[0, 3].set_title('Class 2 de-reddened', fontsize=fontsize)

ax[0, 4].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
DensityContours.get_contours_percentage(ax=ax[0, 4],
                                        x_data=(color_vi - vi_mod)[good_data * class_3],
                                        y_data=(color_ub - ub_mod)[good_data * class_3],
                                        color='royalblue', percent=False, )
ax[0, 4].set_title('Class 3 de-reddened', fontsize=fontsize)







hist, xedges, yedges = np.histogram2d(x=(color_vi - vi_mod)[good_data], y=(color_ub - ub_mod)[good_data],
                      bins=[np.linspace(-1.0, 2.3, 30), np.linspace(-2.2, 1.25, 30)], )

ax[1, 0].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
ax[1, 0].imshow(hist.T, extent=(-1.0, 2.3, 1.25, -2.2))

ax[1, 1].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
ax[1, 1].imshow(np.log10(age_hist).T + 6, extent=(-1.0, 2.3, 1.25, -2.2), vmin=6, vmax=10)


ax[1, 2].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
ax[1, 2].imshow(ebv_hist.T, extent=(-1.0, 2.3, 1.25, -2.2), vmin=0, vmax=0.8)

ax[1, 3].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
ax[1, 3].imshow(np.log10(stellar_m_hist).T, extent=(-1.0, 2.3, 1.25, -2.2), vmin=4, vmax=6)

ax[1, 4].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
ax[1, 4].imshow(ci_hist.T, extent=(-1.0, 2.3, 1.25, -2.2), vmin=1, vmax=2.5)








ax[0, 0].set_ylim(1.25, -2.2)
ax[0, 0].set_xlim(-1.0, 2.3)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[0, 4].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)

ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)
ax[1, 4].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                     direction='in', labelsize=fontsize)


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/ub_vi_dereddening_all_gal.png', bbox_inches='tight', dpi=300)
# plt.show()


#
#
#
# ax[0].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
# ax[1].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
# ax[2].plot(model_vi, model_ub, color='salmon', linewidth=1.2)
#
# for index in range(int(len(color_vi)*0.2)):
#     if ebv[index] == 0:
#         continue
#     ax[0].scatter(color_vi[index], color_ub[index], c='r', s=1)
#     ax[0].scatter((color_vi - vi_mod)[index], (color_ub - ub_mod)[index], c='gray', s=1)
#     ax[0].plot([color_vi[index], (color_vi - vi_mod)[index]], [color_ub[index], (color_ub - ub_mod)[index]],
#                color='k', linestyle='--')
#
# ax[1].scatter(color_vi, color_ub, c='r', s=1)
# ax[2].scatter((color_vi - vi_mod), (color_ub - ub_mod), c='gray', s=1)
#
# ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
#                                      direction='in', labelsize=fontsize)
# ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
#                                      direction='in', labelsize=fontsize)
# ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
#                                      direction='in', labelsize=fontsize)
#
# ax[0].set_title(target.upper() + '  Random 20% BCW Class 1/2/3', fontsize=fontsize)
# ax[1].set_title('All BCW Class 1/2/3', fontsize=fontsize)
# ax[2].set_title('De-reddened', fontsize=fontsize)
#
# ax[0].set_ylim(1.25, -2.2)
# ax[0].set_xlim(-1.0, 2.3)
#
# ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
# ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
#
# # fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
# # fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
#
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('plot_output/individual_gal/ub_vi_dereddening_%s.png' % target, bbox_inches='tight', dpi=300)
# plt.close()

