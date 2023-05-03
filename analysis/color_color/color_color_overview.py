import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde

def contours(ax, x, y, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    origin = 'lower'
    cs = ax.contour(xi, yi, zi, levels=levels,
                    linewidths=(2,),
                    origin=origin)

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)



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


color_vi_hum = np.array([])
color_ub_hum = np.array([])
clcl_color_hum = np.array([])

color_vi_ml = np.array([])
color_ub_ml = np.array([])
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])


mask_good_colors_hum = (color_vi_hum > -2) & (color_vi_hum < 3) & (color_ub_hum > -3) & (color_ub_hum < 2)
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_good_colors_ml = (color_vi_ml > -2) & (color_vi_ml < 3) & (color_ub_ml > -3) & (color_ub_ml < 2)
mask_class_1_ml = (clcl_color_ml == 1) #& (clcl_qual_color_ml >= 0.9)
mask_class_2_ml = (clcl_color_ml == 2) #& (clcl_qual_color_ml >= 0.9)
mask_class_3_ml = (clcl_color_ml == 3) #& (clcl_qual_color_ml >= 0.9)


fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(25, 13))
fontsize = 17



ax_shadow = ax[0, 0].twinx()
ax_shadow.plot(np.NaN, np.NaN, color='red', linewidth=3, linestyle='--', zorder=10, label='BC03, Z=Z$_{\odot}$')
ax_shadow.plot(np.NaN, np.NaN, color='gray', linewidth=3, linestyle='--', zorder=10, label='BC03, Z=Z$_{\odot}$/50')
ax_shadow.get_yaxis().set_visible(False)

ax_shadow.legend(loc=4, frameon=False, fontsize=fontsize - 2)


contours(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_1_hum * mask_good_colors_hum], legend=True, fontsize=fontsize - 4)
contours(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_2_hum * mask_good_colors_hum])
contours(ax=ax[0, 2], x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum],
                     y=color_ub_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum])
contours(ax=ax[0, 3], x=color_vi_hum[mask_class_3_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_3_hum * mask_good_colors_hum])

contours(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_1_ml * mask_good_colors_ml])
contours(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_2_ml * mask_good_colors_ml])
contours(ax=ax[1, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                     y=color_ub_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml])
contours(ax=ax[1, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_3_ml * mask_good_colors_ml])



ax[0, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)

ax[1, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)

ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 2].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 3].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)

ax[1, 0].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 1].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 2].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 3].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)

ax[0, 0].text(-1, 1.45, 'Class 1 Human', fontsize=fontsize)
ax[0, 1].text(-1, 1.45, 'Class 2 Human', fontsize=fontsize)
ax[0, 2].text(-1, 1.45, 'Class 1|2 Human', fontsize=fontsize)
ax[0, 3].text(-1, 1.45, 'Class 3 Human', fontsize=fontsize)

ax[0, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_hum * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_hum * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_hum * mask_good_colors_hum)), fontsize=fontsize)

ax[1, 0].text(-1, 1.45, 'Class 1 ML', fontsize=fontsize)
ax[1, 1].text(-1, 1.45, 'Class 2 ML', fontsize=fontsize)
ax[1, 2].text(-1, 1.45, 'Class 1|2 ML', fontsize=fontsize)
ax[1, 3].text(-1, 1.45, 'Class 3 ML', fontsize=fontsize)

ax[1, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)


ax[0, 0].set_ylim(1.9, -2.2)
ax[0, 0].set_xlim(-1.2, 2.4)

ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_overview.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_overview.pdf', bbox_inches='tight', dpi=300)
fig.clf()

