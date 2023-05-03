import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


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
chi2_hum = np.array([])
v_band_sn_hum = np.array([])

color_vi_ml = np.array([])
color_ub_ml = np.array([])
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])
chi2_ml = np.array([])
v_band_sn_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    chi2_hum_12 = catalog_access.get_hst_cc_min_chi2(target=target)
    v_band_sn_hum_12 = (catalog_access.get_hst_cc_band_flux(target=target, band='F555W') /
                        catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W'))
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    chi2_hum_3 = catalog_access.get_hst_cc_min_chi2(target=target, cluster_class='class3')
    v_band_sn_hum_3 = (catalog_access.get_hst_cc_band_flux(target=target, band='F555W', cluster_class='class3') /
                       catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W', cluster_class='class3'))
    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])
    chi2_hum = np.concatenate([chi2_hum, chi2_hum_12, chi2_hum_3])
    v_band_sn_hum = np.concatenate([v_band_sn_hum, v_band_sn_hum_12, v_band_sn_hum_3])

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    chi2_ml_12 = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml')
    v_band_sn_ml_12 = (catalog_access.get_hst_cc_band_flux(target=target, band='F555W', classify='ml') /
                        catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W', classify='ml'))
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    chi2_ml_3 = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml', cluster_class='class3')
    v_band_sn_ml_3 = (catalog_access.get_hst_cc_band_flux(target=target, band='F555W', classify='ml', cluster_class='class3') /
                      catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W', classify='ml', cluster_class='class3'))
    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])
    chi2_ml = np.concatenate([chi2_ml, chi2_ml_12, chi2_ml_3])
    v_band_sn_ml = np.concatenate([v_band_sn_ml, v_band_sn_ml_12, v_band_sn_ml_3])


mask_good_colors_hum = (color_vi_hum > -2) & (color_vi_hum < 3) & (color_ub_hum > -3) & (color_ub_hum < 2)
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_good_colors_ml = (color_vi_ml > -2) & (color_vi_ml < 3) & (color_ub_ml > -3) & (color_ub_ml < 2)
mask_class_1_ml = (clcl_color_ml == 1) # & (clcl_qual_color_ml >= 0.9)
mask_class_2_ml = (clcl_color_ml == 2) # & (clcl_qual_color_ml >= 0.9)
mask_class_3_ml = (clcl_color_ml == 3) # & (clcl_qual_color_ml >= 0.9)



good_data_hum = (color_vi_hum > -1) & (color_vi_hum < 2.5) & (color_ub_hum > -2.3) & (color_ub_hum < 1.5)
good_data_ml = ((color_vi_ml > -1) & (color_vi_ml < 2.5) & (color_ub_ml > -2.3) & (color_ub_ml < 1.5)
                #& (clcl_qual_color_ml >= 0.9)
                )

x_bins = np.linspace(-1.0, 2.3, 30)
y_bins = np.linspace(-2.2, 1.25, 30)
v_band_sn_hist_cl1_hum = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl2_hum = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl12_hum = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl3_hum = np.zeros((29, 29)) * np.nan


v_band_sn_hist_cl1_ml = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl2_ml = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl12_ml = np.zeros((29, 29)) * np.nan
v_band_sn_hist_cl3_ml = np.zeros((29, 29)) * np.nan

threshold_hum = 5
threshold_ml = 5

for x_index in range(len(x_bins) - 1):
    for y_index in range(len(y_bins) - 1):

        mask_hum = ((color_vi_hum[good_data_hum] > x_bins[x_index]) &
                (color_vi_hum[good_data_hum] < x_bins[x_index+1]) &
                (color_ub_hum[good_data_hum] > y_bins[y_index]) &
                (color_ub_hum[good_data_hum] < y_bins[y_index+1]))
        mask_1_hum = ((color_vi_hum[good_data_hum * mask_class_1_hum] > x_bins[x_index]) &
                  (color_vi_hum[good_data_hum * mask_class_1_hum] < x_bins[x_index+1]) &
                  (color_ub_hum[good_data_hum * mask_class_1_hum] > y_bins[y_index]) &
                  (color_ub_hum[good_data_hum * mask_class_1_hum] < y_bins[y_index+1]))
        mask_2_hum = ((color_vi_hum[good_data_hum * mask_class_2_hum] > x_bins[x_index]) &
                  (color_vi_hum[good_data_hum * mask_class_2_hum] < x_bins[x_index+1]) &
                  (color_ub_hum[good_data_hum * mask_class_2_hum] > y_bins[y_index]) &
                  (color_ub_hum[good_data_hum * mask_class_2_hum] < y_bins[y_index+1]))
        mask_12_hum = ((color_vi_hum[good_data_hum * (mask_class_2_hum + mask_class_1_hum)] > x_bins[x_index]) &
                   (color_vi_hum[good_data_hum * (mask_class_2_hum + mask_class_1_hum)] < x_bins[x_index+1]) &
                   (color_ub_hum[good_data_hum * (mask_class_2_hum + mask_class_1_hum)] > y_bins[y_index]) &
                   (color_ub_hum[good_data_hum * (mask_class_2_hum + mask_class_1_hum)] < y_bins[y_index+1]))
        mask_3_hum = ((color_vi_hum[good_data_hum * mask_class_3_hum] > x_bins[x_index]) &
                  (color_vi_hum[good_data_hum * mask_class_3_hum] < x_bins[x_index+1]) &
                  (color_ub_hum[good_data_hum * mask_class_3_hum] > y_bins[y_index]) &
                  (color_ub_hum[good_data_hum * mask_class_3_hum] < y_bins[y_index+1]))

        mask_ml = ((color_vi_ml[good_data_ml] > x_bins[x_index]) &
                (color_vi_ml[good_data_ml] < x_bins[x_index+1]) &
                (color_ub_ml[good_data_ml] > y_bins[y_index]) &
                (color_ub_ml[good_data_ml] < y_bins[y_index+1]))
        mask_1_ml = ((color_vi_ml[good_data_ml * mask_class_1_ml] > x_bins[x_index]) &
                  (color_vi_ml[good_data_ml * mask_class_1_ml] < x_bins[x_index+1]) &
                  (color_ub_ml[good_data_ml * mask_class_1_ml] > y_bins[y_index]) &
                  (color_ub_ml[good_data_ml * mask_class_1_ml] < y_bins[y_index+1]))
        mask_2_ml = ((color_vi_ml[good_data_ml * mask_class_2_ml] > x_bins[x_index]) &
                  (color_vi_ml[good_data_ml * mask_class_2_ml] < x_bins[x_index+1]) &
                  (color_ub_ml[good_data_ml * mask_class_2_ml] > y_bins[y_index]) &
                  (color_ub_ml[good_data_ml * mask_class_2_ml] < y_bins[y_index+1]))
        mask_12_ml = ((color_vi_ml[good_data_ml * (mask_class_2_ml + mask_class_1_ml)] > x_bins[x_index]) &
                   (color_vi_ml[good_data_ml * (mask_class_2_ml + mask_class_1_ml)] < x_bins[x_index+1]) &
                   (color_ub_ml[good_data_ml * (mask_class_2_ml + mask_class_1_ml)] > y_bins[y_index]) &
                   (color_ub_ml[good_data_ml * (mask_class_2_ml + mask_class_1_ml)] < y_bins[y_index+1]))
        mask_3_ml = ((color_vi_ml[good_data_ml * mask_class_3_ml] > x_bins[x_index]) &
                  (color_vi_ml[good_data_ml * mask_class_3_ml] < x_bins[x_index+1]) &
                  (color_ub_ml[good_data_ml * mask_class_3_ml] > y_bins[y_index]) &
                  (color_ub_ml[good_data_ml * mask_class_3_ml] < y_bins[y_index+1]))

        if sum(mask_1_hum) > threshold_hum:
            v_band_sn_hist_cl1_hum[x_index, y_index] = np.nanmean(v_band_sn_hum[good_data_hum*mask_class_1_hum][mask_1_hum])
        if sum(mask_2_hum) > threshold_hum:
            v_band_sn_hist_cl2_hum[x_index, y_index] = np.nanmean(v_band_sn_hum[good_data_hum*mask_class_2_hum][mask_2_hum])
        if sum(mask_12_hum) > threshold_hum:
            v_band_sn_hist_cl12_hum[x_index, y_index] = np.nanmean(
                v_band_sn_hum[good_data_hum*(mask_class_2_hum + mask_class_1_hum)][mask_12_hum])
        if sum(mask_3_hum) > threshold_hum:
            v_band_sn_hist_cl3_hum[x_index, y_index] = np.nanmean(v_band_sn_hum[good_data_hum*mask_class_3_hum][mask_3_hum])

        if sum(mask_1_ml) > threshold_ml:
            v_band_sn_hist_cl1_ml[x_index, y_index] = np.nanmean(v_band_sn_ml[good_data_ml*mask_class_1_ml][mask_1_ml])
        if sum(mask_2_ml) > threshold_ml:
            v_band_sn_hist_cl2_ml[x_index, y_index] = np.nanmean(v_band_sn_ml[good_data_ml*mask_class_2_ml][mask_2_ml])
        if sum(mask_12_ml) > threshold_ml:
            v_band_sn_hist_cl12_ml[x_index, y_index] = np.nanmean(
                v_band_sn_ml[good_data_ml*(mask_class_2_ml + mask_class_1_ml)][mask_12_ml])
        if sum(mask_3_ml) > threshold_ml:
            v_band_sn_hist_cl3_ml[x_index, y_index] = np.nanmean(v_band_sn_ml[good_data_ml*mask_class_3_ml][mask_3_ml])


# fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(25, 13))
fig = plt.figure(figsize=(23, 13))
fontsize = 20

ax_1_hum = fig.add_axes([0.05, 0.525, 0.23, 0.45])
ax_2_hum = fig.add_axes([0.275, 0.525, 0.23, 0.45])
ax_12_hum = fig.add_axes([0.50, 0.525, 0.23, 0.45])
ax_3_hum = fig.add_axes([0.725, 0.525, 0.23, 0.45])

ax_1_ml = fig.add_axes([0.05, 0.065, 0.23, 0.45])
ax_2_ml = fig.add_axes([0.275, 0.065, 0.23, 0.45])
ax_12_ml = fig.add_axes([0.50, 0.065, 0.23, 0.45])
ax_3_ml = fig.add_axes([0.725, 0.065, 0.23, 0.45])

ax_cbar = fig.add_axes([0.955, 0.26, 0.015, 0.5])

ax_1_hum.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_2_hum.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_12_hum.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_3_hum.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax_1_ml.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_2_ml.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_12_ml.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax_3_ml.plot(model_vi_sol, model_ub_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax_1_hum.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_2_hum.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_12_hum.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_3_hum.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

ax_1_ml.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_2_ml.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_12_ml.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax_3_ml.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

norm_v_band_sn = Normalize(0, 40)
cmap_v_band_sn = 'viridis'

ax_1_hum.imshow(v_band_sn_hist_cl1_hum.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_2_hum.imshow(v_band_sn_hist_cl2_hum.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_12_hum.imshow(v_band_sn_hist_cl12_hum.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_3_hum.imshow(v_band_sn_hist_cl3_hum.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)

ax_1_ml.imshow(v_band_sn_hist_cl1_ml.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_2_ml.imshow(v_band_sn_hist_cl2_ml.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_12_ml.imshow(v_band_sn_hist_cl12_ml.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)
ax_3_ml.imshow(v_band_sn_hist_cl3_ml.T, extent=(-1.0, 2.3, 1.25, -2.2), cmap=cmap_v_band_sn, norm=norm_v_band_sn)


ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap_v_band_sn, norm=norm_v_band_sn, extend='neither', ticks=None)
ax_cbar.set_ylabel(r'S/N (V-band)', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)


ax_1_hum.text(-1, 1.45, 'Class 1 Human', fontsize=fontsize)
ax_2_hum.text(-1, 1.45, 'Class 2 Human', fontsize=fontsize)
ax_12_hum.text(-1, 1.45, 'Class 1|2 Human', fontsize=fontsize)
ax_3_hum.text(-1, 1.45, 'Class 3 Human', fontsize=fontsize)

ax_1_hum.text(-1, 1.7, 'N=%i' % (sum(mask_class_1_hum * mask_good_colors_hum)), fontsize=fontsize)
ax_2_hum.text(-1, 1.7, 'N=%i' % (sum(mask_class_2_hum * mask_good_colors_hum)), fontsize=fontsize)
ax_12_hum.text(-1, 1.7, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum)), fontsize=fontsize)
ax_3_hum.text(-1, 1.7, 'N=%i' % (sum(mask_class_3_hum * mask_good_colors_hum)), fontsize=fontsize)

ax_1_ml.text(-1, 1.45, 'Class 1 ML', fontsize=fontsize)
ax_2_ml.text(-1, 1.45, 'Class 2 ML', fontsize=fontsize)
ax_12_ml.text(-1, 1.45, 'Class 1|2 ML', fontsize=fontsize)
ax_3_ml.text(-1, 1.45, 'Class 3 ML', fontsize=fontsize)

ax_1_ml.text(-1, 1.7, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax_2_ml.text(-1, 1.7, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax_12_ml.text(-1, 1.7, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax_3_ml.text(-1, 1.7, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)

ax_1_hum.set_ylim(1.9, -2.2)
ax_2_hum.set_ylim(1.9, -2.2)
ax_12_hum.set_ylim(1.9, -2.2)
ax_3_hum.set_ylim(1.9, -2.2)
ax_1_ml.set_ylim(1.9, -2.2)
ax_2_ml.set_ylim(1.9, -2.2)
ax_12_ml.set_ylim(1.9, -2.2)
ax_3_ml.set_ylim(1.9, -2.2)

ax_1_hum.set_xlim(-1.2, 2.4)
ax_2_hum.set_xlim(-1.2, 2.4)
ax_12_hum.set_xlim(-1.2, 2.4)
ax_3_hum.set_xlim(-1.2, 2.4)
ax_1_ml.set_xlim(-1.2, 2.4)
ax_2_ml.set_xlim(-1.2, 2.4)
ax_12_ml.set_xlim(-1.2, 2.4)
ax_3_ml.set_xlim(-1.2, 2.4)

ax_1_hum.set_xticklabels([])
ax_2_hum.set_xticklabels([])
ax_12_hum.set_xticklabels([])
ax_3_hum.set_xticklabels([])

ax_2_hum.set_yticklabels([])
ax_12_hum.set_yticklabels([])
ax_3_hum.set_yticklabels([])
ax_2_ml.set_yticklabels([])
ax_12_ml.set_yticklabels([])
ax_3_ml.set_yticklabels([])

ax_1_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_2_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_12_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_3_ml.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_1_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_1_ml.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax_1_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_2_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_12_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_3_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_1_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_2_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_12_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_3_ml.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_v_band_sn.png')
fig.savefig('plot_output/ub_vi_v_band_sn.pdf')
fig.clf()

