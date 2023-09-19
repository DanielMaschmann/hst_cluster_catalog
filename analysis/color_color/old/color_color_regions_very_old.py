import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from matplotlib.patches import Ellipse
import matplotlib
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
import dust_tools.extinction_tools


def density_with_points(ax, x, y, binx=None, biny=None, threshold=1):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 170)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 170)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    kernel = Gaussian2DKernel(x_stddev=2.0)
    hist = convolve(hist, kernel)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask

    hist[hist <= threshold] = np.nan
    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap='Greys_r')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.', s=150)
    ax.set_ylim(ax.get_ylim()[::-1])


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
flux_f275w_sol = data_mod_sol['F275W_UVIS_CHIP2']
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
mag_nuv_sol = hf.conv_mjy2vega(flux=flux_f275w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol
model_nuvb_sol = mag_nuv_sol - mag_b_sol
model_nuvu_sol = mag_nuv_sol - mag_u_sol

# get model
hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f275w_sol50 = data_mod_sol50['F275W_UVIS_CHIP2']
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
mag_nuv_sol50 = hf.conv_mjy2vega(flux=flux_f275w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_sol50 = mag_v_sol50 - mag_i_sol50
model_ub_sol50 = mag_u_sol50 - mag_b_sol50
model_nuvb_sol50 = mag_nuv_sol50 - mag_b_sol50
model_nuvu_sol50 = mag_nuv_sol50 - mag_u_sol50


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
color_nuvb_hum = np.array([])
color_nuvu_hum = np.array([])
clcl_color_hum = np.array([])

color_vi_ml = np.array([])
color_ub_ml = np.array([])
color_nuvb_ml = np.array([])
color_nuvu_ml = np.array([])
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    color_nuvb_hum_12 = catalog_access.get_hst_color_nuvb(target=target)
    color_nuvu_hum_12 = catalog_access.get_hst_color_nuvu(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')
    color_nuvb_hum_3 = catalog_access.get_hst_color_nuvb(target=target, cluster_class='class3')
    color_nuvu_hum_3 = catalog_access.get_hst_color_nuvu(target=target, cluster_class='class3')
    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    color_nuvb_hum = np.concatenate([color_nuvb_hum, color_nuvb_hum_12, color_nuvb_hum_3])
    color_nuvu_hum = np.concatenate([color_nuvu_hum, color_nuvu_hum_12, color_nuvu_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb(target=target, classify='ml')
    color_nuvu_ml_12 = catalog_access.get_hst_color_nuvu(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    color_nuvb_ml_3 = catalog_access.get_hst_color_nuvb(target=target, classify='ml', cluster_class='class3')
    color_nuvu_ml_3 = catalog_access.get_hst_color_nuvu(target=target, classify='ml', cluster_class='class3')
    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    color_nuvb_ml = np.concatenate([color_nuvb_ml, color_nuvb_ml_12, color_nuvb_ml_3])
    color_nuvu_ml = np.concatenate([color_nuvu_ml, color_nuvu_ml_12, color_nuvu_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])


mask_good_colors_hum = (color_vi_hum > -2) & (color_vi_hum < 3) & (color_ub_hum > -3) & (color_ub_hum < 2)
mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_good_colors_ml = ((color_vi_ml > -2) & (color_vi_ml < 3) & (color_ub_ml > -3) & (color_ub_ml < 2) &
                       np.invert(((np.isnan(color_vi_ml) | np.isnan(color_ub_ml)) | (np.isinf(color_vi_ml) | np.isinf(color_ub_ml)))))
mask_class_1_ml = (clcl_color_ml == 1) & (clcl_qual_color_ml >= 0.9)
mask_class_2_ml = (clcl_color_ml == 2) & (clcl_qual_color_ml >= 0.9)
mask_class_3_ml = (clcl_color_ml == 3) & (clcl_qual_color_ml >= 0.9)


# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(17, 17))

figure = plt.figure(figsize=(18, 17))
fontsize = 26
ax_cc = figure.add_axes([0.08, 0.07, 0.9, 0.9])
ax_cbar = figure.add_axes([0.75, 0.9, 0.2, 0.015])

fontsize = 26
x_lim = (-1.2, 3.2)
y_lim = (1.9, -2.2)

ax_cc.plot([], [], color='white', label='N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)))
ax_cc.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}$')
ax_cc.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')


hist, xedges, yedges = np.histogram2d(color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                                      color_ub_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                                      bins=(np.linspace(x_lim[0], x_lim[1], 190), np.linspace(y_lim[1], y_lim[0], 190)))

kernel_std = 2.0
kernel = Gaussian2DKernel(x_stddev=kernel_std)
hist = convolve(hist, kernel)

kernal_rad = (xedges[1] - xedges[0]) * kernel_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.7, 1.2), width=kernal_rad, height=kernal_rad, angle=0, edgecolor='k', fc='None', lw=2)
ax_cc.add_patch(ellipse)
ax_cc.text(-0.7, 1.15, 'Smoothing kernel', horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)


over_dense_regions = hist > 1
mask_high_dens = np.zeros(len(color_vi_ml), dtype=bool)
for x_index in range(len(xedges)-1):
    for y_index in range(len(yedges)-1):
        if over_dense_regions[x_index, y_index]:
            mask = ((color_vi_ml > xedges[x_index]) & (color_vi_ml < xedges[x_index + 1]) &
                    (color_ub_ml > yedges[y_index]) & (color_ub_ml < yedges[y_index + 1]))
            mask_high_dens += mask

hist[hist <= 1] = np.nan
ax_cc.imshow(hist.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap='Greys_r')
ax_cc.scatter(color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml * ~mask_high_dens],
           color_ub_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml * ~mask_high_dens],
           color='k', marker='.', s=400)
ax_cc.set_ylim(ax_cc.get_ylim()[::-1])

# young clusters
ellipse = Ellipse(xy=(0.3, -1.35), width=0.3, height=0.9, angle=-60,
                        edgecolor='lightblue', fc='None', lw=5)
ax_cc.add_patch(ellipse)
ax_cc.text(0.3, -1.35, '1)', horizontalalignment='center', verticalalignment='bottom', color='lightblue', fontsize=fontsize)

# intermediate age
ellipse = Ellipse(xy=(0.65, -0.4), width=0.4, height=1.1, angle=-5,
                        edgecolor='tab:green', fc='None', lw=5)
ax_cc.add_patch(ellipse)
ax_cc.text(0.65, -0.4, '2)', horizontalalignment='center', verticalalignment='bottom', color='tab:green', fontsize=fontsize)


# globular clusters
ellipse = Ellipse(xy=(1.1, -0.1), width=0.3, height=0.5, angle=0,
                        edgecolor='crimson', fc='None', lw=5)
ax_cc.add_patch(ellipse)
ax_cc.text(1.1, -0.1, '3)', horizontalalignment='center', verticalalignment='bottom', color='crimson', fontsize=fontsize)


# # red plume
# ellipse = Ellipse(xy=(1.1, 0.6), width=0.5, height=0.7, angle=0,
#                         edgecolor='darkblue', fc='None', lw=5)
# ax_cc.add_patch(ellipse)
# ax_cc.text(1.1, 0.6, '4)', horizontalalignment='center', verticalalignment='bottom', color='darkblue', fontsize=fontsize)
#


# add reddening vector
# intrinsic colors
vi_int = 0.2
ub_int = -1.9
max_av = 2.0

v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4


max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)


max_color_ext_vi_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av+0.1)
max_color_ext_ub_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av+0.1)

x_av = np.linspace(vi_int, vi_int + max_color_ext_vi, 100)
y_av = np.linspace(ub_int, ub_int + max_color_ext_ub, 100)
av_value = np.linspace(0.0, max_av, 100)

cmap = matplotlib.cm.get_cmap('Spectral_r')
norm = matplotlib.colors.Normalize(vmin=av_value[:-1].min(), vmax=av_value[:-1].max())


points = np.array([x_av[:-1], y_av[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=1.0, zorder=9)

lc.set_array(av_value[:-1])
lc.set_linewidth(20)
line = ax_cc.add_collection(lc)

ax_cc.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap(norm(av_value[-1]))), zorder=10)

ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_xlabel(r'A$_{\rm V}$ [mag]', labelpad=4, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)



ax_cc.set_title('Class 1|2 ML', fontsize=fontsize)
ax_cc.set_xlim(x_lim)
ax_cc.set_ylim(y_lim)

ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# fig.subplots_adjust(wspace=0, hspace=0)
figure.savefig('plot_output/color_color_regions.png')
figure.savefig('plot_output/color_color_regions.pdf')
