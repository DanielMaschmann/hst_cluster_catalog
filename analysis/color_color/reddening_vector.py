import numpy as np

import dust_tools.extinction_tools
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from dust_tools import extinction_tools


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


# red plume
ellipse = Ellipse(xy=(1.1, 0.6), width=0.5, height=0.7, angle=0,
                        edgecolor='darkblue', fc='None', lw=5)
ax_cc.add_patch(ellipse)
ax_cc.text(1.1, 0.6, '4)', horizontalalignment='center', verticalalignment='bottom', color='darkblue', fontsize=fontsize)

x_ebv = np.linspace(0, 1.7, 20)
y_ebv = np.linspace(-1.5, -0.3, 20)
ebv_value = np.linspace(0.0, 2.0, 20)
cmap = matplotlib.cm.get_cmap('Reds')
norm = matplotlib.colors.Normalize(vmin=ebv_value[:-1].min(), vmax=ebv_value[:-1].max())

ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_xlabel(r'E(B-V) [mag]', labelpad=4, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)


ax_cc.annotate("", xy=(x_ebv[-1], y_ebv[-1]), xytext=(x_ebv[-2], y_ebv[-2]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1, color=cmap(norm(ebv_value[-1]))))


points = np.array([x_ebv[:-1], y_ebv[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.8, zorder=10)

lc.set_array(ebv_value[:-1])
lc.set_linewidth(20)
line = ax_cc.add_collection(lc)


ax_cc.set_xlim(x_lim)
ax_cc.set_ylim(y_lim)

ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



# add reddening vector
# intrinsic colors
ub_obs = -0.5
vi_obs = 2.5

ebv = 1.0


u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4

print('u_wave ', u_wave)
print('b_wave ', b_wave)
print('v_wave ', v_wave)
print('i_wave ', i_wave)


color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89(wave1=u_wave, wave2=b_wave, ebv=ebv)
color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89(wave1=v_wave, wave2=i_wave, ebv=ebv)


print('ub_int ', ub_int)
print('vi_int ', vi_int)

ax_cc.scatter(vi_obs, ub_obs, s=100)
ax_cc.scatter(vi_int, ub_int, s=100)




# fig.subplots_adjust(wspace=0, hspace=0)
figure.savefig('plot_output/reddening_vector.png')
# figure.savefig('plot_output/reddening_vector.pdf')
