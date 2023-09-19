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


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)


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




color_vi_ml = np.array([])
color_ub_ml = np.array([])
color_vi_err_ml = np.array([])
color_ub_err_ml = np.array([])
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])
ebv_ml = np.array([])



for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_err_ml_12 = catalog_access.get_hst_color_ub_err(target=target, classify='ml')
    color_vi_err_ml_12 = catalog_access.get_hst_color_vi_err(target=target, classify='ml')
    ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')

    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_err_ml_3 = catalog_access.get_hst_color_ub_err(target=target, classify='ml', cluster_class='class3')
    color_vi_err_ml_3 = catalog_access.get_hst_color_vi_err(target=target, classify='ml', cluster_class='class3')
    ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')

    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])

    color_vi_err_ml = np.concatenate([color_vi_err_ml, color_vi_err_ml_12, color_vi_err_ml_3])
    color_ub_err_ml = np.concatenate([color_ub_err_ml, color_ub_err_ml_12, color_ub_err_ml_3])

    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])
    ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])


mask_good_colors_ml = ((color_vi_ml > -2) & (color_vi_ml < 3) & (color_ub_ml > -3) & (color_ub_ml < 2) & (color_vi_ml != 0) & (color_ub_ml != 0) &
                       np.invert(((np.isnan(color_vi_ml) | np.isnan(color_ub_ml)) | (np.isinf(color_vi_ml) | np.isinf(color_ub_ml)))))
mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3


x_lim = (-0.6, 2.1)
y_lim = (1.2, -2.2)

# bins
x_bins_gauss = np.linspace(x_lim[0], x_lim[1], 100)
y_bins_gauss = np.linspace(y_lim[1], y_lim[0], 100)
# get a mesh
x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
for color_index in range(len(color_vi_ml)):
    if not (mask_class_1_ml + mask_class_2_ml)[color_index]:
        continue
    gauss_map += gauss2d(x=x_mesh, y=y_mesh, x0=color_vi_ml[color_index], y0=color_ub_ml[color_index],
                         sig_x=color_vi_err_ml[color_index], sig_y=color_ub_err_ml[color_index])


# bins
x_bins = np.linspace(x_lim[0], x_lim[1], 100)
y_bins = np.linspace(y_lim[1], y_lim[0], 100)
threshold = 3
snr_vi_map = np.zeros((len(x_bins), len(y_bins)))
snr_ub_map = np.zeros((len(x_bins), len(y_bins)))
ebv_map = np.zeros((len(x_bins), len(y_bins)))

snr_vi_map = np.zeros(snr_vi_map.shape) * np.nan
snr_ub_map = np.zeros(snr_ub_map.shape) * np.nan
ebv_map = np.zeros(ebv_map.shape) * np.nan

for x_index in range(len(x_bins)-1):
    for y_index in range(len(y_bins)-1):
        mask_in_bin = ((color_vi_ml > x_bins[x_index]) & (color_vi_ml < x_bins[x_index + 1]) &
                       (color_ub_ml > y_bins[y_index]) & (color_ub_ml < y_bins[y_index + 1]))
        mask_selected_obj = mask_in_bin * (mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml
        if sum(mask_selected_obj) > threshold:
            snr_vi_map[x_index, y_index] = np.nanmean(color_vi_err_ml[mask_selected_obj])
            snr_ub_map[x_index, y_index] = np.nanmean(color_ub_err_ml[mask_selected_obj])
            ebv_map[x_index, y_index] = np.nanmedian(ebv_ml[mask_selected_obj])


figure = plt.figure(figsize=(30, 10))
fontsize = 26

ax_cc_sn_vi = figure.add_axes([0.05, 0.05, 0.23, 0.9])
ax_cc_sn_ub = figure.add_axes([0.28, 0.05, 0.23, 0.9])
ax_cc_reg = figure.add_axes([0.51, 0.05, 0.23, 0.9])
ax_cc_ebv = figure.add_axes([0.74, 0.05, 0.23, 0.9])

ax_cbar_sn_vi = figure.add_axes([0.07, 0.96, 0.19, 0.015])
ax_cbar_sn_ub = figure.add_axes([0.30, 0.96, 0.19, 0.015])
ax_cbar_ebv = figure.add_axes([0.76, 0.96, 0.19, 0.015])


ax_cc_sn_vi.plot([], [], color='white', label='N=%i' % (sum((mask_class_1_ml + mask_class_2_ml))))
ax_cc_sn_vi.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}$')
ax_cc_sn_vi.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--', label=r'BC03, Z$_{\odot}/50$')

ax_cc_sn_ub.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_sn_ub.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_reg.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_reg.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')

ax_cc_ebv.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=4, linestyle='--')
ax_cc_ebv.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=4, linestyle='--')


cmap_vi = matplotlib.cm.get_cmap('viridis')
norm_vi = matplotlib.colors.Normalize(vmin=0, vmax=0.12)

cmap_ub = matplotlib.cm.get_cmap('plasma')
norm_ub = matplotlib.colors.Normalize(vmin=0, vmax=0.5)

cmap_ebv = matplotlib.cm.get_cmap('cividis')
norm_ebv = matplotlib.colors.Normalize(vmin=0, vmax=0.5)


ax_cc_sn_vi.imshow(snr_vi_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_vi, norm=norm_vi)
ax_cc_sn_ub.imshow(snr_ub_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_ub, norm=norm_ub)

ax_cc_reg.imshow(gauss_map, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
             cmap='Greys', vmin=10, vmax=np.max(gauss_map))
ax_cc_ebv.imshow(ebv_map.T, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]), cmap=cmap_ebv, norm=norm_ebv)



ColorbarBase(ax_cbar_sn_vi, orientation='horizontal', cmap=cmap_vi, norm=norm_vi, extend='neither', ticks=None)
ax_cbar_sn_vi.set_xlabel(r'$\sigma_{\rm V-I}$', labelpad=-4, fontsize=fontsize)
ax_cbar_sn_vi.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)

ColorbarBase(ax_cbar_sn_ub, orientation='horizontal', cmap=cmap_ub, norm=norm_ub, extend='neither', ticks=None)
ax_cbar_sn_ub.set_xlabel(r'$\sigma_{\rm U-B}$', labelpad=-4, fontsize=fontsize)
ax_cbar_sn_ub.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)

ColorbarBase(ax_cbar_ebv, orientation='horizontal', cmap=cmap_ebv, norm=norm_ebv, extend='neither', ticks=None)
ax_cbar_ebv.set_xlabel(r'E(B-V) [mag]', labelpad=-4, fontsize=fontsize)
ax_cbar_ebv.tick_params(axis='both', which='both', width=2, direction='in', top=False, bottom=True, labelbottom=True, labeltop=False, labelsize=fontsize)


ax_cc_sn_vi.set_xlim(x_lim)
ax_cc_sn_ub.set_xlim(x_lim)
ax_cc_reg.set_xlim(x_lim)
ax_cc_ebv.set_xlim(x_lim)
ax_cc_sn_vi.set_ylim(y_lim)
ax_cc_sn_ub.set_ylim(y_lim)
ax_cc_reg.set_ylim(y_lim)
ax_cc_ebv.set_ylim(y_lim)



# young clusters
ellipse = Ellipse(xy=(0.3, -1.35), width=0.3, height=0.9, angle=-60,
                        edgecolor='darkblue', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(0.3, -1.35, '1)', horizontalalignment='center', verticalalignment='bottom', color='darkblue', fontsize=fontsize+10)

# intermediate age
ellipse = Ellipse(xy=(0.65, -0.4), width=0.4, height=1.1, angle=-5,
                        edgecolor='tab:green', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(0.65, -0.4, '2)', horizontalalignment='center', verticalalignment='bottom', color='tab:green', fontsize=fontsize+10)


# globular clusters
ellipse = Ellipse(xy=(1.1, -0.1), width=0.3, height=0.5, angle=0,
                        edgecolor='crimson', fc='None', lw=5)
ax_cc_reg.add_patch(ellipse)
ax_cc_reg.text(1.1, -0.1, '3)', horizontalalignment='center', verticalalignment='bottom', color='crimson', fontsize=fontsize+10)



# add reddening vector
# intrinsic colors
vi_int = 0.5
ub_int = -1.9
max_av = 2.2
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

cmap_av = matplotlib.cm.get_cmap('Spectral_r')
norm_av = matplotlib.colors.Normalize(vmin=av_value[:-1].min(), vmax=av_value[:-1].max())

points = np.array([x_av[:-1], y_av[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap_av, norm=norm_av, alpha=1.0, zorder=9)

lc.set_array(av_value[:-1])
lc.set_linewidth(20)
line = ax_cc_reg.add_collection(lc)

ax_cc_reg.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap_av(norm_av(av_value[-1]))), zorder=10)


ax_cc_reg.text(0.90, -1.67, r'A$_{\rm V}$', fontsize=fontsize+3, rotation=-30)
ax_cc_reg.text(0.48, -1.93, r'0', fontsize=fontsize+3, rotation=-30)
ax_cc_reg.text(1.32, -1.40, r'2', fontsize=fontsize+3, rotation=-30)


# ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap_av, norm=norm_av, extend='neither', ticks=[0, 0.5, 1, 1.5, 2.0])
# ax_cbar.set_xlabel(r'A$_{\rm V}$ [mag]', labelpad=4, fontsize=fontsize)
# ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
#                     labeltop=True, labelsize=fontsize)
#


# ax_cc_reg.set_title('Class 1|2 ML', fontsize=fontsize)

ax_cc_sn_ub.set_yticklabels([])
ax_cc_reg.set_yticklabels([])
ax_cc_ebv.set_yticklabels([])

ax_cc_sn_vi.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_ub.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_reg.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_ebv.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc_sn_vi.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_cc_reg.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc_sn_vi.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_sn_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_reg.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

# fig.subplots_adjust(wspace=0, hspace=0)
figure.savefig('plot_output/color_color_regions.png')
figure.savefig('plot_output/color_color_regions.pdf')
