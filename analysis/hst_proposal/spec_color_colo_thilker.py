import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from photometry_tools import helper_func as hf
from photometry_tools.data_access import CatalogAccess
from cigale_helper import cigale_wrapper as cw
from matplotlib.patches import ConnectionPatch
import matplotlib
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
import dust_tools.extinction_tools

# set star cluster mass to scale the model SEDs
cluster_mass = 1E4 * u.Msun
# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 15 * u.Mpc
# crate wrapper class object
cigale_wrapper_obj = cw.CigaleModelWrapper()

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)

# get model
hdu_a_no_dust = fits.open('../cigale_model/sfh2exp/hst_proposal/no_dust/out/models-block-0.fits')
data_mod_no_dust = hdu_a_no_dust[1].data
attenuation_no_dust = data_mod_no_dust['attenuation.A550']
age_mod_no_dust = data_mod_no_dust['sfh.age']
flux_f275w_no_dust = data_mod_no_dust['hst.wfc3.F275W']
flux_f555w_no_dust = data_mod_no_dust['hst.wfc3.F555W']
flux_f814w_no_dust = data_mod_no_dust['hst.wfc3.F814W']
flux_f336w_no_dust = data_mod_no_dust['hst.wfc3.F336W']
flux_f438w_no_dust = data_mod_no_dust['hst.wfc3.F438W']
flux_h_alpha_no_dust = data_mod_no_dust['line.H-alpha'] * 1e9
mag_v_no_dust = hf.conv_mjy2vega(flux=flux_f555w_no_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_no_dust = hf.conv_mjy2vega(flux=flux_f814w_no_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_no_dust = hf.conv_mjy2vega(flux=flux_f336w_no_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_no_dust = hf.conv_mjy2vega(flux=flux_f438w_no_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_nuv_no_dust = hf.conv_mjy2vega(flux=flux_f275w_no_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_no_dust = mag_v_no_dust - mag_i_no_dust
model_ub_no_dust = mag_u_no_dust - mag_b_no_dust

hdu_a_dust = fits.open('../cigale_model/sfh2exp/hst_proposal/dust/out/models-block-0.fits')
data_mod_dust = hdu_a_dust[1].data
attenuation_dust = data_mod_dust['attenuation.A550']
age_mod_dust = data_mod_dust['sfh.age']
flux_f275w_dust = data_mod_dust['hst.wfc3.F275W']
flux_f555w_dust = data_mod_dust['hst.wfc3.F555W']
flux_f814w_dust = data_mod_dust['hst.wfc3.F814W']
flux_f336w_dust = data_mod_dust['hst.wfc3.F336W']
flux_f438w_dust = data_mod_dust['hst.wfc3.F438W']
mag_v_dust = hf.conv_mjy2vega(flux=flux_f555w_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_dust = hf.conv_mjy2vega(flux=flux_f814w_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_dust = hf.conv_mjy2vega(flux=flux_f336w_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_dust = hf.conv_mjy2vega(flux=flux_f438w_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_nuv_dust = hf.conv_mjy2vega(flux=flux_f275w_dust, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_dust = mag_v_dust - mag_i_dust
model_ub_dust = mag_u_dust - mag_b_dust

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



figure = plt.figure(figsize=(30, 10))
fontsize = 26
ax_cc = figure.add_axes([0.68, 0.08, 0.26, 0.91])
ax_cbar = figure.add_axes([0.95, 0.15, 0.015, 0.77])

ax_sed_1 = figure.add_axes([0.04, 0.08, 0.29, 0.91])
ax_sed_2 = figure.add_axes([0.34, 0.08, 0.29, 0.91])
ax_cbar_av = figure.add_axes([0.46, 0.35, 0.15, 0.02])


sol_x_lim_ub = (0.52, 0.81)
sol_y_lim_ub = (-0.96, -1.22)
scatter_size = 80

ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-0.9, 1.8)


# ax_cc.plot(model_vi_no_dust, model_ub_no_dust, c=flux_h_alpha_no_dust, linewidth=3, label='BC03, Z=Z$_{\odot}$')

catalog_access.load_hst_cc_list(target_list=['ngc0628c'], classify='ml')
cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target='ngc0628c', classify='ml')
color_ub_hum_12 = catalog_access.get_hst_color_ub(target='ngc0628c', classify='ml')
color_vi_hum_12 = catalog_access.get_hst_color_vi(target='ngc0628c', classify='ml')

cmap = matplotlib.cm.get_cmap('Spectral_r')
norm = matplotlib.colors.Normalize(vmin=0, vmax=max(flux_h_alpha_no_dust))



ax_cc.plot([], [], color=cmap(norm(0)), linestyle='-', linewidth=10, zorder=0,
           label=r'Z=Z$_{\odot}$')
ax_cc.plot(model_vi_sol50, model_ub_sol50, color='tab:red', linestyle='--', linewidth=5, zorder=0,
           label=r'Z=Z$_{\odot}$/50')
ax_cc.scatter(color_vi_hum_12[cluster_class_hum_12 == 1], color_ub_hum_12[cluster_class_hum_12 == 1],
              color='darkorange', s=100, zorder=0, label='NGC 628 Class I')
ax_cc.scatter(color_vi_hum_12[cluster_class_hum_12 == 2], color_ub_hum_12[cluster_class_hum_12 == 2],
              color='forestgreen', s=100, zorder=0, label='NGC 628 Class II')




x_line = model_vi_no_dust
y_line = model_ub_no_dust
color_value = flux_h_alpha_no_dust

from scipy.interpolate import interp1d

# for index in range(len(x_line)-2):
#     print(index)
#     interp_y_func = interp1d(x_line[index:index+2], y_line[index:index+2], kind='quadratic')
#     interp_color_func = interp1d(x_line[index:index+2], color_value[index:index+2], kind='quadratic')
#
#     dummy_x_line = np.linspace(min(x_line[index:index+2]), max(x_line[index:index+2]), 100)
#     dummy_y_line = interp_y_func(dummy_x_line)
#     dummy_color = interp_color_func(dummy_x_line)
#     ax_cc.scatter(dummy_x_line, dummy_y_line, c=dummy_color, cmap=cmap)

# interp_y_func = interp1d(x_line, y_line, kind='nearest')
# interp_color_func = interp1d(x_line, color_value, kind='nearest')
#
# dummy_x_line = np.linspace(min(x_line), max(x_line), 10000)
# dummy_y_line = interp_y_func(dummy_x_line)
# dummy_color = interp_color_func(dummy_x_line)
#
# # for
# ax_cc.scatter(dummy_x_line, dummy_y_line, c=dummy_color)

# NPOINTS = len(x_line)
# for i in range(10):
#     ax_cc.set_prop_cycle(color=[cmap(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
#     for i in range(NPOINTS-1):
#         ax_cc.plot(x_line[i:i+2],y_line[i:i+2])
# #
points = np.array([x_line[:-1], y_line[:-1]]).T.reshape(-1, 1, 2)
# points = np.array([dummy_x_line[:-1], dummy_y_line[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.95, zorder=0)

lc.set_array(color_value[:-1])
# lc.set_array(dummy_color[:-1])
lc.set_linewidth(10)
line = ax_cc.add_collection(lc)


vi_1_my = model_vi_no_dust[age_mod_no_dust == 1][0]
ub_1_my = model_ub_no_dust[age_mod_no_dust == 1][0]

vi_2_my = model_vi_no_dust[age_mod_no_dust == 2][0]
ub_2_my = model_ub_no_dust[age_mod_no_dust == 2][0]

vi_3_my = model_vi_no_dust[age_mod_no_dust == 3][0]
ub_3_my = model_ub_no_dust[age_mod_no_dust == 3][0]

vi_4_my = model_vi_no_dust[age_mod_no_dust == 4][0]
ub_4_my = model_ub_no_dust[age_mod_no_dust == 4][0]

vi_5_my = model_vi_no_dust[age_mod_no_dust == 5][0]
ub_5_my = model_ub_no_dust[age_mod_no_dust == 5][0]

vi_10_my = model_vi_no_dust[age_mod_no_dust == 10][0]
ub_10_my = model_ub_no_dust[age_mod_no_dust == 10][0]

vi_100_my = model_vi_no_dust[age_mod_no_dust == 102][0]
ub_100_my = model_ub_no_dust[age_mod_no_dust == 102][0]

vi_1000_my = model_vi_no_dust[age_mod_no_dust == 1028][0]
ub_1000_my = model_ub_no_dust[age_mod_no_dust == 1028][0]

vi_10000_my = model_vi_no_dust[age_mod_no_dust == 10308][0]
ub_10000_my = model_ub_no_dust[age_mod_no_dust == 10308][0]





ax_cc.annotate('1,2,3 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my - 0.1, ub_1_my - 0.2), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my + 0.1, ub_10_my + 0.4), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax_cc.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my - 0.5, ub_1000_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

vi_10000_my_sol50 = model_vi_sol50[age_mod_sol50 == 10308][0]
ub_10000_my_sol50 = model_ub_sol50[age_mod_sol50 == 10308][0]
ax_cc.annotate('10 Gyr', xy=(vi_10000_my_sol50, ub_10000_my_sol50), xycoords='data',
                    xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))



ax_cc.text(0.9, -1.75, r'A$_{\rm V}$', fontsize=fontsize+3, rotation=-30)
ax_cc.text(0.5, -2.0, r'0', fontsize=fontsize+3, rotation=-30)
ax_cc.text(1.3, -1.5, r'2', fontsize=fontsize+3, rotation=-30)
# add reddening vector
# intrinsic colors

cmap_dust = matplotlib.cm.get_cmap('inferno_r')
norm_dust = matplotlib.colors.Normalize(vmin=0, vmax=max(attenuation_dust))
av_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

vi_int = 0.5
ub_int = -1.9

max_av = av_list[-1]

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

points = np.array([x_av[:-1], y_av[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap_dust, norm=norm_dust, alpha=0.95, zorder=0)

lc.set_array(av_value[:-1])
lc.set_linewidth(20)
line = ax_cc.add_collection(lc)

ax_cc.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap_dust(norm_dust(av_value[-1]))), zorder=10)



ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_cc.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
# ax_cc.set_xticklabels([])
ax_cc.legend(frameon=False, fontsize=fontsize, loc=3)

def plot_age_only(ax, age, color='k', linestyle='-', linewidth=3):
    id = np.where(age_mod_no_dust == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/hst_proposal/no_dust/out/%i_best_model.fits'%id,
                                            cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)

def plot_dust_only(ax, dust, color='k', linestyle='-', linewidth=3):
    id = np.where(attenuation_dust == dust)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/hst_proposal/dust/out/%i_best_model.fits'%id,
                                            cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


# age_list = [1, 2, 3, 5, 20, 30, 50, 100, 200, 300, 500, 900]
age_list = [5, 10, 100, 1000, 10308]
# color_list = ['r', 'b', 'm', 'g', 'cyan', 'k', 'r', 'b', 'm', 'g', 'cyan', 'k']
color_list = ['r', 'b', 'm', 'g', 'orange']
age_list = list(reversed(age_list))
color_list = list(reversed(color_list))

for age, color in zip(age_list, color_list):
    plot_age_only(ax=ax_sed_1, age=age, color=color, linestyle='-', linewidth=3)


for dust in av_list:
    color = cmap_dust(norm_dust(dust))
    plot_dust_only(ax=ax_sed_2, dust=dust, color=color, linestyle='-', linewidth=3)

ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_ylabel(r'H$\alpha$ flux [10$^{-9}$ W/m$^{2}$] ', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)

ColorbarBase(ax_cbar_av, orientation='horizontal', cmap=cmap_dust, norm=norm_dust, extend='neither', ticks=None)
ax_cbar_av.set_xlabel(r'A$_{\rm V}$ [mag]', labelpad=0, fontsize=fontsize)
ax_cbar_av.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)





ax_sed_1.annotate(r'Ly$\alpha$', xy=(121.57, 2e-1), xycoords='data',
                    xytext=(170.57, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.annotate(r'  H$\alpha$', xy=(656.5, 2e-1), xycoords='data',
                    xytext=(856.5, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.text(856.5, 2e-1, 'F657N', fontsize=fontsize-5)
ax_sed_1.annotate(r' Pa$\alpha$', xy=(1875, 4e-2), xycoords='data',
                    xytext=(2575, 7e-2), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.text(2575, 4.7e-2, 'F187N', fontsize=fontsize-5)

ax_sed_2.annotate(r'Ly$\alpha$', xy=(121.57, 2e-1), xycoords='data',
                    xytext=(170.57, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_2.annotate(r'  H$\alpha$', xy=(656.5, 2e-1), xycoords='data',
                    xytext=(856.5, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_2.text(856.5, 2e-1, 'F657N', fontsize=fontsize-5)
ax_sed_2.annotate(r' Pa$\alpha$', xy=(1875, 4e-2), xycoords='data',
                    xytext=(2575, 7e-2), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_2.text(2575, 4.7e-2, 'F187N', fontsize=fontsize-5)


# plot observation filters
cigale_wrapper_obj.plot_hst_filters(ax=ax_sed_1, fontsize=fontsize-5, color='k')
# cigale_wrapper_obj.plot_hst_nircam_filters(ax=ax_sed_1, fontsize=fontsize-5,)

ax_sed_1.axvspan(650, 662, alpha=0.3, color='k')


ax_sed_1.axvspan(1.331*1000, 1.668*1000, alpha=0.3, color='r')
ax_sed_1.axvspan(1.863*1000, 1.885*1000, alpha=0.3, color='r')
ax_sed_1.axvspan(2.831*1000, 3.157*1000, alpha=0.3, color='r')




ax_sed_1.text(320, 8e-1, r'HST WFC3', color='k', fontsize=fontsize)
ax_sed_1.text(1500, 8e-1, r'NIRCAM', color='red', fontsize=fontsize)


ax_sed_1.text(150, 3e-2, r'5 Myr', color='r', fontsize=fontsize)
ax_sed_1.text(150, 3e-3, r'10 Myr', color='b', fontsize=fontsize)
ax_sed_1.text(150, 6e-4, r'100 Myr', color='m', fontsize=fontsize)
ax_sed_1.text(200, 5e-5, r'1 Gyr', color='g', fontsize=fontsize)
ax_sed_1.text(500, 9e-6, r'10 Gyr', color='orange', fontsize=fontsize)



ax_sed_1.text(300, 2e-6, r'Z=Z$_{\odot}$, M$_{*}$=10$^{4}$ M$_{\odot}$, D=15 Mpc', fontsize=fontsize)
ax_sed_2.text(300, 5e-6, r'Z=Z$_{\odot}$, M$_{*}$=10$^{4}$ M$_{\odot}$, D=15 Mpc', fontsize=fontsize)
ax_sed_2.text(300, 2e-6, r'Age=5 Myr, A$_{\rm V}$=0-2 mag', fontsize=fontsize)

ax_sed_1.set_xlim(0.08*1e3, 4 * 1e3)
ax_sed_2.set_xlim(0.08*1e3, 4 * 1e3)

ax_sed_1.set_ylim(7e-7, 1.5e0)
ax_sed_2.set_ylim(7e-7, 1.5e0)

ax_sed_1.set_xscale('log')
ax_sed_1.set_yscale('log')
ax_sed_2.set_xscale('log')
ax_sed_2.set_yscale('log')


ax_sed_1.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed_2.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed_1.set_ylabel(r'F$_{\nu}$ [mJy]', labelpad=-7, fontsize=fontsize)
ax_sed_2.set_yticklabels([])

ax_sed_1.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
ax_sed_2.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
# ax_sed_1.legend(loc='upper left', fontsize=fontsize-6, ncol=3, columnspacing=1, handlelength=1, handletextpad=0.6)


plt.savefig('plot_output/sed_color_color_ages.png')
plt.savefig('plot_output/sed_color_color_ages.pdf')

exit()