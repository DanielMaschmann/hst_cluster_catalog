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
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.analysis_tools import CigaleModelWrapper
from photometry_tools import plotting_tools
from scipy.constants import c
import multicolorfits as mcf


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
hdu_a_no_dust = fits.open('../cigale_model/sfh2exp/hst_proposal/no_dust_dl/out/models-block-0.fits')
data_mod_no_dust = hdu_a_no_dust[1].data
age_mod_no_dust = data_mod_no_dust['sfh.age']
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
model_vi_no_dust = mag_v_no_dust - mag_i_no_dust
model_ub_no_dust = mag_u_no_dust - mag_b_no_dust

hdu_a_dust = fits.open('../cigale_model/sfh2exp/hst_proposal/dust/out/models-block-0.fits')
data_mod_dust = hdu_a_dust[1].data
attenuation_dust = data_mod_dust['attenuation.A550']
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
model_vi_dust = mag_v_dust - mag_i_dust
model_ub_dust = mag_u_dust - mag_b_dust

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



figure = plt.figure(figsize=(35, 10))
fontsize = 26
# ax_cc = figure.add_axes([0.728, 0.08, 0.23, 0.91])
# ax_cbar = figure.add_axes([0.96, 0.15, 0.01, 0.77])
#
# ax_sed_1 = figure.add_axes([0.035, 0.08, 0.24, 0.91])
#
# ax_sed_2 = figure.add_axes([0.3, 0.08, 0.385, 0.91])
#
# ax_hst_1 = figure.add_axes([0.415, 0.775, 0.2, 0.2])
# ax_hst_2 = figure.add_axes([0.415, 0.085, 0.2, 0.2])
#
# ax_nircam_1 = figure.add_axes([0.52, 0.775, 0.2, 0.2])
# ax_nircam_2 = figure.add_axes([0.52, 0.085, 0.2, 0.2])


ax_cc = figure.add_axes([0.035, 0.08, 0.23, 0.91])
ax_cbar = figure.add_axes([0.267, 0.15, 0.01, 0.77])

ax_sed_1 = figure.add_axes([0.31, 0.08, 0.24, 0.91])

ax_sed_2 = figure.add_axes([0.61, 0.08, 0.385, 0.91])

ax_hst_1 = figure.add_axes([0.73, 0.775, 0.2, 0.2])
ax_hst_2 = figure.add_axes([0.73, 0.085, 0.2, 0.2])

ax_nircam_1 = figure.add_axes([0.85, 0.775, 0.2, 0.2])
ax_nircam_2 = figure.add_axes([0.85, 0.085, 0.2, 0.2])



ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-0.9, 1.8)


catalog_access.load_hst_cc_list(target_list=['ngc0628c'], classify='ml')
cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target='ngc0628c', classify='ml')
color_ub_hum_12 = catalog_access.get_hst_color_ub(target='ngc0628c', classify='ml')
color_vi_hum_12 = catalog_access.get_hst_color_vi(target='ngc0628c', classify='ml')


cmap_age = matplotlib.cm.get_cmap('Spectral_r')
norm_age = matplotlib.colors.Normalize(vmin=0, vmax=max(flux_h_alpha_no_dust))


ax_cc.plot([], [], color=cmap_age(norm_age(0)), linestyle='-', linewidth=10, zorder=0,
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
points = np.array([x_line[:-1], y_line[:-1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap_age, norm=norm_age, alpha=0.95, zorder=0)
lc.set_array(color_value[:-1])
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
vi_10000_my_sol50 = model_vi_sol50[age_mod_sol50 == 10308][0]
ub_10000_my_sol50 = model_ub_sol50[age_mod_sol50 == 10308][0]

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
# ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
#                     xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my - 0.15, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_cc.annotate('10 Gyr', xy=(vi_10000_my_sol50, ub_10000_my_sol50), xycoords='data',
                    xytext=(vi_10000_my - 0.15, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax_cc.text(0.9, -1.75, r'A$_{\rm V}$', fontsize=fontsize+3, rotation=-30)
ax_cc.text(0.5, -2.0, r'0', fontsize=fontsize+3, rotation=-30)
ax_cc.text(1.3, -1.5, r'2', fontsize=fontsize+3, rotation=-30)

# add reddening vector
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
line_dust = ax_cc.add_collection(lc)
ax_cc.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap_dust(norm_dust(av_value[-1]))), zorder=10)

ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_cc.set_ylabel('U (F336W) - B (F438W)', labelpad=-5, fontsize=fontsize)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
# ax_cc.set_xticklabels([])
ax_cc.legend(frameon=False, fontsize=fontsize, loc=3)




def plot_age_only(ax, age, color='k', linestyle='-', linewidth=3):
    id = np.where(age_mod_no_dust == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/hst_proposal/no_dust_dl/out/%i_best_model.fits'%id,
                                            cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)

def plot_dust_only(ax, dust, color='k', linestyle='-', linewidth=3):
    id = np.where(attenuation_dust == dust)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/hst_proposal/dust/out/%i_best_model.fits'%id,
                                            cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


age_list = [5, 10, 100, 1000, 10308]
color_list = ['r', 'b', 'm', 'g', 'orange']
age_list = list(reversed(age_list))
color_list = list(reversed(color_list))

for age, color in zip(age_list, color_list):
    plot_age_only(ax=ax_sed_1, age=age, color=color, linestyle='-', linewidth=3)


# for dust in av_list:
#     color = cmap_dust(norm_dust(dust))
#     plot_dust_only(ax=ax_sed_2, dust=dust, color=color, linestyle='-', linewidth=3)

ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap_age, norm=norm_age, extend='neither', ticks=None)
ax_cbar.set_ylabel(r'H$\alpha$ flux [10$^{-9}$ W/m$^{2}$] ', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)

# ColorbarBase(ax_cbar_av, orientation='horizontal', cmap=cmap_dust, norm=norm_dust, extend='neither', ticks=None)
# ax_cbar_av.set_xlabel(r'A$_{\rm V}$ [mag]', labelpad=0, fontsize=fontsize)
# ax_cbar_av.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
#                     labeltop=True, labelsize=fontsize)
#




target = 'ngc0628c'
band_list = ['F275W', 'F336W', 'F435W', 'F555W', 'F658N', 'F814W', 'F200W', 'F300M', 'F335M', 'F360M']
# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc0628',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=band_list, load_err=False)

ra_1 = 24.16348
dec_1 = 15.76589

ra_2 = 24.18709
dec_2 = 15.79751

# size of image
size_of_cutout = (3, 3)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)

cutout_dict_1 = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_1, dec_cutout=dec_1,
                                                     cutout_size=size_of_cutout, include_err=False,
                                                     band_list=band_list)
cutout_dict_2 = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_2, dec_cutout=dec_2,
                                                     cutout_size=size_of_cutout, include_err=False,
                                                     band_list=band_list)
source_1 = SkyCoord(ra=ra_1, dec=dec_1, unit=(u.degree, u.degree), frame='fk5')
source_2 = SkyCoord(ra=ra_2, dec=dec_2, unit=(u.degree, u.degree), frame='fk5')


# phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr', band_list=band_list)


grey_hst_h_alpha_r_1 = mcf.greyRGBize_image(cutout_dict_1['F658N_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_hst_h_alpha_g_1 = mcf.greyRGBize_image(cutout_dict_1['F555W_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_hst_h_alpha_b_1 = mcf.greyRGBize_image(cutout_dict_1['F435W_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
hst_h_alpha_r_purple_1 = mcf.colorize_image(grey_hst_h_alpha_r_1, '#FF3131', colorintype='hex', gammacorr_color=2.2)
hst_h_alpha_g_orange_1 = mcf.colorize_image(grey_hst_h_alpha_g_1, '#FFF9DB', colorintype='hex', gammacorr_color=2.2)
hst_h_alpha_b_blue_1 = mcf.colorize_image(grey_hst_h_alpha_b_1, '#1773E9', colorintype='hex', gammacorr_color=2.2)
rgb_hst_h_alpha_image_1 = mcf.combine_multicolor([hst_h_alpha_r_purple_1, hst_h_alpha_g_orange_1, hst_h_alpha_b_blue_1], gamma=2.2, inverse=False)

grey_hst_h_alpha_r_2 = mcf.greyRGBize_image(cutout_dict_2['F658N_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_hst_h_alpha_g_2 = mcf.greyRGBize_image(cutout_dict_2['F555W_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_hst_h_alpha_b_2 = mcf.greyRGBize_image(cutout_dict_2['F435W_img_cutout'].data, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
hst_h_alpha_r_purple_2 = mcf.colorize_image(grey_hst_h_alpha_r_2, '#FF3131', colorintype='hex', gammacorr_color=2.2)
hst_h_alpha_g_orange_2 = mcf.colorize_image(grey_hst_h_alpha_g_2, '#FFF9DB', colorintype='hex', gammacorr_color=2.2)
hst_h_alpha_b_blue_2 = mcf.colorize_image(grey_hst_h_alpha_b_2, '#1773E9', colorintype='hex', gammacorr_color=2.2)
rgb_hst_h_alpha_image_2 = mcf.combine_multicolor([hst_h_alpha_r_purple_2, hst_h_alpha_g_orange_2, hst_h_alpha_b_blue_2], gamma=2.2, inverse=False)


nircam_img_r_1 = plotting_tools.reproject_image(data=cutout_dict_1['F360M_img_cutout'].data,
                                              wcs=cutout_dict_1['F360M_img_cutout'].wcs,
                                              new_wcs=cutout_dict_1['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_1['F360M_img_cutout'].data.shape)
nircam_img_g_1 = plotting_tools.reproject_image(data=cutout_dict_1['F335M_img_cutout'].data,
                                              wcs=cutout_dict_1['F335M_img_cutout'].wcs,
                                              new_wcs=cutout_dict_1['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_1['F360M_img_cutout'].data.shape)
nircam_img_b_1 = plotting_tools.reproject_image(data=cutout_dict_1['F200W_img_cutout'].data,
                                              wcs=cutout_dict_1['F200W_img_cutout'].wcs,
                                              new_wcs=cutout_dict_1['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_1['F360M_img_cutout'].data.shape)

nircam_img_r_2 = plotting_tools.reproject_image(data=cutout_dict_2['F360M_img_cutout'].data,
                                              wcs=cutout_dict_2['F360M_img_cutout'].wcs,
                                              new_wcs=cutout_dict_2['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_2['F360M_img_cutout'].data.shape)
nircam_img_g_2 = plotting_tools.reproject_image(data=cutout_dict_2['F335M_img_cutout'].data,
                                              wcs=cutout_dict_2['F335M_img_cutout'].wcs,
                                              new_wcs=cutout_dict_2['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_2['F360M_img_cutout'].data.shape)
nircam_img_b_2 = plotting_tools.reproject_image(data=cutout_dict_2['F200W_img_cutout'].data,
                                              wcs=cutout_dict_2['F200W_img_cutout'].wcs,
                                              new_wcs=cutout_dict_2['F360M_img_cutout'].wcs,
                                              new_shape=cutout_dict_2['F360M_img_cutout'].data.shape)

grey_nircam_r_1 = mcf.greyRGBize_image(nircam_img_r_1, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_nircam_g_1 = mcf.greyRGBize_image(nircam_img_g_1, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_nircam_b_1 = mcf.greyRGBize_image(nircam_img_b_1, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
nircam_r_purple_1 = mcf.colorize_image(grey_nircam_r_1, '#EE4B2B', colorintype='hex', gammacorr_color=2.2)
nircam_g_orange_1 = mcf.colorize_image(grey_nircam_g_1, '#AAFF00', colorintype='hex', gammacorr_color=2.2)
nircam_b_blue_1 = mcf.colorize_image(grey_nircam_b_1, '#0096FF', colorintype='hex', gammacorr_color=2.2)
rgb_nircam_image_1 = mcf.combine_multicolor([nircam_r_purple_1, nircam_g_orange_1, nircam_b_blue_1], gamma=2.2, inverse=False)

grey_nircam_r_2 = mcf.greyRGBize_image(nircam_img_r_2, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_nircam_g_2 = mcf.greyRGBize_image(nircam_img_g_2, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
grey_nircam_b_2 = mcf.greyRGBize_image(nircam_img_b_2, rescalefn='asinh', scaletype='perc', min_max=[1.,99.9], gamma=2.2, checkscale=False)
nircam_r_purple_2 = mcf.colorize_image(grey_nircam_r_2, '#EE4B2B', colorintype='hex', gammacorr_color=2.2)
nircam_g_orange_2 = mcf.colorize_image(grey_nircam_g_2, '#AAFF00', colorintype='hex', gammacorr_color=2.2)
nircam_b_blue_2 = mcf.colorize_image(grey_nircam_b_2, '#0096FF', colorintype='hex', gammacorr_color=2.2)
rgb_nircam_image_2 = mcf.combine_multicolor([nircam_r_purple_2, nircam_g_orange_2, nircam_b_blue_2], gamma=2.2, inverse=False)


hdu_hny_results = fits.open('data/fitted_data/hst_ha_nircam_young_results.fits')
data_hny = hdu_hny_results[1].data
index_hny = data_hny['id'] == 1930
hdu_hoy_results = fits.open('data/fitted_data/hst_young_results.fits')
data_hoy = hdu_hoy_results[1].data
index_hoy = data_hoy['id'] == 1930

hdu_hng_results = fits.open('data/fitted_data/hst_ha_nircam_globs_results.fits')
data_hng = hdu_hng_results[1].data
index_hng = data_hng['id'] == 4460
hdu_hog_results = fits.open('data/fitted_data/globs_results.fits')
data_hog = hdu_hog_results[1].data
index_hog = data_hog['id'] == 4460

av_hny = data_hny['best.attenuation.A550'][index_hny][0]
age_hny = data_hny['best.sfh.age'][index_hny][0]
mstar_hny = data_hny['best.stellar.m_star'][index_hny][0]
av_hoy = data_hoy['best.attenuation.A550'][index_hoy][0]
age_hoy = data_hoy['best.sfh.age'][index_hoy][0]
mstar_hoy = data_hoy['best.stellar.m_star'][index_hoy][0]

av_hng = data_hng['best.attenuation.A550'][index_hng][0]
age_hng = data_hng['best.sfh.age'][index_hng][0]
mstar_hng = data_hng['best.stellar.m_star'][index_hng][0]
av_hog = data_hog['best.attenuation.A550'][index_hog][0]
age_hog = data_hog['best.sfh.age'][index_hog][0]
mstar_hog = data_hog['best.stellar.m_star'][index_hog][0]


print('av young HST + NIRCAM + halpha ', av_hny)
print('age young HST + NIRCAM + halpha ', age_hny)
print('mstar young HST + NIRCAM + halpha ', mstar_hny)
print('av young HST only ', av_hoy)
print('age young HST only ', age_hoy)
print('mstar young HST only ', mstar_hoy)
print('av globular HST + NIRCAM + halpha ', av_hng)
print('age globular HST + NIRCAM + halpha ', age_hng)
print('mstar globular HST + NIRCAM + halpha ', mstar_hng)
print('av globular HST only ', av_hog)
print('age globular HST only ', age_hog)
print('mstar globular HST only ', mstar_hog)

mag_v_hny = hf.conv_mjy2vega(flux=data_hny['best.F555W_UVIS_CHIP2'][index_hny][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_hny = hf.conv_mjy2vega(flux=data_hny['best.F814W_UVIS_CHIP2'][index_hny][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_hny = hf.conv_mjy2vega(flux=data_hny['best.F336W_UVIS_CHIP2'][index_hny][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_hny = hf.conv_mjy2vega(flux=data_hny['best.F435W_ACS'][index_hny][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_hny = mag_v_hny - mag_i_hny
model_ub_hny = mag_u_hny - mag_b_hny

mag_v_hoy = hf.conv_mjy2vega(flux=data_hoy['best.F555W_UVIS_CHIP2'][index_hoy][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_hoy = hf.conv_mjy2vega(flux=data_hoy['best.F814W_UVIS_CHIP2'][index_hoy][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_hoy = hf.conv_mjy2vega(flux=data_hoy['best.F336W_UVIS_CHIP2'][index_hoy][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_hoy = hf.conv_mjy2vega(flux=data_hoy['best.F435W_ACS'][index_hoy][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_hoy = mag_v_hoy - mag_i_hoy
model_ub_hoy = mag_u_hoy - mag_b_hoy

mag_v_hng = hf.conv_mjy2vega(flux=data_hng['best.F555W_UVIS_CHIP2'][index_hng][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_hng = hf.conv_mjy2vega(flux=data_hng['best.F814W_UVIS_CHIP2'][index_hng][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_hng = hf.conv_mjy2vega(flux=data_hng['best.F336W_UVIS_CHIP2'][index_hng][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_hng = hf.conv_mjy2vega(flux=data_hng['best.F435W_ACS'][index_hng][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_hng = mag_v_hng - mag_i_hng
model_ub_hng = mag_u_hng - mag_b_hng

mag_v_hog = hf.conv_mjy2vega(flux=data_hog['best.F555W_UVIS_CHIP2'][index_hog][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_hog = hf.conv_mjy2vega(flux=data_hog['best.F814W_UVIS_CHIP2'][index_hog][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_hog = hf.conv_mjy2vega(flux=data_hog['best.F336W_UVIS_CHIP2'][index_hog][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_hog = hf.conv_mjy2vega(flux=data_hog['best.F435W_ACS'][index_hog][0], ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_hog = mag_v_hog - mag_i_hog
model_ub_hog = mag_u_hog - mag_b_hog

# ax_cc.scatter(model_vi_hny, model_ub_hny, marker='*', color='k', s=120)
# ax_cc.scatter(model_vi_hoy, model_ub_hoy, marker='*', color='k', s=120)
# ax_cc.scatter(model_vi_hng, model_ub_hng, marker='*', color='r', s=120)
# ax_cc.scatter(model_vi_hog, model_ub_hog, marker='*', color='r', s=120)


# load fit
hdu_best_model_hny = fits.open('data/fitted_data/1930_best_model_hst_ha_nircam.fits')
data_hny = hdu_best_model_hny[1].data
header_hny = hdu_best_model_hny[1].header
wavelength_spec_hny = data_hny["wavelength"] * 1e-3
surf_hny = 4.0 * np.pi * float(header_hny["universe.luminosity_distance"]) ** 2
fact_hny = 1e29 * 1e-3 * wavelength_spec_hny ** 2 / c / surf_hny
spectrum_hny = data_hny['Fnu']
wavelength_hny = data_hny['wavelength']

hdu_best_model_hoy = fits.open('data/fitted_data/1930_best_model_hst.fits')
data_hoy = hdu_best_model_hoy[1].data
header_hoy = hdu_best_model_hoy[1].header
wavelength_spec_hoy = data_hoy["wavelength"] * 1e-3
surf_hoy = 4.0 * np.pi * float(header_hoy["universe.luminosity_distance"]) ** 2
fact_hoy = 1e29 * 1e-3 * wavelength_spec_hoy ** 2 / c / surf_hoy
spectrum_hoy = data_hoy['Fnu']
wavelength_hoy = data_hoy['wavelength']

hdu_best_model_hng = fits.open('data/fitted_data/4460_best_model_hst_ha_nircam.fits')
data_hng = hdu_best_model_hng[1].data
header_hng = hdu_best_model_hng[1].header
wavelength_spec_hng = data_hng["wavelength"] * 1e-3
surf_hng = 4.0 * np.pi * float(header_hng["universe.luminosity_distance"]) ** 2
fact_hng = 1e29 * 1e-3 * wavelength_spec_hng ** 2 / c / surf_hng
spectrum_hng = data_hng['Fnu']
wavelength_hng = data_hng['wavelength']

hdu_best_model_hog = fits.open('data/fitted_data/4460_best_model.fits')
data_hog = hdu_best_model_hog[1].data
header_hog = hdu_best_model_hog[1].header
wavelength_spec_hog = data_hog["wavelength"] * 1e-3
surf_hog = 4.0 * np.pi * float(header_hog["universe.luminosity_distance"]) ** 2
fact_hog = 1e29 * 1e-3 * wavelength_spec_hog ** 2 / c / surf_hog
spectrum_hog = data_hog['Fnu']
wavelength_hog = data_hog['wavelength']


# get observations
hdu_y_obs = fits.open('data/fitted_data/observations_young.fits')
fluxes_y = hdu_y_obs[1].data[index_hny]

# get observations
hdu_g_obs = fits.open('data/fitted_data/observationsglobs.fits')
fluxes_g = hdu_g_obs[1].data[index_hng]

ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F275W', unit='nano'), fluxes_y['F275W_UVIS_CHIP2'][0] * 1e3, yerr=fluxes_y['F275W_UVIS_CHIP2_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F336W', unit='nano'), fluxes_y['F336W_UVIS_CHIP2'][0] * 1e3, yerr=fluxes_y['F336W_UVIS_CHIP2_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F435W', unit='nano'), fluxes_y['F435W_ACS'][0] * 1e3, yerr=fluxes_y['F435W_ACS_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F555W', unit='nano'), fluxes_y['F555W_UVIS_CHIP2'][0] * 1e3, yerr=fluxes_y['F555W_UVIS_CHIP2_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F658N', unit='nano'), fluxes_y['HST_ACS_WFC.F658N'][0] * 1e3, yerr=fluxes_y['HST_ACS_WFC.F658N_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F814W', unit='nano'), fluxes_y['F814W_UVIS_CHIP2'][0] * 1e3, yerr=fluxes_y['F814W_UVIS_CHIP2_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F200W', unit='nano'), fluxes_y['jwst.nircam.F200W'][0] * 1e3, yerr=fluxes_y['jwst.nircam.F200W_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F300M', unit='nano'), fluxes_y['jwst.nircam.F300M'][0] * 1e3, yerr=fluxes_y['jwst.nircam.F300M_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F335M', unit='nano'), fluxes_y['jwst.nircam.F335M'][0] * 1e3, yerr=fluxes_y['jwst.nircam.F335M_err'][0] * 1e3,
                fmt='o', c='k', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F360M', unit='nano'), fluxes_y['jwst.nircam.F360M'][0] * 1e3, yerr=fluxes_y['jwst.nircam.F360M_err'][0] * 1e3,
                fmt='o', c='k', ms=15)

ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F275W', unit='nano'), fluxes_g['F275W_UVIS_CHIP2'][0], yerr=fluxes_g['F275W_UVIS_CHIP2_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F336W', unit='nano'), fluxes_g['F336W_UVIS_CHIP2'][0], yerr=fluxes_g['F336W_UVIS_CHIP2_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F435W', unit='nano'), fluxes_g['F435W_ACS'][0], yerr=fluxes_g['F435W_ACS_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F555W', unit='nano'), fluxes_g['F555W_UVIS_CHIP2'][0], yerr=fluxes_g['F555W_UVIS_CHIP2_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F658N', unit='nano'), fluxes_g['HST_ACS_WFC.F658N'][0], yerr=fluxes_g['HST_ACS_WFC.F658N_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F814W', unit='nano'), fluxes_g['F814W_UVIS_CHIP2'][0], yerr=fluxes_g['F814W_UVIS_CHIP2_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F200W', unit='nano'), fluxes_g['jwst.nircam.F200W'][0], yerr=fluxes_g['jwst.nircam.F200W_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F300M', unit='nano'), fluxes_g['jwst.nircam.F300M'][0], yerr=fluxes_g['jwst.nircam.F300M_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F335M', unit='nano'), fluxes_g['jwst.nircam.F335M'][0], yerr=fluxes_g['jwst.nircam.F335M_err'][0],
                fmt='o', c='darkred', ms=15)
ax_sed_2.errorbar(phangs_photometry.get_band_wave(band='F360M', unit='nano'), fluxes_g['jwst.nircam.F360M'][0], yerr=fluxes_g['jwst.nircam.F360M_err'][0],
                fmt='o', c='darkred', ms=15)





ax_sed_2.plot(wavelength_hoy, spectrum_hoy * 1e3, linewidth=3, color='k', linestyle='--', label='Stellar attenuated')
ax_sed_2.plot(wavelength_hny, spectrum_hny * 1e3, linewidth=3, color='k', label='Stellar attenuated')

ax_sed_2.plot(wavelength_hog, spectrum_hog, linewidth=3, color='darkred', linestyle='--', label='Stellar attenuated')
ax_sed_2.plot(wavelength_hng, spectrum_hng, linewidth=3, color='darkred', label='Stellar attenuated')




phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr', band_list=band_list)

# hst_h_alpha_r = plotting_tools.reproject_image(data=cutout_dict['F658N_img_cutout'].data,
#                                                wcs=cutout_dict['F658N_img_cutout'].wcs,
#                                                new_wcs=new_wcs,
#                                                new_shape=new_shape)
# hst_h_alpha_g = plotting_tools.reproject_image(data=cutout_dict['F555W_img_cutout'].data,
#                                                wcs=cutout_dict['F555W_img_cutout'].wcs,
#                                                new_wcs=new_wcs,
#                                                new_shape=new_shape)
# hst_h_alpha_b = plotting_tools.reproject_image(data=cutout_dict['F435W_img_cutout'].data,
#                                                wcs=cutout_dict['F435W_img_cutout'].wcs,
#                                                new_wcs=new_wcs,
#                                                new_shape=new_shape)

ax_hst_1.imshow(rgb_hst_h_alpha_image_1)
ax_hst_1.axis('off')
ax_hst_2.imshow(rgb_hst_h_alpha_image_2)
ax_hst_2.axis('off')

ax_nircam_1.imshow(rgb_nircam_image_1)
ax_nircam_1.axis('off')
ax_nircam_2.imshow(rgb_nircam_image_2)
ax_nircam_2.axis('off')

ax_sed_1.annotate(r'Ly$\alpha$', xy=(121.57, 2e-1), xycoords='data',
                    xytext=(170.57, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.annotate(r'  H$\alpha$', xy=(656.5, 2e-1), xycoords='data',
                    xytext=(856.5, 3e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.text(856.5, 2e-1, 'F657N', fontsize=fontsize-5)
ax_sed_1.annotate(r' Pa$\alpha$', xy=(1875, 9e-2), xycoords='data',
                    xytext=(2575, 1.5e-1), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sed_1.text(2575, 9e-2, 'F187N', fontsize=fontsize-5)

# ax_sed_2.annotate(r'Ly$\alpha$', xy=(121.57, 2e-1), xycoords='data',
#                     xytext=(170.57, 3e-1), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_sed_2.annotate(r'  H$\alpha$', xy=(656.5, 2e-1), xycoords='data',
#                     xytext=(856.5, 3e-1), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_sed_2.text(856.5, 2e-1, 'F657N', fontsize=fontsize-5)
# ax_sed_2.annotate(r' Pa$\alpha$', xy=(1875, 4e-2), xycoords='data',
#                     xytext=(2575, 7e-2), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_sed_2.text(2575, 4.7e-2, 'F187N', fontsize=fontsize-5)
#

ax_sed_2.text(85, 9.9e4, r'HST only', fontsize=fontsize-3)
ax_sed_2.text(85, 3e4, r'age=%i Myr, M$_{*}$=%i$\times$10$^{2}$M$_{\odot}$, A$_{\rm V}$=%.1fmag' % (age_hoy, int(round(mstar_hoy/100)), av_hoy), fontsize=fontsize-3)

ax_sed_2.text(85, 3e3, r'HST + H$\alpha$ + NIRCAM', fontsize=fontsize-3)
ax_sed_2.text(85, 1e3, r'age=%i Myr, M$_{*}$=%i$\times$10$^{2}$M$_{\odot}$, A$_{\rm V}$=%.1fmag' % (age_hny, int(round(mstar_hny/100)), av_hny), fontsize=fontsize-3)

ax_sed_2.text(95, 1.5e-1, r'HST only', color='darkred', fontsize=fontsize-3)
ax_sed_2.text(95, 4e-2, r'age=%i Myr, M$_{*}$=%i$\times$10$^{3}$M$_{\odot}$, A$_{\rm V}$=%.1fmag' % (age_hog, int(round(mstar_hog/1000)), av_hog), color='darkred', fontsize=fontsize-3)
ax_sed_2.text(95, 6e-3, r'HST + H$\alpha$ + NIRCAM', color='darkred', fontsize=fontsize-3)
ax_sed_2.text(95, 2e-3, r'age=%i Gyr, M$_{*}$=%i$\times$10$^{4}$M$_{\odot}$, A$_{\rm V}$=%.1fmag' % (round(age_hng/1000), int(round(mstar_hng/10000)), av_hng), color='darkred', fontsize=fontsize-3)



# plot observation filters
cigale_wrapper_obj.plot_hst_filters(ax=ax_sed_1, fontsize=fontsize-5, color='k')
cigale_wrapper_obj.plot_hst_filters(ax=ax_sed_2, fontsize=fontsize-5, color='k')
# cigale_wrapper_obj.plot_hst_nircam_filters(ax=ax_sed_1, fontsize=fontsize-5,)


ax_sed_2.text(130, 1e1, r'$\times 10^{3}$', fontsize=fontsize)


ax_sed_1.axvspan(650, 662, alpha=0.3, color='k')
ax_sed_2.axvspan(650, 662, alpha=0.3, color='k')


ax_sed_1.axvspan(1.331*1000, 1.668*1000, alpha=0.3, color='r')
ax_sed_1.axvspan(1.863*1000, 1.885*1000, alpha=0.3, color='r')
ax_sed_1.axvspan(2.831*1000, 3.157*1000, alpha=0.3, color='r')
ax_sed_1.axvspan(3.177*1000, 3.537*1000, alpha=0.3, color='r')



ax_sed_2.axvspan(1.331*1000, 1.668*1000, alpha=0.3, color='r')
ax_sed_2.axvspan(1.863*1000, 1.885*1000, alpha=0.3, color='r')
ax_sed_2.axvspan(2.831*1000, 3.157*1000, alpha=0.3, color='r')
ax_sed_2.axvspan(3.177*1000, 3.537*1000, alpha=0.3, color='r')

ax_sed_1.text(320, 8e-1, r'HST WFC3', color='k', fontsize=fontsize)
ax_sed_1.text(1500, 8e-1, r'NIRCAM', color='red', fontsize=fontsize)


ax_sed_1.text(150, 1.5e-2, r'5 Myr', color='r', fontsize=fontsize)
ax_sed_1.text(150, 1.5e-3, r'10 Myr', color='b', fontsize=fontsize)
ax_sed_1.text(150, 3e-4, r'100 Myr', color='m', fontsize=fontsize)
ax_sed_1.text(200, 2e-5, r'1 Gyr', color='g', fontsize=fontsize)
ax_sed_1.text(500, 7e-6, r'10 Gyr', color='orange', fontsize=fontsize)

ax_sed_1.text(300, 2e-6, r'Z=Z$_{\odot}$, M$_{*}$=10$^{4}$ M$_{\odot}$, D=15 Mpc', fontsize=fontsize)
# ax_sed_2.text(300, 5e-6, r'Z=Z$_{\odot}$, M$_{*}$=10$^{4}$ M$_{\odot}$, D=15 Mpc', fontsize=fontsize)
# ax_sed_2.text(300, 2e-6, r'Age=5 Myr, A$_{\rm V}$=0-2 mag', fontsize=fontsize)


# ax_sed_2.text(0.5)
# # ax_sed_2.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
# ax_sed_2.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
# ax_sed_2.set_xlim(200 * 1e-3, 4)
# ax_sed_2.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)


ax_sed_1.set_xlim(0.08*1e3, 4 * 1e3)
ax_sed_2.set_xlim(0.08*1e3, 4 * 1e3)

ax_sed_1.set_ylim(7e-7, 1.5e0)
ax_sed_2.set_ylim(3e-8, 8e5)

ax_sed_1.set_xscale('log')
ax_sed_1.set_yscale('log')
ax_sed_2.set_xscale('log')
ax_sed_2.set_yscale('log')

ax_sed_1.yaxis.tick_right()
ax_sed_1.yaxis.set_label_position("right")
ax_sed_1.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed_2.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed_1.set_ylabel(r'F$_{\nu}$ [mJy]', labelpad=-1, fontsize=fontsize)
# ax_sed_2.set_yticklabels([])

ax_sed_1.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
ax_sed_2.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
# ax_sed_1.legend(loc='upper left', fontsize=fontsize-6, ncol=3, columnspacing=1, handlelength=1, handletextpad=0.6)


plt.savefig('plot_output/sed_color_color_prop.png')
plt.savefig('plot_output/sed_color_color_prop.pdf')

exit()