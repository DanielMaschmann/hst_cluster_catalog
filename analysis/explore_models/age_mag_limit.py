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
from matplotlib import patheffects


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)

# print(dust_tools.extinction_tools.ExtinctionTools.ebv2av(ebv=0.2, r_v=3.1))
#
# exit()

def get_abs_v_mag(age, mass_m_sun, dist_mpc, ebv=0.0, id=0):

    cigale_wrapper_obj.sed_modules_params['dustext']['E_BV'] = [ebv]
    cigale_wrapper_obj.sed_modules_params['sfh2exp']['age'] = [age]
    parameter_list = ['attenuation.E_BV']
    # run cigale
    cigale_wrapper_obj.create_cigale_model()
    # load model into constructor
    cigale_wrapper_obj.load_cigale_model_block()

    model_blocks = fits.open('out/models-block-%i.fits' % id)[1].data

    # get v-band flux in mJy
    flux_v_band = model_blocks['F555W_UVIS_CHIP2']# * u.mJy

    # scale the flux to the wanted solar mass
    flux_v_band *= (mass_m_sun * u.M_sun) / (model_blocks['stellar.m_star'] * u.M_sun)
    # get the distance in m (this is normally at 10 pc)
    distance_in_m = model_blocks['universe.luminosity_distance'] * u.m

    # scale the distance
    dist_scale_factor = distance_in_m ** 2 / (((dist_mpc * u.Mpc).to(u.m))**2)
    # rescale the flux to the wanted distance
    flux_v_band *= dist_scale_factor

    mag_v_band = hf.conv_mjy2vega(flux=flux_v_band,
                                 ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                                 vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))

    abs_mag_v_band = hf.conv_mag2abs_mag(mag=mag_v_band, dist=dist_mpc)

    # get absolute magnitude
    return abs_mag_v_band


        #
        # print(mod.names)
        # print(model_blocks.names)
        # print(model_blocks['stellar.m_star'])
        # print('dist ', model_blocks['universe.luminosity_distance'])
        # print(flux_v_band)
        # # flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
        #
        # cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='out/%i_best_model.fits'%id,
        #                                      cluster_mass=mass, distance_Mpc=distance_Mpc,
        #                                      color=color, linestyle=linestyle, linewidth=linewidth)
        #



# crate wrapper class object
cigale_wrapper_obj = cw.CigaleModelWrapper()

age_list = [1,3,5,7,10,
            10,  30, 50, 70,
            100, 300, 500, 700,
            1000, 3000, 5000, 7000,
            10000, 13750]
abs_v_band_mag_list = []
for age in age_list:
    abs_v_band_mag = get_abs_v_mag(age=age, mass_m_sun=1e5, dist_mpc=10, ebv=0.0)
    abs_v_band_mag_list.append(abs_v_band_mag)


plt.scatter(age_list, abs_v_band_mag_list)
plt.plot(age_list, abs_v_band_mag_list)
plt.xscale('log')
plt.xlabel('age [Myr]')
plt.ylabel(r'M$_{\rm V}$ [mag]')
# plt.show()
plt.savefig('plot_output/m_v_age.png')
exit()



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


figure = plt.figure(figsize=(30, 10))
fontsize = 26
ax_sed = figure.add_axes([0.40, 0.08, 0.545, 0.91])
ax_cbar = figure.add_axes([0.95, 0.15, 0.015, 0.77])

ax_cc = figure.add_axes([0.05, 0.08, 0.30, 0.91])


sol_x_lim_ub = (0.52, 0.81)
sol_y_lim_ub = (-0.96, -1.22)
scatter_size = 80

ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-0.9, 1.8)

index_1_gyr = np.where(age_mod_sol50 == 500)




ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_cc.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
# ax_cc.set_xticklabels([])



# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 10 * u.Mpc

def plot_age_only(ax, age, mass, color='k', linestyle='-', linewidth=3):

    id = np.where(age_mod_sol == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/no_dust/sol_met_50/out/%i_best_model.fits'%id,
                                            cluster_mass=mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)
def plot_age_only_sol50(ax, age, mass, color='k', linestyle='-', linewidth=3):

    id = np.where(age_mod_sol == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/no_dust/sol_met/out/%i_best_model.fits'%id,
                                            cluster_mass=mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


# dust_list = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dust_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
# av_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05,
#            1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]
av_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
# av_list = [0, 2.2]


# dust_list = [0, 1.05]

cmap = matplotlib.cm.get_cmap('autumn_r')

# rgba = cmap(0.5)
# norm = matplotlib.colors.Normalize(vmin=1.0, vmax=10000)
norm = matplotlib.colors.Normalize(vmin=0, vmax=av_list[-1])

# set star cluster mass to scale the model SEDs
cluster_mass_young = 3E4 * u.Msun
cluster_mass_old_1 = 1E5 * u.Msun
cluster_mass_old_2 = 1E6 * u.Msun

for av in av_list:
    print(av)
    color = cmap(norm(av))
    ebv = dust_tools.extinction_tools.ExtinctionTools.av2ebv(av=av)
    plot_age_ebv(ax=ax_sed, age=5, mass=cluster_mass_young, ebv=ebv, color=color, linestyle='-', linewidth=3)


plot_age_only(ax=ax_sed, age=100, mass=cluster_mass_old_1, color='k', linestyle='--', linewidth=4)
plot_age_only(ax=ax_sed, age=10000, mass=cluster_mass_old_2, color='k', linestyle=':', linewidth=4)


ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap, norm=norm, extend='neither', ticks=[0., 0.5, 1.0, 1.5, 2.0, 2.5])
ax_cbar.set_ylabel(r'A$_{\rm V}$ [mag]', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)


ax_cc.text(0.9, -1.75, r'A$_{\rm V}$', fontsize=fontsize+3, rotation=-30)
ax_cc.text(0.5, -2.0, r'0', fontsize=fontsize+3, rotation=-30)
ax_cc.text(1.3, -1.5, r'2', fontsize=fontsize+3, rotation=-30)

# add reddening vector
# intrinsic colors
# vi_int = 0.2
# ub_int = -1.9
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

lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.95, zorder=0)

lc.set_array(av_value[:-1])
lc.set_linewidth(20)
line = ax_cc.add_collection(lc)

ax_cc.annotate("", xy=(vi_int+max_color_ext_vi_arr, ub_int+max_color_ext_ub_arr), xytext=(x_av[-1], y_av[-1]),
               arrowprops=dict(arrowstyle="-|>, head_width=2.5, head_length=2.5", linewidth=1,
                               color=cmap(norm(av_value[-1]))), zorder=10)



# ax_sed.plot([], [], color=cmap(norm(0)), linestyle='-', linewidth=3, label=r'Age = 5 Myr, Z=Z$_{\odot}$, M$_{*}$ = 3 $\times$ 10$^{4}$ M$_{\odot}$, E(B-V) = 0-1.05')
ax_sed.plot([], [], color='gray', linestyle='-', linewidth=3, label=r'Age = 5 Myr, Z=Z$_{\odot}$, M$_{*}$ = 3 $\times$ 10$^{4}$ M$_{\odot}$,A$_{\rm V}$ = 0-2.2')
ax_sed.plot([], [], color='k', linestyle='--', linewidth=4, label=r'Age = 100 Myr, Z=Z$_{\odot}$, M$_{*}$ = 10$^{5}$ M$_{\odot}$, A$_{\rm V}$ = 0')
ax_sed.plot([], [], color='k', linestyle=':', linewidth=4, label=r'Age = 10 Gyr, Z=Z$_{\odot}$/50, M$_{*}$ = 10$^{6}$ M$_{\odot}$, A$_{\rm V}$ = 0')
ax_sed.legend(loc='upper left', fontsize=fontsize, frameon=False)
# ax_sed.text(500, 2, r'Z=Z$_{\odot}$, M$_{*}$ = 10$^{5}$ M$_{\odot}$, D = 10 Mpc', fontsize=fontsize)
ax_sed.text(700, 0.8, r'D = 10 Mpc', fontsize=fontsize)


# plot observation filters
cigale_wrapper_obj.plot_hst_filters(ax=ax_sed, fontsize=fontsize-5, text_hight=5e-4, color='k', text=True)

ax_sed.set_xlim(230, 0.9 * 1e3)
ax_sed.set_ylim(3e-4, 1.5e0)

ax_sed.set_xscale('log')
ax_sed.set_yscale('log')

ax_sed.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed.set_ylabel(r'F$_{\nu}$ [mJy]', fontsize=fontsize)
ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)


plt.savefig('plot_output/sed_color_color_dust.png')
plt.savefig('plot_output/sed_color_color_dust.pdf')
