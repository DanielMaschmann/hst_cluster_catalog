import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from photometry_tools import helper_func as hf
from photometry_tools.data_access import CatalogAccess
from cigale_helper import cigale_wrapper as cw
from matplotlib.patches import ConnectionPatch

def scale_reddening_vector(cluster_ebv, x_comp, y_comp):

    """This function scales the reddening vector for the given E(B-V) value of a star cluster"""
    comparison_scale = 1.0 / 3.2
    scale_factor = cluster_ebv / comparison_scale

    return x_comp * scale_factor, y_comp * scale_factor


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)



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



figure = plt.figure(figsize=(30, 13))
fontsize = 28
ax_sol = figure.add_axes([0.06, 0.08, 0.45, 0.85])
ax_sol50 = figure.add_axes([0.53, 0.08, 0.45, 0.85])
ax_sol_zoom = figure.add_axes([0.37, 0.6, 0.20, 0.39])

sol_x_lim = (0.52, 0.81)
sol_y_lim = (-0.96, -1.22)

scatter_size = 80

ax_sol.set_ylim(1.25, -2.2)
ax_sol.set_xlim(-0.9, 1.8)
ax_sol50.set_ylim(1.25, -2.2)
ax_sol50.set_xlim(-0.9, 1.4)

ax_sol_zoom.set_ylim(sol_y_lim)
ax_sol_zoom.set_xlim(sol_x_lim)

ax_sol.plot([sol_x_lim[0], sol_x_lim[0]], [sol_y_lim[0], sol_y_lim[1]], color='k', linewidth=2)
ax_sol.plot([sol_x_lim[1], sol_x_lim[1]], [sol_y_lim[0], sol_y_lim[1]], color='k', linewidth=2)
ax_sol.plot([sol_x_lim[0], sol_x_lim[1]], [sol_y_lim[0], sol_y_lim[0]], color='k', linewidth=2)
ax_sol.plot([sol_x_lim[0], sol_x_lim[1]], [sol_y_lim[1], sol_y_lim[1]], color='k', linewidth=2)

con_spec_1 = ConnectionPatch(
    xyA=(sol_x_lim[0], sol_y_lim[1]), coordsA=ax_sol.transData,
    xyB=(ax_sol_zoom.get_xlim()[0], ax_sol_zoom.get_ylim()[1]), coordsB=ax_sol_zoom.transData,
    linestyle="--", linewidth=2, color='k')
figure.add_artist(con_spec_1)
con_spec_2 = ConnectionPatch(
    xyA=(sol_x_lim[0], sol_y_lim[0]), coordsA=ax_sol.transData,
    xyB=(ax_sol_zoom.get_xlim()[0], ax_sol_zoom.get_ylim()[0]), coordsB=ax_sol_zoom.transData,
    linestyle="--", linewidth=2, color='k')
figure.add_artist(con_spec_2)


ax_sol.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, label='BC03, Z=Z$_{\odot}$')
ax_sol.plot(model_vi_sol50, model_ub_sol50, color='dimgray', linewidth=2, linestyle='--', alpha=0.5, label='BC03, Z=Z$_{\odot}$/50')
ax_sol50.plot(model_vi_sol50, model_ub_sol50, color='dimgray', linewidth=3, label='BC03, Z=Z$_{\odot}/50$')
ax_sol50.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='--', alpha=0.5, label='BC03, Z=Z$_{\odot}$')

ax_sol_zoom.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3)


vi_1_my = model_vi_sol[age_mod_sol == 1][0]
ub_1_my = model_ub_sol[age_mod_sol == 1][0]

vi_4_my = model_vi_sol[age_mod_sol == 4][0]
ub_4_my = model_ub_sol[age_mod_sol == 4][0]

vi_5_my = model_vi_sol[age_mod_sol == 5][0]
ub_5_my = model_ub_sol[age_mod_sol == 5][0]

vi_10_my = model_vi_sol[age_mod_sol == 10][0]
ub_10_my = model_ub_sol[age_mod_sol == 10][0]

vi_100_my = model_vi_sol[age_mod_sol == 102][0]
ub_100_my = model_ub_sol[age_mod_sol == 102][0]

vi_1000_my = model_vi_sol[age_mod_sol == 1028][0]
ub_1000_my = model_ub_sol[age_mod_sol == 1028][0]

vi_10000_my = model_vi_sol[age_mod_sol == 10308][0]
ub_10000_my = model_ub_sol[age_mod_sol == 10308][0]

ax_sol.scatter(vi_1_my, ub_1_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_4_my, ub_4_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_5_my, ub_5_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_10_my, ub_10_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_100_my, ub_100_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_1000_my, ub_1000_my, c='k', s=scatter_size, zorder=10)
ax_sol.scatter(vi_10000_my, ub_10000_my, c='k', s=scatter_size, zorder=10)

ax_sol.annotate('1 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my - 0.1, ub_1_my - 0.3), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my - 0.5, ub_10_my - 0.4), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my, ub_1000_my - 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my - 0.5, ub_10000_my + 0.5), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


vi_12_my_zoom = model_vi_sol[age_mod_sol == 12][0]
ub_12_my_zoom = model_ub_sol[age_mod_sol == 12][0]

vi_14_my_zoom = model_vi_sol[age_mod_sol == 14][0]
ub_14_my_zoom = model_ub_sol[age_mod_sol == 14][0]

vi_18_my_zoom = model_vi_sol[age_mod_sol == 18][0]
ub_18_my_zoom = model_ub_sol[age_mod_sol == 18][0]

vi_20_my_zoom = model_vi_sol[age_mod_sol == 20][0]
ub_20_my_zoom = model_ub_sol[age_mod_sol == 20][0]

vi_25_my_zoom = model_vi_sol[age_mod_sol == 25][0]
ub_25_my_zoom = model_ub_sol[age_mod_sol == 25][0]

vi_32_my_zoom = model_vi_sol[age_mod_sol == 32][0]
ub_32_my_zoom = model_ub_sol[age_mod_sol == 32][0]


ax_sol_zoom.scatter(vi_12_my_zoom, ub_12_my_zoom, c='k', s=scatter_size, zorder=10)
ax_sol_zoom.scatter(vi_14_my_zoom, ub_14_my_zoom, c='k', s=scatter_size, zorder=10)
ax_sol_zoom.scatter(vi_18_my_zoom, ub_18_my_zoom, c='k', s=scatter_size, zorder=10)
ax_sol_zoom.scatter(vi_20_my_zoom, ub_20_my_zoom, c='k', s=scatter_size, zorder=10)
ax_sol_zoom.scatter(vi_25_my_zoom, ub_25_my_zoom, c='k', s=scatter_size, zorder=10)
ax_sol_zoom.scatter(vi_32_my_zoom, ub_32_my_zoom, c='k', s=scatter_size, zorder=10)


ax_sol_zoom.annotate('12 Myr', xy=(vi_12_my_zoom, ub_12_my_zoom), xycoords='data', xytext=(vi_12_my_zoom + 0.05, ub_12_my_zoom + 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol_zoom.annotate('14 Myr', xy=(vi_14_my_zoom, ub_14_my_zoom), xycoords='data', xytext=(vi_14_my_zoom - 0.05, ub_14_my_zoom - 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol_zoom.annotate('18 Myr', xy=(vi_18_my_zoom, ub_18_my_zoom), xycoords='data', xytext=(vi_18_my_zoom + 0.03, ub_18_my_zoom - 0.01), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol_zoom.annotate('20 Myr', xy=(vi_20_my_zoom, ub_20_my_zoom), xycoords='data', xytext=(vi_20_my_zoom - 0.06, ub_20_my_zoom - 0.03), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol_zoom.annotate('25 Myr', xy=(vi_25_my_zoom, ub_25_my_zoom), xycoords='data', xytext=(vi_25_my_zoom + 0.06, ub_25_my_zoom + 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol_zoom.annotate('32 Myr', xy=(vi_32_my_zoom, ub_32_my_zoom), xycoords='data', xytext=(vi_32_my_zoom - 0.01, ub_32_my_zoom - 0.07), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


vi_1_my = model_vi_sol50[age_mod_sol50 == 1][0]
ub_1_my = model_ub_sol50[age_mod_sol50 == 1][0]

vi_4_my = model_vi_sol50[age_mod_sol50 == 4][0]
ub_4_my = model_ub_sol50[age_mod_sol50 == 4][0]

vi_5_my = model_vi_sol50[age_mod_sol50 == 5][0]
ub_5_my = model_ub_sol50[age_mod_sol50 == 5][0]

vi_10_my = model_vi_sol50[age_mod_sol50 == 10][0]
ub_10_my = model_ub_sol50[age_mod_sol50 == 10][0]

vi_100_my = model_vi_sol50[age_mod_sol50 == 102][0]
ub_100_my = model_ub_sol50[age_mod_sol50 == 102][0]

vi_1000_my = model_vi_sol50[age_mod_sol50 == 1028][0]
ub_1000_my = model_ub_sol50[age_mod_sol50 == 1028][0]

vi_10000_my = model_vi_sol50[age_mod_sol50 == 10308][0]
ub_10000_my = model_ub_sol50[age_mod_sol50 == 10308][0]

ax_sol50.scatter(vi_1_my, ub_1_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_4_my, ub_4_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_5_my, ub_5_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_10_my, ub_10_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_100_my, ub_100_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_1000_my, ub_1000_my, c='k', s=scatter_size, zorder=10)
ax_sol50.scatter(vi_10000_my, ub_10000_my, c='k', s=scatter_size, zorder=10)

ax_sol50.annotate('1 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my + 0.3, ub_1_my - 0.3), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my + 0.3, ub_4_my - 0.3), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my + 0.0, ub_5_my - 0.3), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my - 0.0, ub_10_my + 0.6), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my, ub_1000_my - 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_sol50.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my - 0.5, ub_10000_my + 0.5), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


ax_sol.set_title(r'Z = Z$_{\odot}$', fontsize=fontsize + 4)
ax_sol50.set_title(r'Z = Z$_{\odot}/50$', fontsize=fontsize + 4)

ax_sol.legend(frameon=False, loc=3, fontsize=fontsize)
ax_sol50.legend(frameon=False, loc=3, fontsize=fontsize)
ax_sol.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_sol50.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_sol_zoom.yaxis.tick_right()
ax_sol_zoom.tick_params(axis='both', which='both', width=1.5, length=4, left=True, right=True, top=True, direction='in', labelsize=fontsize)
ax_sol.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_sol.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_sol50.set_yticklabels([])
ax_sol50.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

# plt.show()

plt.savefig('plot_output/explain_model_1.png')


exit()



def plot_age_ebv(age=1, ebv=0.0, color='k', linestyle='-', linewidth=3):
    """

    :param age:
    :param ebv:
    :param color:
    :param linestyle:
    :param linewidth:
    """
    cigale_wrapper_obj.sed_modules_params['dustext']['E_BV'] = [ebv]
    cigale_wrapper_obj.sed_modules_params['sfh2exp']['age'] = [age]
    parameter_list = ['attenuation.E_BV']
    # run cigale
    cigale_wrapper_obj.create_cigale_model()
    # load model into constructor
    cigale_wrapper_obj.load_cigale_model_block()
    # plot cigale model
    for id_index, id in zip(range(len(cigale_wrapper_obj.model_table_dict['id']['value'])),
                            cigale_wrapper_obj.model_table_dict['id']['value']):
        cigale_wrapper_obj.plot_cigale_model(ax=ax_models, model_file_name='out/%i_best_model.fits'%id,
                                             cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


plot_age_ebv(age=1, ebv=0.0, color='tab:green', linestyle='-')
plot_age_ebv(age=1, ebv=1.0, color='tab:green', linestyle='--')
plot_age_ebv(age=1, ebv=2.0, color='tab:green', linestyle=':')

plot_age_ebv(age=10, ebv=0.0, color='tab:blue', linestyle='-')
plot_age_ebv(age=10, ebv=1.0, color='tab:blue', linestyle='--')
plot_age_ebv(age=10, ebv=2.0, color='tab:blue', linestyle=':')

plot_age_ebv(age=100, ebv=0.0, color='tab:red', linestyle='-')
plot_age_ebv(age=100, ebv=1.0, color='tab:red', linestyle='--')
plot_age_ebv(age=100, ebv=2.0, color='tab:red', linestyle=':')

plot_age_ebv(age=10000, ebv=0.0, color='tab:orange', linestyle='-')
plot_age_ebv(age=10000, ebv=1.0, color='tab:orange', linestyle='--')
plot_age_ebv(age=10000, ebv=2.0, color='tab:orange', linestyle=':')

ax_models.plot([], [], color='k', linewidth=3, linestyle='-', label='E(B-V) = 0')
ax_models.plot([], [], color='k', linewidth=3, linestyle='--', label='E(B-V) = 1')
ax_models.plot([], [], color='k', linewidth=3, linestyle=':', label='E(B-V) = 2')

ax_models.plot([], [], color='tab:green', linewidth=3, linestyle='-', label='Age = 1 My')
ax_models.plot([], [], color='tab:blue', linewidth=3, linestyle='-', label='Age = 10 My')
ax_models.plot([], [], color='tab:red', linewidth=3, linestyle='-', label='Age = 100 My')
ax_models.plot([], [], color='tab:orange', linewidth=3, linestyle='-', label='Age = 10 Gy')


ax_models.text(500, 100, r'Z=0.02 dex, M$_{*}$ = 10$^{5}$ M$_{\odot}$, D = 10 Mpc', fontsize=fontsize)

ax_models.set_xlim(230, 0.9* 1e3)
ax_models.set_ylim(7e-8, 1.5e3)

ax_models.set_xscale('log')
ax_models.set_yscale('log')

ax_models.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_models.set_ylabel(r'F$_{\nu}$ [mJy]', fontsize=fontsize)
ax_models.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
ax_models.legend(loc='upper left', fontsize=fontsize-6, ncol=3, columnspacing=1, handlelength=1, handletextpad=0.6)

# plt.figtext(0.17, 0.035, 'HST WFC3', fontsize=fontsize-4)


plt.savefig('plot_output/model_choice.png')
plt.savefig('plot_output/model_choice.pdf')

