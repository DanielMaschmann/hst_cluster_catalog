import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from photometry_tools import helper_func as hf
from photometry_tools.data_access import CatalogAccess
from cigale_helper import cigale_wrapper as cw


def scale_reddening_vector(cluster_ebv, x_comp, y_comp):

    """This function scales the reddening vector for the given E(B-V) value of a star cluster"""
    comparison_scale = 1.0 / 3.2
    scale_factor = cluster_ebv / comparison_scale

    return x_comp * scale_factor, y_comp * scale_factor


# set star cluster mass to scale the model SEDs
cluster_mass = 1E5 * u.Msun
# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 10 * u.Mpc
# crate wrapper class object
cigale_wrapper_obj = cw.CigaleModelWrapper()


# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/out/models-block-0.fits')
data_mod = hdu_a[1].data
age_mod = data_mod['sfh.age']
flux_f555w = data_mod['F555W_UVIS_CHIP2']
flux_f814w = data_mod['F814W_UVIS_CHIP2']
flux_f336w = data_mod['F336W_UVIS_CHIP2']
flux_f438w = data_mod['F438W_UVIS_CHIP2']

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)


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



figure = plt.figure(figsize=(30, 10))
fontsize = 26
ax_models = figure.add_axes([0.38, 0.08, 0.61, 0.91])
ax_explain = figure.add_axes([0.05, 0.08, 0.28, 0.91])
# plot observation filters
cigale_wrapper_obj.plot_hst_filters(ax=ax_models, fontsize=fontsize-5)



ax_explain.plot(model_vi, model_ub, color='red', linewidth=3)
x_de_red, y_dered = scale_reddening_vector(0.5, 0.43, 0.3)
ax_explain.annotate('', xy=(1.7 - x_de_red, -1.5 - y_dered), xycoords='data', xytext=(1.7, -1.5), textcoords='data',
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=3, ls='-'))
ax_explain.text(1.1, -1.6, 'E(B-V) = 0.5', fontsize=fontsize, rotation=-29)


vi_1_my = model_vi[age_mod == 1][0]
ub_1_my = model_ub[age_mod == 1][0]

vi_4_my = model_vi[age_mod == 4][0]
ub_4_my = model_ub[age_mod == 4][0]

vi_5_my = model_vi[age_mod == 5][0]
ub_5_my = model_ub[age_mod == 5][0]

vi_10_my = model_vi[age_mod == 10][0]
ub_10_my = model_ub[age_mod == 10][0]

vi_100_my = model_vi[age_mod == 102][0]
ub_100_my = model_ub[age_mod == 102][0]

vi_1000_my = model_vi[age_mod == 1028][0]
ub_1000_my = model_ub[age_mod == 1028][0]

vi_10000_my = model_vi[age_mod == 10308][0]
ub_10000_my = model_ub[age_mod == 10308][0]

ax_explain.scatter(vi_1_my, ub_1_my, c='k', zorder=10)
ax_explain.scatter(vi_4_my, ub_4_my, c='k', zorder=10)
ax_explain.scatter(vi_5_my, ub_5_my, c='k', zorder=10)
ax_explain.scatter(vi_10_my, ub_10_my, c='k', zorder=10)
ax_explain.scatter(vi_100_my, ub_100_my, c='k', zorder=10)
ax_explain.scatter(vi_1000_my, ub_1000_my, c='k', zorder=10)
ax_explain.scatter(vi_10000_my, ub_10000_my, c='k', zorder=10)

ax_explain.annotate('1 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my + 0.5, ub_1_my), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my + 0.5, ub_10_my), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my, ub_1000_my - 0.5), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax_explain.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
                    xytext=(vi_10000_my - 0.5, ub_10000_my + 0.5), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax_explain.set_ylim(1.25, -2.2)
ax_explain.set_xlim(-1.0, 1.8)
# ax_explain.set_xticklabels([])
ax_explain.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_explain.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_explain.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


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

