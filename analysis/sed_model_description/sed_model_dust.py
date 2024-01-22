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


nuvb_label_dict = {
    1: {'offsets': [0.25, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.05, 0.1], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [0.1, 0.0], 'ha': 'left', 'va': 'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}

nuvb_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.7, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [+0.05, 0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.1, 0.4], 'label': '13.8 Gyr', 'ha': 'right', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.4], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, 0.2], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}

def display_models(ax, y_color='nuvb',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-', color_arrow_sol='darkcyan', arrow_linestyle_sol='--',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-', color_arrow_sol50='darkviolet', arrow_linestyle_sol50='--',
                   label_sol=None, label_sol50=None):

    y_model_sol = globals()['model_%s_sol' % y_color]
    y_model_sol50 = globals()['model_%s_sol50' % y_color]

    ax.plot(model_vi_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)
    ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)

    ax.plot(model_vi_sol50[age_mod_sol50 <= age_cut_sol50], y_model_sol50[age_mod_sol50 <= age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle='--', zorder=10)


    if age_dots_sol is None:
        age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
    for age in age_dots_sol:
        ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='b', s=80, zorder=20)

    if age_dots_sol50 is None:
        age_dots_sol50 = [500, 1000, 13750]
    for age in age_dots_sol50:
        ax.scatter(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age], color='tab:pink', s=80, zorder=20)

    if age_labels:
        label_dict = globals()['%s_label_dict' % y_color]
        pe = [patheffects.withStroke(linewidth=3, foreground="w")]
        for age in label_dict.keys():

            ax.text(model_vi_sol[age_mod_sol == age]+label_dict[age]['offsets'][0],
                    y_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][1],
                    label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize,
                    path_effects=pe)

        annotation_dict = globals()['%s_annotation_dict' % y_color]
        for age in annotation_dict.keys():

            # txt_sol = ax.annotate(' ', #annotation_dict[age]['label'],
            #             xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
            #             xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
            #                     y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
            #             fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
            #             ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
            #                   arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0,edgecolor="none",
            #                                   facecolor=color_arrow_sol, lw=3, ls='-'))
            # txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
            #                                       patheffects.Normal()])
            txt_sol = ax.annotate(' ',
                                  xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color=color_arrow_sol, lw=3, ls=arrow_linestyle_sol))
            txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            txt_sol50 = ax.annotate(' ',
                        xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color=color_arrow_sol50, lw=3, ls=arrow_linestyle_sol50),
                        path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
            txt_sol50.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            ax.text(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40, path_effects=pe)



age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_bi_sol = np.load('../color_color/data_output/model_bi_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('../color_color/data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_bi_sol50 = np.load('../color_color/data_output/model_bi_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')

def plot_age_ebv(ax, age, mass, ebv=0.0, color='k', linestyle='-', linewidth=3):
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
        cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='out/%i_best_model.fits'%id,
                                             cluster_mass=mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)



# crate wrapper class object
cigale_wrapper_obj = cw.CigaleModelWrapper()

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)

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

# ax_cc.plot(model_vi_sol, model_ub_sol, color='tab:cyan', linewidth=3, label='BC03, Z=Z$_{\odot}$')
# ax_cc.plot(model_vi_sol50, model_ub_sol50, color='m', linewidth=3, linestyle='--')
# ax_cc.plot(model_vi_sol50[index_1_gyr[0][0]:], model_ub_sol50[index_1_gyr[0][0]:], color='m', linewidth=3,
#            linestyle='-', label='BC03, Z=Z$_{\odot}$/50')

display_models(ax=ax_cc, age_label_fontsize=fontsize+2, age_labels=True, y_color='ub', color_arrow_sol='grey',
               label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')


#
# vi_1_my = model_vi_sol[age_mod_sol == 1][0]
# ub_1_my = model_ub_sol[age_mod_sol == 1][0]
# nuvb_1_my = model_nuvb_sol[age_mod_sol == 1][0]
#
# vi_2_my = model_vi_sol[age_mod_sol == 2][0]
# ub_2_my = model_ub_sol[age_mod_sol == 2][0]
# nuvb_2_my = model_nuvb_sol[age_mod_sol == 2][0]
#
# vi_3_my = model_vi_sol[age_mod_sol == 3][0]
# ub_3_my = model_ub_sol[age_mod_sol == 3][0]
# nuvb_3_my = model_nuvb_sol[age_mod_sol == 3][0]
#
# vi_4_my = model_vi_sol[age_mod_sol == 4][0]
# ub_4_my = model_ub_sol[age_mod_sol == 4][0]
# nuvb_4_my = model_nuvb_sol[age_mod_sol == 4][0]
#
# vi_5_my = model_vi_sol[age_mod_sol == 5][0]
# ub_5_my = model_ub_sol[age_mod_sol == 5][0]
# nuvb_5_my = model_nuvb_sol[age_mod_sol == 5][0]
#
# vi_10_my = model_vi_sol[age_mod_sol == 10][0]
# ub_10_my = model_ub_sol[age_mod_sol == 10][0]
# nuvb_10_my = model_nuvb_sol[age_mod_sol == 10][0]
#
# vi_100_my = model_vi_sol[age_mod_sol == 102][0]
# ub_100_my = model_ub_sol[age_mod_sol == 102][0]
# nuvb_100_my = model_nuvb_sol[age_mod_sol == 102][0]
#
# vi_1000_my = model_vi_sol[age_mod_sol == 1028][0]
# ub_1000_my = model_ub_sol[age_mod_sol == 1028][0]
# nuvb_1000_my = model_nuvb_sol[age_mod_sol == 1028][0]
#
# vi_10000_my = model_vi_sol[age_mod_sol == 10308][0]
# ub_10000_my = model_ub_sol[age_mod_sol == 10308][0]
# nuvb_10000_my = model_nuvb_sol[age_mod_sol == 10308][0]
#
# vi_1000_my_sol50 = model_vi_sol50[age_mod_sol50 == 1028][0]
# ub_1000_my_sol50 = model_ub_sol50[age_mod_sol50 == 1028][0]
# nuvb_1000_my_sol50 = model_nuvb_sol50[age_mod_sol50 == 1028][0]
#
# vi_10000_my_sol50 = model_vi_sol50[age_mod_sol50 == 10308][0]
# ub_10000_my_sol50 = model_ub_sol50[age_mod_sol50 == 10308][0]
# nuvb_10000_my_sol50 = model_nuvb_sol50[age_mod_sol50 == 10308][0]
#
#
#
# ax_cc.scatter(vi_1_my, ub_1_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_2_my, ub_2_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_3_my, ub_3_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_4_my, ub_4_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_5_my, ub_5_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10_my, ub_10_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_100_my, ub_100_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_1000_my, ub_1000_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10000_my, ub_10000_my, c='k', s=scatter_size, zorder=10)
#
# ax_cc.scatter(vi_1000_my_sol50, ub_1000_my_sol50, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10000_my_sol50, ub_10000_my_sol50, c='k', s=scatter_size, zorder=10)
#
#
# ax_cc.annotate('1,2,3 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my - 0.1, ub_1_my - 0.3), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my + 0.1, ub_10_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
#
# ax_cc.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my - 0.5, ub_1000_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('1 Gyr', xy=(vi_1000_my_sol50, ub_1000_my_sol50), xycoords='data', xytext=(vi_1000_my - 0.5, ub_1000_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
#                     xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('10 Gyr', xy=(vi_10000_my_sol50, ub_10000_my_sol50), xycoords='data',
#                     xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))



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
