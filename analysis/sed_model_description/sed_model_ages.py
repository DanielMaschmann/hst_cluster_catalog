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
from matplotlib import patheffects


nuvb_label_dict = {
    1: {'offsets': [0.25, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    #1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
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
    500: {'offset': [+0.3, -0.3], 'label': '500 Myr', 'ha': 'left', 'va': 'center'},
    1000: {'offset': [-0.3, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
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
    # ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
    #         color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)
    #
    # ax.plot(model_vi_sol50[age_mod_sol50 <= age_cut_sol50], y_model_sol50[age_mod_sol50 <= age_cut_sol50],
    #         color=color_sol50, linewidth=linewidth_sol50, linestyle='--', zorder=10)


    if age_dots_sol is None:
        age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
    for age in age_dots_sol:
        ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='k', s=80, zorder=20)

    # if age_dots_sol50 is None:
    #     age_dots_sol50 = [500, 1000, 13750]
    # for age in age_dots_sol50:
    #     ax.scatter(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age], color='tab:pink', s=80, zorder=20)

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
            # txt_sol50 = ax.annotate(' ',
            #             xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
            #             xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
            #                     y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
            #             fontsize=age_label_fontsize, xycoords='data', textcoords='data',
            #             ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
            #                   arrowprops=dict(arrowstyle='-|>', color=color_arrow_sol50, lw=3, ls=arrow_linestyle_sol50),
            #             path_effects=[patheffects.withStroke(linewidth=3,
            #                                             foreground="w")])
            # txt_sol50.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
            #                                       patheffects.Normal()])
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



def plot_age_ebv(ax, age=1, ebv=0.0, color='k', linestyle='-', linewidth=3):
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
                                             cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 10 * u.Mpc

def plot_age_only_sol50(ax, age, mass, color='k', linestyle='-', linewidth=3):

    id = np.where(age_mod_sol == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/no_dust/sol_met_50/out/%i_best_model.fits'%id,
                                            cluster_mass=mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)

# set star cluster mass to scale the model SEDs
cluster_mass = 1E5 * u.Msun
# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 10 * u.Mpc
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
ax_cc_zoom = figure.add_axes([0.28, 0.61, 0.14, 0.35])

# plot observation filters
cigale_wrapper_obj.plot_hst_filters(ax=ax_sed, fontsize=fontsize-5, color='k')

sol_x_lim_ub = (0.52, 0.81)
sol_y_lim_ub = (-0.96, -1.22)
scatter_size = 80

ax_cc.set_ylim(1.25, -2.2)
ax_cc.set_xlim(-0.9, 1.8)

index_1_gyr = np.where(age_mod_sol50 == 500)

#
# ax_cc.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=3, label='BC03, Z=Z$_{\odot}$')
# # ax_cc.plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=3, linestyle='--')
# # ax_cc.plot(model_vi_sol50[index_1_gyr[0][0]:], model_ub_sol50[index_1_gyr[0][0]:], color='darkorange', linewidth=3, linestyle='-', label='BC03, Z=Z$_{\odot}$/50')
ax_cc_zoom.plot(model_vi_sol, model_ub_sol, color='tab:cyan', linewidth=3)

display_models(ax=ax_cc, age_label_fontsize=fontsize+2, age_labels=True, y_color='ub', color_arrow_sol='grey',
               label_sol=r'BC03, Z$_{\odot}$ (no neb.)', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')

# get model
hdu_a_sol_neb = fits.open('/home/benutzer/Documents/projects/hst_cluster_catalog/analysis/cigale_model/sfh2exp/no_dust/sol_met_gas/out/models-block-0.fits')
data_mod_sol_neb = hdu_a_sol_neb[1].data
# print(data_mod_sol_neb.names)

ew_h_alpha = data_mod_sol_neb['param.EW(656.3/1.0)']
age_mod_sol_neb = data_mod_sol_neb['sfh.age']
logu_mod_sol_neb = data_mod_sol_neb['nebular.logU']
ne_mod_sol_neb = data_mod_sol_neb['nebular.ne']
f_esc_mod_sol_neb = data_mod_sol_neb['nebular.f_esc']

flux_f275w_sol_neb = data_mod_sol_neb['F275W_UVIS_CHIP2']
flux_f336w_sol_neb = data_mod_sol_neb['F336W_UVIS_CHIP2']
flux_f438w_sol_neb = data_mod_sol_neb['F438W_UVIS_CHIP2']
flux_f555w_sol_neb = data_mod_sol_neb['F555W_UVIS_CHIP2']
flux_f814w_sol_neb = data_mod_sol_neb['F814W_UVIS_CHIP2']


mag_nuv_sol_neb = hf.conv_mjy2vega(flux=flux_f275w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sol_neb = hf.conv_mjy2vega(flux=flux_f336w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol_neb = hf.conv_mjy2vega(flux=flux_f438w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sol_neb = hf.conv_mjy2vega(flux=flux_f555w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol_neb = hf.conv_mjy2vega(flux=flux_f814w_sol_neb, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvu_sol_neb = mag_nuv_sol_neb - mag_u_sol_neb
model_nuvb_sol_neb = mag_nuv_sol_neb - mag_b_sol_neb
model_ub_sol_neb = mag_u_sol_neb - mag_b_sol_neb
model_bv_sol_neb = mag_b_sol_neb - mag_v_sol_neb
model_bi_sol_neb = mag_b_sol_neb - mag_i_sol_neb
model_vi_sol_neb = mag_v_sol_neb - mag_i_sol_neb

mask_ne10 = (ne_mod_sol_neb == 10)
mask_ne100 = (ne_mod_sol_neb == 100)
mask_ne1000 = (ne_mod_sol_neb == 1000)

mask_logu2 = (logu_mod_sol_neb == -2)
mask_logu3 = (logu_mod_sol_neb == -3)
mask_logu4 = (logu_mod_sol_neb == -4)

mask_ages = ((age_mod_sol_neb == 1) | (age_mod_sol_neb == 2) | (age_mod_sol_neb == 3) | (age_mod_sol_neb == 4) | (age_mod_sol_neb == 5) | (age_mod_sol_neb == 6))
mask_f_esc = f_esc_mod_sol_neb == 0.5

ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu2], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu2], marker='P', color='r', s=100, label='ne=10, logU=-2')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu3], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu3], marker='P', color='g', s=100, label='ne=10, logU=-3')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu4], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne10*mask_logu4], marker='P', color='b', s=100, label='ne=10, logU=-4')

ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu2], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu2], marker='o', color='r', s=100, label='ne=100, logU=-2')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu3], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu3], marker='o', color='g', s=100, label='ne=100, logU=-3')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu4], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne100*mask_logu4], marker='o', color='b', s=100, label='ne=100, logU=-4')

ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu2], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu2], marker='X', color='r', s=100, label='ne=1000, logU=-2')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu3], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu3], marker='X', color='g', s=100, label='ne=1000, logU=-3')
ax_cc.scatter(model_vi_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu4], model_ub_sol_neb[mask_ages*mask_f_esc*mask_ne1000*mask_logu4], marker='X', color='b', s=100, label='ne=1000, logU=-4')




ax_cc_zoom.set_ylim(sol_y_lim_ub)
ax_cc_zoom.set_xlim(sol_x_lim_ub)
ax_cc.plot([sol_x_lim_ub[0], sol_x_lim_ub[0]], [sol_y_lim_ub[0], sol_y_lim_ub[1]], color='k', linewidth=2)
ax_cc.plot([sol_x_lim_ub[1], sol_x_lim_ub[1]], [sol_y_lim_ub[0], sol_y_lim_ub[1]], color='k', linewidth=2)
ax_cc.plot([sol_x_lim_ub[0], sol_x_lim_ub[1]], [sol_y_lim_ub[0], sol_y_lim_ub[0]], color='k', linewidth=2)
ax_cc.plot([sol_x_lim_ub[0], sol_x_lim_ub[1]], [sol_y_lim_ub[1], sol_y_lim_ub[1]], color='k', linewidth=2)

con_spec_1 = ConnectionPatch(
    xyA=(sol_x_lim_ub[0], sol_y_lim_ub[1]), coordsA=ax_cc.transData,
    xyB=(ax_cc_zoom.get_xlim()[0], ax_cc_zoom.get_ylim()[1]), coordsB=ax_cc_zoom.transData,
    linestyle="--", linewidth=2, color='k')
figure.add_artist(con_spec_1)
con_spec_2 = ConnectionPatch(
    xyA=(sol_x_lim_ub[0], sol_y_lim_ub[0]), coordsA=ax_cc.transData,
    xyB=(ax_cc_zoom.get_xlim()[0], ax_cc_zoom.get_ylim()[0]), coordsB=ax_cc_zoom.transData,
    linestyle="--", linewidth=2, color='k')
figure.add_artist(con_spec_2)

vi_1_my = model_vi_sol[age_mod_sol == 1][0]
ub_1_my = model_ub_sol[age_mod_sol == 1][0]
nuvb_1_my = model_nuvb_sol[age_mod_sol == 1][0]

vi_2_my = model_vi_sol[age_mod_sol == 2][0]
ub_2_my = model_ub_sol[age_mod_sol == 2][0]
nuvb_2_my = model_nuvb_sol[age_mod_sol == 2][0]

vi_3_my = model_vi_sol[age_mod_sol == 3][0]
ub_3_my = model_ub_sol[age_mod_sol == 3][0]
nuvb_3_my = model_nuvb_sol[age_mod_sol == 3][0]

vi_4_my = model_vi_sol[age_mod_sol == 4][0]
ub_4_my = model_ub_sol[age_mod_sol == 4][0]
nuvb_4_my = model_nuvb_sol[age_mod_sol == 4][0]

vi_5_my = model_vi_sol[age_mod_sol == 5][0]
ub_5_my = model_ub_sol[age_mod_sol == 5][0]
nuvb_5_my = model_nuvb_sol[age_mod_sol == 5][0]

vi_10_my = model_vi_sol[age_mod_sol == 10][0]
ub_10_my = model_ub_sol[age_mod_sol == 10][0]
nuvb_10_my = model_nuvb_sol[age_mod_sol == 10][0]

vi_100_my = model_vi_sol[age_mod_sol == 102][0]
ub_100_my = model_ub_sol[age_mod_sol == 102][0]
nuvb_100_my = model_nuvb_sol[age_mod_sol == 102][0]

vi_1000_my = model_vi_sol[age_mod_sol == 1028][0]
ub_1000_my = model_ub_sol[age_mod_sol == 1028][0]
nuvb_1000_my = model_nuvb_sol[age_mod_sol == 1028][0]

vi_10000_my = model_vi_sol[age_mod_sol == 10308][0]
ub_10000_my = model_ub_sol[age_mod_sol == 10308][0]
nuvb_10000_my = model_nuvb_sol[age_mod_sol == 10308][0]

vi_1000_my_sol50 = model_vi_sol50[age_mod_sol50 == 1028][0]
ub_1000_my_sol50 = model_ub_sol50[age_mod_sol50 == 1028][0]
nuvb_1000_my_sol50 = model_nuvb_sol50[age_mod_sol50 == 1028][0]

vi_10000_my_sol50 = model_vi_sol50[age_mod_sol50 == 10308][0]
ub_10000_my_sol50 = model_ub_sol50[age_mod_sol50 == 10308][0]
nuvb_10000_my_sol50 = model_nuvb_sol50[age_mod_sol50 == 10308][0]



# ax_cc.scatter(vi_1_my, ub_1_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_2_my, ub_2_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_3_my, ub_3_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_4_my, ub_4_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_5_my, ub_5_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10_my, ub_10_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_100_my, ub_100_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_1000_my, ub_1000_my, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10000_my, ub_10000_my, c='k', s=scatter_size, zorder=10)

# ax_cc.scatter(vi_1000_my_sol50, ub_1000_my_sol50, c='k', s=scatter_size, zorder=10)
# ax_cc.scatter(vi_10000_my_sol50, ub_10000_my_sol50, c='k', s=scatter_size, zorder=10)


# ax_cc.annotate('1,2,3 Myr', xy=(vi_1_my, ub_1_my), xycoords='data', xytext=(vi_1_my - 0.1, ub_1_my - 0.3), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('4 Myr', xy=(vi_4_my, ub_4_my), xycoords='data', xytext=(vi_4_my - 0.5, ub_4_my), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('5 Myr', xy=(vi_5_my, ub_5_my), xycoords='data', xytext=(vi_5_my - 0.5, ub_5_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('10 Myr', xy=(vi_10_my, ub_10_my), xycoords='data', xytext=(vi_10_my - 0.5, ub_10_my - 0.4), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('100 Myr', xy=(vi_100_my, ub_100_my), xycoords='data', xytext=(vi_100_my - 0.5, ub_100_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
#
# ax_cc.annotate('1 Gyr', xy=(vi_1000_my, ub_1000_my), xycoords='data', xytext=(vi_1000_my - 0.5, ub_1000_my + 0.5), fontsize=fontsize,
#                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# # ax_cc.annotate('1 Gyr', xy=(vi_1000_my_sol50, ub_1000_my_sol50), xycoords='data', xytext=(vi_1000_my - 0.5, ub_1000_my + 0.5), fontsize=fontsize,
# #                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# ax_cc.annotate('10 Gyr', xy=(vi_10000_my, ub_10000_my), xycoords='data',
#                     xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
#                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
# # ax_cc.annotate('10 Gyr', xy=(vi_10000_my_sol50, ub_10000_my_sol50), xycoords='data',
# #                     xytext=(vi_10000_my + 0.0, ub_10000_my - 0.7), textcoords='data', fontsize=fontsize,
# #                     arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


vi_12_my_zoom = model_vi_sol[age_mod_sol == 12][0]
ub_12_my_zoom = model_ub_sol[age_mod_sol == 12][0]
nuvb_12_my_zoom = model_nuvb_sol[age_mod_sol == 12][0]

vi_14_my_zoom = model_vi_sol[age_mod_sol == 14][0]
ub_14_my_zoom = model_ub_sol[age_mod_sol == 14][0]
nuvb_14_my_zoom = model_nuvb_sol[age_mod_sol == 14][0]

vi_18_my_zoom = model_vi_sol[age_mod_sol == 18][0]
ub_18_my_zoom = model_ub_sol[age_mod_sol == 18][0]
nuvb_18_my_zoom = model_nuvb_sol[age_mod_sol == 18][0]

vi_20_my_zoom = model_vi_sol[age_mod_sol == 20][0]
ub_20_my_zoom = model_ub_sol[age_mod_sol == 20][0]
nuvb_20_my_zoom = model_nuvb_sol[age_mod_sol == 20][0]

vi_25_my_zoom = model_vi_sol[age_mod_sol == 25][0]
ub_25_my_zoom = model_ub_sol[age_mod_sol == 25][0]
nuvb_25_my_zoom = model_nuvb_sol[age_mod_sol == 25][0]

vi_32_my_zoom = model_vi_sol[age_mod_sol == 32][0]
ub_32_my_zoom = model_ub_sol[age_mod_sol == 32][0]
nuvb_32_my_zoom = model_nuvb_sol[age_mod_sol == 32][0]


ax_cc_zoom.scatter(vi_12_my_zoom, ub_12_my_zoom, c='k', s=scatter_size, zorder=10)
ax_cc_zoom.scatter(vi_14_my_zoom, ub_14_my_zoom, c='k', s=scatter_size, zorder=10)
ax_cc_zoom.scatter(vi_18_my_zoom, ub_18_my_zoom, c='k', s=scatter_size, zorder=10)
ax_cc_zoom.scatter(vi_20_my_zoom, ub_20_my_zoom, c='k', s=scatter_size, zorder=10)
ax_cc_zoom.scatter(vi_25_my_zoom, ub_25_my_zoom, c='k', s=scatter_size, zorder=10)
ax_cc_zoom.scatter(vi_32_my_zoom, ub_32_my_zoom, c='k', s=scatter_size, zorder=10)


ax_cc_zoom.annotate('12 Myr', color='red', xy=(vi_12_my_zoom, ub_12_my_zoom), xycoords='data', xytext=(vi_12_my_zoom + 0.05, ub_12_my_zoom + 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))
ax_cc_zoom.annotate('14 Myr', color='red', xy=(vi_14_my_zoom, ub_14_my_zoom), xycoords='data', xytext=(vi_14_my_zoom - 0.05, ub_14_my_zoom - 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))
ax_cc_zoom.annotate('18 Myr', color='red', xy=(vi_18_my_zoom, ub_18_my_zoom), xycoords='data', xytext=(vi_18_my_zoom + 0.03, ub_18_my_zoom - 0.01), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))
ax_cc_zoom.annotate('20 Myr', color='red', xy=(vi_20_my_zoom, ub_20_my_zoom), xycoords='data', xytext=(vi_20_my_zoom - 0.06, ub_20_my_zoom - 0.03), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))
ax_cc_zoom.annotate('25 Myr', color='red', xy=(vi_25_my_zoom, ub_25_my_zoom), xycoords='data', xytext=(vi_25_my_zoom + 0.06, ub_25_my_zoom + 0.05), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))
ax_cc_zoom.annotate('32 Myr', color='red', xy=(vi_32_my_zoom, ub_32_my_zoom), xycoords='data', xytext=(vi_32_my_zoom - 0.01, ub_32_my_zoom - 0.07), fontsize=fontsize,
                     textcoords='data', arrowprops=dict(arrowstyle='-|>', color='grey', lw=2, ls='-'))


ax_cc.legend(frameon=False, loc=3, fontsize=fontsize-5)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_cc_zoom.yaxis.tick_right()
ax_cc_zoom.tick_params(axis='both', which='both', width=1.5, length=4, left=True, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
# ax_cc.set_xticklabels([])


def plot_age_only(ax, age, color='k', linestyle='-', linewidth=3):

    id = np.where(age_mod_sol == age)[0][0]
    cigale_wrapper_obj.plot_cigale_model(ax=ax, model_file_name='../cigale_model/sfh2exp/no_dust/sol_met/out/%i_best_model.fits'%id,
                                            cluster_mass=cluster_mass, distance_Mpc=distance_Mpc,
                                             color=color, linestyle=linestyle, linewidth=linewidth)


age_list = [1, 2, 3, 4, 5, 7, 10, 20, 30, 50, 100, 200, 300, 500, 800, 1000, 1584, 2819, 3255, 5015, 7727, 10308, 13750]
age_list = list(reversed(age_list))

cmap = matplotlib.cm.get_cmap('Spectral_r')

# rgba = cmap(0.5)
# norm = matplotlib.colors.Normalize(vmin=1.0, vmax=10000)
norm = matplotlib.colors.LogNorm(vmin=1.0, vmax=10000)

for age in age_list:
    color = cmap(norm(age))
    plot_age_only(ax=ax_sed, age=age, color=color, linestyle='-', linewidth=3)


cluster_mass_old_2 = 1E6 * u.Msun

plot_age_only_sol50(ax=ax_sed, age=10000, mass=cluster_mass_old_2, color='k', linestyle=':', linewidth=4)


ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_ylabel(r'Age [Myr]', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)

hf.plot_reddening_vect(ax=ax_cc,
                       x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=0.1, y_color_int=-1.8, av_val=1,
                       linewidth=3, line_color='k', text=True, fontsize=fontsize,
                       x_text_offset=-0.05, y_text_offset=-0.1)

# ax_sed.text(500, 2, r'Z=Z$_{\odot}$, M$_{*}$ = 10$^{5}$ M$_{\odot}$, D = 10 Mpc', fontsize=fontsize)
ax_sed.text(700, 0.8, r'D = 10 Mpc', fontsize=fontsize)


ax_sed.plot([], [], color='gray', linestyle='-', linewidth=3, label=r'Age = 1 Myr - 13.8 Gyr, Z=Z$_{\odot}$, M$_{*}$ = 10$^{5}$ M$_{\odot}$,A$_{\rm V}$ = 0')
ax_sed.plot([], [], color='k', linestyle=':', linewidth=4, label=r'Age = 10 Gyr, Z=Z$_{\odot}$/50, M$_{*}$ = 10$^{6}$ M$_{\odot}$, A$_{\rm V}$ = 0')

# ax_sed.legend(loc='upper left', fontsize=fontsize-6, ncol=3, columnspacing=1, handlelength=1, handletextpad=0.6)
ax_sed.legend(fontsize=fontsize-6, loc=4)

ax_sed.set_xlim(230, 0.9* 1e3)
ax_sed.set_ylim(7e-8, 1.5e1)

ax_sed.set_xscale('log')
ax_sed.set_yscale('log')

ax_sed.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_sed.set_ylabel(r'F$_{\nu}$ [mJy]                       ', fontsize=fontsize)
ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)


plt.savefig('plot_output/sed_color_color_ages.png')
plt.savefig('plot_output/sed_color_color_ages.pdf')

exit(9)

