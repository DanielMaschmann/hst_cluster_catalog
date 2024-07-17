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
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from lmfit import Model, Parameters
from photutils.segmentation import deblend_sources
import math
from matplotlib import patheffects, colors


nuvb_label_dict = {
    1: {'offsets': [0.25, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
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
                   age_cut_sol5=5e2,
                   age_dots_sol=None,
                   age_dots_sol5=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=10, linestyle_sol='-', color_arrow_sol='darkcyan', arrow_linestyle_sol='--',
                   color_sol5='tab:green', linewidth_sol5=10, linestyle_sol5='-', color_arrow_sol5='darkcyan', arrow_linestyle_sol5='--',
                   color_sol50='m', linewidth_sol50=10, linestyle_sol50='-', color_arrow_sol50='darkviolet', arrow_linestyle_sol50='--',
                   label_sol=None, label_sol5=None, label_sol50=None):

    y_model_sol = globals()['model_%s_sol' % y_color]
    y_model_sol5 = globals()['model_%s_sol5' % y_color]
    y_model_sol50 = globals()['model_%s_sol50' % y_color]

    ax.plot(model_vi_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)

    ax.plot(model_vi_sol5[age_mod_sol5 > age_cut_sol5], y_model_sol5[age_mod_sol5 > age_cut_sol5],
            color=color_sol5, linewidth=linewidth_sol5, linestyle=linestyle_sol5, zorder=10, label=label_sol5)
    ax.plot(model_vi_sol5[age_mod_sol5 <= age_cut_sol5], y_model_sol5[age_mod_sol5 <= age_cut_sol5],
            color=color_sol5, linewidth=linewidth_sol5, linestyle='--', zorder=10)

    ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)
    ax.plot(model_vi_sol50[age_mod_sol50 <= age_cut_sol50], y_model_sol50[age_mod_sol50 <= age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle='--', zorder=10)


    # if age_dots_sol is None:
    #     age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
    # for age in age_dots_sol:
    #     ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='b', s=80, zorder=20)
    #
    # if age_dots_sol5 is None:
    #     age_dots_sol5 = [1, 5, 10, 100, 500, 1000, 13750]
    # for age in age_dots_sol5:
    #     ax.scatter(model_vi_sol5[age_mod_sol5 == age], y_model_sol5[age_mod_sol5 == age], color='darkgreen', s=80, zorder=20)
    #
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

# load models
age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')

age_mod_sol5 = np.load('../color_color/data_output/age_mod_sol5.npy')
model_vi_sol5 = np.load('../color_color/data_output/model_vi_sol5.npy')
model_ub_sol5 = np.load('../color_color/data_output/model_ub_sol5.npy')
model_bv_sol5 = np.load('../color_color/data_output/model_bv_sol5.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')


# gauss_dict_ubvi_ml_1 = np.load('data_output/gauss_dict_ubvi_ml_1_thresh_10.npy', allow_pickle=True).item()
# gauss_dict_bvvi_ml_1 = np.load('data_output/gauss_dict_bvvi_ml_1_thresh_10.npy', allow_pickle=True).item()

hull_gc_ubvi_ml_1_thresh_3 = fits.open('../identify_gc/data_output/convex_hull_gc_ubvi_thres2_ml.fits')[1].data
hull_gc_ubvi_ml_1_thresh_10 = fits.open('../identify_gc/data_output/convex_hull_gc_ubvi_thres10_ml.fits')[1].data
hull_gc_bvvi_ml_1_thresh_3 = fits.open('../identify_gc/data_output/convex_hull_gc_bvvi_thres2_ml.fits')[1].data
hull_gc_bvvi_ml_1_thresh_10 = fits.open('../identify_gc/data_output/convex_hull_gc_bvvi_thres10_ml.fits')[1].data

vi_hull_gc_ubvi_ml_1_thresh_3 = hull_gc_ubvi_ml_1_thresh_3['vi']
ub_hull_gc_ubvi_ml_1_thresh_3 = hull_gc_ubvi_ml_1_thresh_3['ub']
vi_hull_gc_ubvi_ml_1_thresh_10 = hull_gc_ubvi_ml_1_thresh_10['vi']
ub_hull_gc_ubvi_ml_1_thresh_10 = hull_gc_ubvi_ml_1_thresh_10['ub']

vi_hull_gc_bvvi_ml_1_thresh_3 = hull_gc_bvvi_ml_1_thresh_3['vi']
bv_hull_gc_bvvi_ml_1_thresh_3 = hull_gc_bvvi_ml_1_thresh_3['bv']
vi_hull_gc_bvvi_ml_1_thresh_10 = hull_gc_bvvi_ml_1_thresh_10['vi']
bv_hull_gc_bvvi_ml_1_thresh_10 = hull_gc_bvvi_ml_1_thresh_10['bv']



target = 'ngc4321'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

catalog_access.load_hst_cc_list(target_list=[target], classify='ml')

cluster_id = catalog_access.get_hst_cc_phangs_candidate_id(target=target, classify='ml')
color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
color_bv_ml_12 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml')
age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
age_ir4_ml_12 = catalog_access.get_hst_cc_ir4_age(target=target, classify='ml')
ebv_ir4_ml_12 = catalog_access.get_hst_cc_ir4_ebv(target=target, classify='ml')
mass_ir4_ml_12 = catalog_access.get_hst_cc_ir4_stellar_m(target=target, classify='ml')
yro_fix_ml_12 = catalog_access.get_sed_fix_flag(target=target, classify='ml', fix_type='YRO') == 1
ogc_fix_ml_12 = catalog_access.get_sed_fix_flag(target=target, classify='ml', fix_type='OGC') == 1
ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
x_ml_12, y_ml_12 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')


# get CO data from extra info table
extra_file_path = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/'
extra_file = extra_file_path + '%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target
extra_tab_hdu = fits.open(extra_file)
extra_data_table = extra_tab_hdu[1].data
extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
extra_co_intensity = extra_data_table['COmom0_strict']
co_intensity = np.zeros(len(cluster_id))
for running_index, cluster_id in enumerate(cluster_id):
    index_extra_table = np.where(extra_cluster_id == cluster_id)
    co_intensity[running_index] = extra_co_intensity[index_extra_table]


x_lim_vi = (-0.7, 2.4)
y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)

vi_int = 1.5
nuvb_int = -2.2
ub_int = -1.6
bv_int = -0.3
av_value = 1

figure = plt.figure(figsize=(40, 60))
fontsize = 55



ax_ubvi = figure.add_axes([0.075, 0.715, 0.43, 0.28])
ax_bvvi = figure.add_axes([0.565, 0.715, 0.43, 0.28])

ax_pixel = figure.add_axes([0.05, 0.395, 0.43, 0.29])
ax_pixel_co = figure.add_axes([0.485, 0.395, 0.43, 0.29])
ax_co_cbar = figure.add_axes([0.925, 0.4375, 0.015, 0.21])


ax_ebv_age_old = figure.add_axes([0.065, 0.2, 0.43, 0.165])
ax_ebv_age_fix = figure.add_axes([0.565, 0.2, 0.43, 0.165])

ax_mass_age_old = figure.add_axes([0.065, 0.03, 0.43, 0.165])
ax_mass_age_fix = figure.add_axes([0.565, 0.03, 0.43, 0.165])

ax_ubvi.scatter(color_vi_ml_12, color_ub_ml_12, s=140, alpha=0.7, color='grey')
ax_bvvi.scatter(color_vi_ml_12, color_bv_ml_12, s=140, alpha=0.7, color='grey')

ax_ubvi.scatter(color_vi_ml_12[yro_fix_ml_12], color_ub_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_ubvi.scatter(color_vi_ml_12[ogc_fix_ml_12], color_ub_ml_12[ogc_fix_ml_12], s=200, color='red')
ax_bvvi.scatter(color_vi_ml_12[yro_fix_ml_12], color_bv_ml_12[yro_fix_ml_12], s=200, color='blue', label='YRO fix')
ax_bvvi.scatter(color_vi_ml_12[ogc_fix_ml_12], color_bv_ml_12[ogc_fix_ml_12], s=200, color='red', label='OGC fix')

display_models(ax=ax_ubvi, age_label_fontsize=fontsize+2, y_color='ub',
               label_sol=r'BC03, Z$_{\odot}$',
               label_sol5=r'BC03, Z$_{\odot}/5\,(> 500\,{\rm Myr})$',
               label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_bvvi, age_label_fontsize=fontsize+2, y_color='bv')
ax_ubvi.legend(frameon=False, loc=3, fontsize=fontsize)
ax_bvvi.legend(frameon=False, loc=3, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax_ubvi, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize, x_text_offset=-0.0, y_text_offset=-0.1)
hf.plot_reddening_vect(ax=ax_bvvi, x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize, x_text_offset=+0.04, y_text_offset=-0.05)

ax_ubvi.plot(vi_hull_gc_ubvi_ml_1_thresh_10, ub_hull_gc_ubvi_ml_1_thresh_10, color='red', linewidth=8)
ax_bvvi.plot(vi_hull_gc_bvvi_ml_1_thresh_10, bv_hull_gc_bvvi_ml_1_thresh_10, color='red', linewidth=8)
ax_bvvi.plot([0.77295, 0.77295], [2.5, 0.6528], alpha=0.7, color='grey', linewidth=6, linestyle='--')
ax_bvvi.plot([1.27252, 2.9], [0.2831, 0.2831], alpha=0.7, color='grey', linewidth=6, linestyle='--')

ax_ubvi.text(0.02, 0.95, 'NGC4321', horizontalalignment='left', verticalalignment='center', fontsize=fontsize+5,
              transform=ax_ubvi.transAxes)


ax_ubvi.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_bvvi.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ubvi.set_ylabel('U (F336W) - B (F438W)', labelpad=0, fontsize=fontsize)
ax_bvvi.set_ylabel('B (F438W) - V (F555W)', labelpad=-30, fontsize=fontsize)

ax_ubvi.set_xlim(x_lim_vi)
ax_bvvi.set_xlim(x_lim_vi)
ax_ubvi.set_ylim(y_lim_ub)
ax_bvvi.set_ylim(y_lim_bv)
ax_ubvi.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)
ax_bvvi.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)


ax_pixel.scatter(x_ml_12, y_ml_12, s=140, alpha=0.7, color='grey')
ax_pixel.scatter(x_ml_12[yro_fix_ml_12], y_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_pixel.scatter(x_ml_12[ogc_fix_ml_12], y_ml_12[ogc_fix_ml_12], s=200, color='red')



cmap_co = matplotlib.cm.get_cmap('inferno_r')
norm_co = colors.LogNorm(vmin=0.5, vmax=580)
# norm_age = matplotlib.colors.Normalize(vmin=6, vmax=10.5)

ax_pixel_co.scatter(x_ml_12, y_ml_12, s=140, alpha=0.7, color='grey')

co_intensity[co_intensity == 0] = 0.05
ax_pixel_co.scatter(x_ml_12, y_ml_12, c=co_intensity, norm=norm_co, cmap=cmap_co, s=200)

ColorbarBase(ax_co_cbar, orientation='vertical', cmap=cmap_co, norm=norm_co, extend='neither', ticks=None)
ax_co_cbar.set_ylabel(r'$\Sigma$ CO(2-1) [mJy]', labelpad=0, fontsize=fontsize)
ax_co_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)
ax_co_cbar.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
ax_co_cbar.set_yticklabels(['1', '2', '5', '10', '20', '50', '100', '200', '500'])

ax_pixel.set_xlabel('PHANGS X [pix]', fontsize=fontsize)
ax_pixel_co.set_xlabel('PHANGS X [pix]', fontsize=fontsize)
ax_pixel.set_ylabel('PHANGS Y [pix]', labelpad=27, fontsize=fontsize)

ax_pixel.set_yticklabels(ax_pixel.get_yticklabels(), rotation=90, va="center", ha='right')
ax_pixel_co.set_yticklabels([])
ax_pixel_co.set_xticks([2000, 3000, 4000, 5000, 6000, 7000, 8000])
ax_pixel_co.set_xticklabels(['', '3000', '4000', '5000', '6000', '7000', '8000'])
ax_pixel.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)
ax_pixel_co.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)


random_age = np.random.uniform(low=-0.1, high=0.1, size=len(age_ml_12))
ax_ebv_age_old.scatter(np.log10(age_ir4_ml_12) + 6 + random_age, ebv_ir4_ml_12, s=140, alpha=0.7, color='grey')
ax_ebv_age_fix.scatter(np.log10(age_ml_12) + 6 + random_age, ebv_ml_12, s=140, alpha=0.7, color='grey')

ax_ebv_age_old.scatter(np.log10(age_ir4_ml_12[yro_fix_ml_12]) + 6 + random_age[yro_fix_ml_12], ebv_ir4_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_ebv_age_old.scatter(np.log10(age_ir4_ml_12[ogc_fix_ml_12]) + 6 + random_age[ogc_fix_ml_12], ebv_ir4_ml_12[ogc_fix_ml_12], s=200, color='red')
ax_ebv_age_fix.scatter(np.log10(age_ml_12[yro_fix_ml_12]) + 6 + random_age[yro_fix_ml_12], ebv_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_ebv_age_fix.scatter(np.log10(age_ml_12[ogc_fix_ml_12]) + 6 + random_age[ogc_fix_ml_12], ebv_ml_12[ogc_fix_ml_12], s=200, color='red')

ax_ebv_age_old.set_yscale('log')
ax_ebv_age_fix.set_yscale('log')

ax_ebv_age_fix.set_ylim(ax_ebv_age_old.get_ylim())

ax_ebv_age_old.set_xticklabels([])
ax_ebv_age_fix.set_xticklabels([])

ax_ebv_age_old.set_ylabel('E(B-V) [mag] (standard)', labelpad=0, fontsize=fontsize)
ax_ebv_age_fix.set_ylabel('E(B-V) [mag] (SED fixed)', labelpad=0, fontsize=fontsize)

ax_ebv_age_old.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_age_fix.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)


ax_mass_age_old.scatter(np.log10(age_ir4_ml_12) + 6 + random_age, mass_ir4_ml_12, s=140, alpha=0.7, color='grey')
ax_mass_age_fix.scatter(np.log10(age_ml_12) + 6 + random_age, mass_ml_12, s=140, alpha=0.7, color='grey')

ax_mass_age_old.scatter(np.log10(age_ir4_ml_12[yro_fix_ml_12]) + 6 + random_age[yro_fix_ml_12], mass_ir4_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_mass_age_old.scatter(np.log10(age_ir4_ml_12[ogc_fix_ml_12]) + 6 + random_age[ogc_fix_ml_12], mass_ir4_ml_12[ogc_fix_ml_12], s=200, color='red')
ax_mass_age_fix.scatter(np.log10(age_ml_12[yro_fix_ml_12]) + 6 + random_age[yro_fix_ml_12], mass_ml_12[yro_fix_ml_12], s=200, color='blue')
ax_mass_age_fix.scatter(np.log10(age_ml_12[ogc_fix_ml_12]) + 6 + random_age[ogc_fix_ml_12], mass_ml_12[ogc_fix_ml_12], s=200, color='red')

ax_mass_age_old.set_ylabel(r'M$_*$ [M$_{\odot}$] (standard)', labelpad=30, fontsize=fontsize)
ax_mass_age_fix.set_ylabel(r'M$_*$ [M$_{\odot}$] (SED fixed)', labelpad=30, fontsize=fontsize)

ax_mass_age_old.set_xlabel('Age [yr] (standard)', labelpad=0, fontsize=fontsize)
ax_mass_age_fix.set_xlabel('Age [yr] (SED fixed)', labelpad=0, fontsize=fontsize)

ax_mass_age_old.set_yscale('log')
ax_mass_age_fix.set_yscale('log')

ax_mass_age_old.set_xticks([6, 7, 8, 9, 10])
ax_mass_age_old.set_xticklabels(['1', '10', '100', '1000', '10000'])

ax_mass_age_fix.set_xticks([6, 7, 8, 9, 10])
ax_mass_age_fix.set_xticklabels(['1', '10', '100', '1000', '10000'])

ax_mass_age_fix.set_ylim(ax_mass_age_old.get_ylim())



ax_mass_age_old.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)
ax_mass_age_fix.tick_params(axis='both', which='both', width=3, length=13, right=True, top=True, direction='in', labelsize=fontsize)


plt.savefig('plot_output/sed_fix_example_pattern.png')
plt.savefig('plot_output/sed_fix_example_pattern.pdf')
