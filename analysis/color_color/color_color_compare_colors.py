import numpy as np
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
    13750: {'offset': [+0.05, -0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}


def display_models(ax, y_color='nuvb',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-',
                   label_sol=None, label_sol50=None):

    y_model_sol = globals()['model_%s_sol' % y_color]
    y_model_sol50 = globals()['model_%s_sol50' % y_color]

    ax.plot(model_vi_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)
    ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)

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

            txt_sol = ax.annotate(' ', #annotation_dict[age]['label'],
                        xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'),
                        path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
            txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            txt_sol50 = ax.annotate(' ',
                        xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'),
                        path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
            txt_sol50.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                  patheffects.Normal()])
            ax.text(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40, path_effects=pe)


age_mod_sol = np.load('data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('data_output/model_ub_sol.npy')
model_bv_sol = np.load('data_output/model_bv_sol.npy')
model_bi_sol = np.load('data_output/model_bi_sol.npy')
model_vi_sol = np.load('data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('data_output/model_bv_sol50.npy')
model_bi_sol50 = np.load('data_output/model_bi_sol50.npy')
model_vi_sol50 = np.load('data_output/model_vi_sol50.npy')


color_vi_hum = np.load('data_output/color_vi_hum.npy')
color_ub_hum = np.load('data_output/color_ub_hum.npy')
color_bv_hum = np.load('data_output/color_bv_hum.npy')
color_bi_hum = np.load('data_output/color_bi_hum.npy')
color_nuvu_hum = np.load('data_output/color_nuvu_hum.npy')
color_nuvb_hum = np.load('data_output/color_nuvb_hum.npy')
color_vi_err_hum = np.load('data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('data_output/color_bv_err_hum.npy')
color_bi_err_hum = np.load('data_output/color_bi_err_hum.npy')
color_nuvu_err_hum = np.load('data_output/color_nuvu_err_hum.npy')
color_nuvb_err_hum = np.load('data_output/color_nuvb_err_hum.npy')
detect_nuv_hum = np.load('data_output/detect_nuv_hum.npy')
detect_u_hum = np.load('data_output/detect_u_hum.npy')
detect_b_hum = np.load('data_output/detect_b_hum.npy')
detect_v_hum = np.load('data_output/detect_v_hum.npy')
detect_i_hum = np.load('data_output/detect_i_hum.npy')
clcl_color_hum = np.load('data_output/clcl_color_hum.npy')
age_hum = np.load('data_output/age_hum.npy')
ebv_hum = np.load('data_output/ebv_hum.npy')
color_vi_ml = np.load('data_output/color_vi_ml.npy')
color_ub_ml = np.load('data_output/color_ub_ml.npy')
color_bv_ml = np.load('data_output/color_bv_ml.npy')
color_bi_ml = np.load('data_output/color_bi_ml.npy')
color_nuvu_ml = np.load('data_output/color_nuvu_ml.npy')
color_nuvb_ml = np.load('data_output/color_nuvb_ml.npy')
color_vi_err_ml = np.load('data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('data_output/color_bv_err_ml.npy')
color_bi_err_ml = np.load('data_output/color_bi_err_ml.npy')
color_nuvu_err_ml = np.load('data_output/color_nuvu_err_ml.npy')
color_nuvb_err_ml = np.load('data_output/color_nuvb_err_ml.npy')
detect_nuv_ml = np.load('data_output/detect_nuv_ml.npy')
detect_u_ml = np.load('data_output/detect_u_ml.npy')
detect_b_ml = np.load('data_output/detect_b_ml.npy')
detect_v_ml = np.load('data_output/detect_v_ml.npy')
detect_i_ml = np.load('data_output/detect_i_ml.npy')
clcl_color_ml = np.load('data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('data_output/clcl_qual_color_ml.npy')
age_ml = np.load('data_output/age_ml.npy')
ebv_ml = np.load('data_output/ebv_ml.npy')
mag_mask_ml = np.load('data_output/mag_mask_ml.npy')

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_nuvbvi_hum = detect_nuv_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_bvvi_hum = detect_b_hum * detect_v_hum * detect_i_hum

x_lim_vi = (-0.7, 2.4)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)
n_bins = 190
kernal_std = 3.0

mask_good_colors_nuvbvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_nuvb_hum > (y_lim_nuvb[1] - 1)) & (color_nuvb_hum < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_hum)
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_hum)
mask_good_colors_bvvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                               (color_bv_hum > (y_lim_bv[1] - 1)) & (color_bv_hum < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_hum)


fig, ax = plt.subplots(ncols=3, nrows=3, sharex='all', sharey='row', figsize=(24, 20))
fontsize = 25


hf.density_with_points(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                    y=color_nuvb_hum[mask_class_1_hum * mask_good_colors_nuvbvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                    y=color_nuvb_hum[mask_class_2_hum * mask_good_colors_nuvbvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[0, 2], x=color_vi_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                    y=color_nuvb_hum[mask_class_3_hum * mask_good_colors_nuvbvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

hf.density_with_points(ax=ax[1, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                    y=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[1, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                    y=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[1, 2], x=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                    y=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

hf.density_with_points(ax=ax[2, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                    y=color_bv_hum[mask_class_1_hum * mask_good_colors_bvvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[2, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                    y=color_bv_hum[mask_class_2_hum * mask_good_colors_bvvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)
hf.density_with_points(ax=ax[2, 2], x=color_vi_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                    y=color_bv_hum[mask_class_3_hum * mask_good_colors_bvvi_hum],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

vi_int = 1.75
nuvb_int = -2.2
ub_int = -1.8
bv_int = -0.5
av_value = 1

hf.plot_reddening_vect(ax=ax[0, 0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax[0, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[0, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax[1, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[1, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax[2, 0], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[2, 1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[2, 2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


display_models(ax=ax[0, 0], age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax[0, 1], age_label_fontsize=fontsize+2)
display_models(ax=ax[0, 2], age_label_fontsize=fontsize+2, age_labels=True)

display_models(ax=ax[1, 0], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[1, 1], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[1, 2], age_label_fontsize=fontsize+2, y_color='ub', age_labels=True)

display_models(ax=ax[2, 0], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[2, 1], age_label_fontsize=fontsize+2, y_color='bv')
display_models(ax=ax[2, 2], age_label_fontsize=fontsize+2, y_color='bv', age_labels=True)


# display different regions
# Young Cluster Locus (blue)
# Middle-Age Plume (green)
# Old Globular Cluster Clump (red)

txt_ogcc = ax[1, 0].annotate('Old Globular \n Cluster Clump', xy=(1.15, 0.0), xytext=(1.85, 0.9), fontsize=fontsize,
                            xycoords='data', textcoords='data', ha='center', va='center', zorder=30,
                            arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
txt_ogcc.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"), patheffects.Normal()])

txt_map = ax[1, 0].annotate('Middle-Age \n Plume', xy=(0.65, -0.5), xytext=(-0.1, 0.7), fontsize=fontsize,
                            xycoords='data', textcoords='data', ha='center', va='center', zorder=30,
                            arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
txt_map.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"), patheffects.Normal()])

txt_ycl = ax[1, 0].annotate('Young \n Cluster Locus', xy=(0.4, -1.4), xytext=(0.7, -1.9), fontsize=fontsize,
                            xycoords='data', textcoords='data', ha='center', va='center', zorder=30,
                            arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
txt_ycl.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"), patheffects.Normal()])

pe = [patheffects.withStroke(linewidth=3, foreground="w")]
ax[0, 0].text(0.02, 0.95, 'a)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[0, 0].transAxes, path_effects=pe)
ax[0, 1].text(0.02, 0.95, 'b)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[0, 1].transAxes, path_effects=pe)
ax[0, 2].text(0.02, 0.95, 'c)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[0, 2].transAxes, path_effects=pe)
ax[1, 0].text(0.02, 0.95, 'd)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[1, 0].transAxes, path_effects=pe)
ax[1, 1].text(0.02, 0.95, 'e)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[1, 1].transAxes, path_effects=pe)
ax[1, 2].text(0.02, 0.95, 'f)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[1, 2].transAxes, path_effects=pe)
ax[2, 0].text(0.02, 0.95, 'g)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[2, 0].transAxes, path_effects=pe)
ax[2, 1].text(0.02, 0.95, 'h)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[2, 1].transAxes, path_effects=pe)
ax[2, 2].text(0.02, 0.95, 'i)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[2, 2].transAxes, path_effects=pe)


xedges = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins)
yedges = np.linspace(y_lim_nuvb[0], y_lim_nuvb[1], n_bins)
kernal_rad_width = (xedges[1] - xedges[0]) * kernal_std
kernal_rad_hight = (yedges[1] - yedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.5, 2.8), width=kernal_rad_width, height=kernal_rad_hight, angle=0, edgecolor='r', fc='None', lw=2)
ax[0, 1].add_patch(ellipse)
ax[0, 1].text(-0.4, 2.8, 'Smoothing', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


ax[0, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.23,
              'N=%i' % (sum(mask_class_1_hum * mask_good_colors_nuvbvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.15,
              'N=%i' % (sum(mask_class_2_hum * mask_good_colors_nuvbvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'N=%i' % (sum(mask_class_3_hum * mask_good_colors_nuvbvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_1_hum * mask_good_colors_ubvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_2_hum * mask_good_colors_ubvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_3_hum * mask_good_colors_ubvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[2, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_1_hum * mask_good_colors_bvvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_2_hum * mask_good_colors_bvvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_3_hum * mask_good_colors_bvvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[0, 0].legend(frameon=False, loc=3, #bbox_to_anchor=(0, 0.05),
                fontsize=fontsize-2)

ax[0, 0].set_title('Class 1 (Human) \n Symmetric Clusters', fontsize=fontsize)
ax[0, 1].set_title('Class 2 (Human) \n Asymmetric Clusters', fontsize=fontsize)
ax[0, 2].set_title('Class 3 (Human) \n Compact Associations', fontsize=fontsize)

ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')',labelpad=27, fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')',labelpad=27, fontsize=fontsize)
ax[2, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)



ax[0, 0].set_xlim(x_lim_vi)
ax[0, 1].set_xlim(x_lim_vi)
ax[0, 2].set_xlim(x_lim_vi)
ax[0, 0].set_ylim(y_lim_nuvb)
ax[0, 1].set_ylim(y_lim_nuvb)
ax[0, 2].set_ylim(y_lim_nuvb)

ax[1, 0].set_xlim(x_lim_vi)
ax[1, 1].set_xlim(x_lim_vi)
ax[1, 2].set_xlim(x_lim_vi)
ax[1, 0].set_ylim(y_lim_ub)
ax[1, 1].set_ylim(y_lim_ub)
ax[1, 2].set_ylim(y_lim_ub)

ax[2, 0].set_xlim(x_lim_vi)
ax[2, 1].set_xlim(x_lim_vi)
ax[2, 2].set_xlim(x_lim_vi)
ax[2, 0].set_ylim(y_lim_bv)
ax[2, 1].set_ylim(y_lim_bv)
ax[2, 2].set_ylim(y_lim_bv)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



fig.subplots_adjust(left=0.06, bottom=0.04, right=0.995, top=0.96, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/cc_compare.png')
fig.savefig('plot_output/cc_compare.pdf')
fig.clf()
plt.close()

exit()

