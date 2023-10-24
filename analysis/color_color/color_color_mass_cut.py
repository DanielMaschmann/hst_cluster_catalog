import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from photometry_tools import helper_func
from matplotlib import patheffects

nuvb_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
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
mass_hum = np.load('data_output/mass_hum.npy')
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
mass_ml = np.load('data_output/mass_ml.npy')
mag_mask_ml = np.load('data_output/mag_mask_ml.npy')

mask_class_hum_12 = (clcl_color_hum == 1) | (clcl_color_hum == 2)
mask_class_ml_12 = (clcl_color_ml == 1) | (clcl_color_ml == 2)


mask_detect_nuvbvi_ml = detect_nuv_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_bvvi_ml = detect_b_ml * detect_v_ml * detect_i_ml

x_lim_vi = (-0.7, 2.4)
y_lim_ub = (2.1, -2.2)

n_bins = 190
kernal_std = 3.0

mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_ml)

mask_mass_1_ml = mass_ml > 1e4
mask_mass_2_ml = (mass_ml < 1e4) & (mass_ml > 5000)
mask_mass_3_ml = mass_ml < 5000

mask_mass_1_hum = mass_hum > 1e4
mask_mass_2_hum = (mass_hum < 1e4) & (mass_hum > 5000)
mask_mass_3_hum = mass_hum < 5000

# print('HUM C1 + C2 + C3 ', sum(mass_hum < 1e4), ' of ',  len(mass_hum), ' have masses < 1e4 M_sol. in fraction this is: ', sum(mass_hum < 1e4) / len(mass_hum))
# print('ML C1 + C2 + C3 ', sum(mass_ml < 1e4), ' of ',  len(mass_ml), ' have masses < 1e4 M_sol. in fraction this is: ', sum(mass_ml < 1e4) / len(mass_ml))
#
# print('HUM C1 + C2 ', sum((mass_hum < 1e4)*mask_class_hum_12), ' of ', sum(mask_class_hum_12), ' have masses < 1e4 M_sol. in fraction this is: ', sum((mass_hum < 1e4)*mask_class_hum_12) / sum(mask_class_hum_12))
# print('ML C1 + C2 ', sum((mass_ml < 1e4)*mask_class_ml_12), ' of ', sum(mask_class_ml_12), ' have masses < 1e4 M_sol. in fraction this is: ', sum((mass_ml < 1e4)*mask_class_ml_12) / sum(mask_class_ml_12))
# exit()

# print('HUM C1 + C2 + C3 ', sum(mass_hum < 1e5), ' of ',  len(mass_hum), ' have masses < 1e5 M_sol. in fraction this is: ', sum(mass_hum < 1e5) / len(mass_hum))
# print('ML C1 + C2 + C3 ', sum(mass_ml < 1e5), ' of ',  len(mass_ml), ' have masses < 1e5 M_sol. in fraction this is: ', sum(mass_ml < 1e5) / len(mass_ml))
#
# print('HUM C1 + C2 ', sum((mass_hum < 1e5)*mask_class_hum_12), ' of ', sum(mask_class_hum_12), ' have masses < 1e5 M_sol. in fraction this is: ', sum((mass_hum < 1e5)*mask_class_hum_12) / sum(mask_class_hum_12))
# print('ML C1 + C2 ', sum((mass_ml < 1e5)*mask_class_ml_12), ' of ', sum(mask_class_ml_12), ' have masses < 1e5 M_sol. in fraction this is: ', sum((mass_ml < 1e5)*mask_class_ml_12) / sum(mask_class_ml_12))
# exit()

fig, ax = plt.subplots(ncols=3, nrows=1, sharex='all', sharey='row', figsize=(22, 8))
fontsize = 23

helper_func.density_with_points(ax=ax[0], x=color_vi_ml[mask_class_ml_12 * mask_mass_1_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_ml_12 * mask_mass_1_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std)
helper_func.density_with_points(ax=ax[1], x=color_vi_ml[mask_class_ml_12 * mask_mass_2_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_ml_12 * mask_mass_2_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std)
helper_func.density_with_points(ax=ax[2], x=color_vi_ml[mask_class_ml_12 * mask_mass_3_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_ml_12 * mask_mass_3_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std)

display_models(ax=ax[0], y_color='ub', age_labels=True, age_label_fontsize=fontsize)
display_models(ax=ax[1], y_color='ub')
display_models(ax=ax[2], y_color='ub', label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')

vi_int = 1.5
ub_int = -1.3
av_value = 1

helper_func.plot_reddening_vect(ax=ax[0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=True, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)
helper_func.plot_reddening_vect(ax=ax[1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=False, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)
helper_func.plot_reddening_vect(ax=ax[2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=False, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)

xedges = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins)
yedges = np.linspace(y_lim_ub[0], y_lim_ub[1], n_bins)
kernal_rad_width = (xedges[1] - xedges[0]) * kernal_std
kernal_rad_hight = (yedges[1] - yedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.55, 1.85), width=kernal_rad_width, height=kernal_rad_hight, angle=0, edgecolor='r', fc='None', lw=2)
ax[1].add_patch(ellipse)
ax[1].text(-0.45, 1.85, 'Smoothing kernel', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[0].text(-0.6, 1.85, 'N=%i' % (sum(mask_class_ml_12 * mask_mass_1_ml)), horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1].text(-0.6, 1.55, 'N=%i' % (sum(mask_class_ml_12 * mask_mass_2_ml)), horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2].text(-0.6, 1.2, 'N=%i' % (sum(mask_class_ml_12 * mask_mass_3_ml)), horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

pe = [patheffects.withStroke(linewidth=3, foreground="w")]
ax[0].text(0.02, 0.95, 'a)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[0].transAxes, path_effects=pe)
ax[1].text(0.02, 0.95, 'b)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[1].transAxes, path_effects=pe)
ax[2].text(0.02, 0.95, 'c)', horizontalalignment='left', verticalalignment='center', fontsize=fontsize,
              transform=ax[2].transAxes, path_effects=pe)

ax[0].text(0.95, 0.95, r'M$_{*}$ > 10$^{4} {\rm M_{\odot}}$', horizontalalignment='right', verticalalignment='center', fontsize=fontsize,
              transform=ax[0].transAxes, path_effects=pe)
ax[1].text(0.95, 0.95, r'5 x 10$^{3} {\rm M_{\odot}}$ < M$_{*}$ < 10$^{4} {\rm M_{\odot}}$', horizontalalignment='right', verticalalignment='center', fontsize=fontsize,
              transform=ax[1].transAxes, path_effects=pe)
ax[2].text(0.95, 0.95, r'M$_{*}$ < 5 x 10$^{3} {\rm M_{\odot}}$', horizontalalignment='right', verticalalignment='center', fontsize=fontsize,
              transform=ax[2].transAxes, path_effects=pe)


ax[0].set_title(r'Class 1+2 Clusters (ML)', fontsize=fontsize)
ax[1].set_title(r'Class 1+2 Clusters (ML)', fontsize=fontsize)
ax[2].set_title(r'Class 1+2 Clusters (ML)', fontsize=fontsize)

ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[2].legend(frameon=False, loc=3, fontsize=fontsize)

ax[0].set_xlim(x_lim_vi)
ax[0].set_ylim(y_lim_ub)


ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


fig.subplots_adjust(left=0.06, bottom=0.09, right=0.995, top=0.95, wspace=0.01, hspace=0.01)
# plt.tight_layout()
fig.savefig('plot_output/cc_mass_cut.png')
fig.savefig('plot_output/cc_mass_cut.pdf')
fig.clf()
plt.close()
