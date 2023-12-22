import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colorbar import ColorbarBase
from photometry_tools import data_access
from photometry_tools import helper_func as hf

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
                   color_sol='darkred', linewidth_sol=4, linestyle_sol='-', color_arrow_sol='darkcyan', arrow_linestyle_sol='--',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-', color_arrow_sol50='darkviolet', arrow_linestyle_sol50='--',
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

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')




age_map_hum_1 = np.load('data_output/age_map_hum_1.npy')
age_map_hum_2 = np.load('data_output/age_map_hum_2.npy')
age_map_hum_3 = np.load('data_output/age_map_hum_3.npy')
age_map_ml_1 = np.load('data_output/age_map_ml_1.npy')
age_map_ml_2 = np.load('data_output/age_map_ml_2.npy')
age_map_ml_3 = np.load('data_output/age_map_ml_3.npy')

ebv_map_hum_1 = np.load('data_output/ebv_map_hum_1.npy')
ebv_map_hum_2 = np.load('data_output/ebv_map_hum_2.npy')
ebv_map_hum_3 = np.load('data_output/ebv_map_hum_3.npy')
ebv_map_ml_1 = np.load('data_output/ebv_map_ml_1.npy')
ebv_map_ml_2 = np.load('data_output/ebv_map_ml_2.npy')
ebv_map_ml_3 = np.load('data_output/ebv_map_ml_3.npy')

age_map_hum_1_ir4 = np.load('data_output/age_map_hum_1_ir4.npy')
age_map_hum_2_ir4 = np.load('data_output/age_map_hum_2_ir4.npy')
age_map_hum_3_ir4 = np.load('data_output/age_map_hum_3_ir4.npy')
age_map_ml_1_ir4 = np.load('data_output/age_map_ml_1_ir4.npy')
age_map_ml_2_ir4 = np.load('data_output/age_map_ml_2_ir4.npy')
age_map_ml_3_ir4 = np.load('data_output/age_map_ml_3_ir4.npy')

ebv_map_hum_1_ir4 = np.load('data_output/ebv_map_hum_1_ir4.npy')
ebv_map_hum_2_ir4 = np.load('data_output/ebv_map_hum_2_ir4.npy')
ebv_map_hum_3_ir4 = np.load('data_output/ebv_map_hum_3_ir4.npy')
ebv_map_ml_1_ir4 = np.load('data_output/ebv_map_ml_1_ir4.npy')
ebv_map_ml_2_ir4 = np.load('data_output/ebv_map_ml_2_ir4.npy')
ebv_map_ml_3_ir4 = np.load('data_output/ebv_map_ml_3_ir4.npy')


vi_int = 1.2
ub_int = -1.5
av_value = 1

x_lim_vi = (-0.6, 1.9)
y_lim_ub = (0.8, -1.9)

n_bins_ubvi = 50

threshold_hum = 5
threshold_ml = 5

cmap_age = matplotlib.cm.get_cmap('rainbow')
norm_age = matplotlib.colors.Normalize(vmin=6, vmax=10.5)



figure = plt.figure(figsize=(30, 20))
fontsize = 35

panel_width = 0.28
panel_hight = 0.43
ax_age_hum_1 = figure.add_axes([0.065, 0.525, panel_width, panel_hight])
ax_age_ml_1 = figure.add_axes([0.065, 0.06, panel_width, panel_hight])

ax_age_hum_2 = figure.add_axes([0.35, 0.525, panel_width, panel_hight])
ax_age_ml_2 = figure.add_axes([0.35, 0.06, panel_width, panel_hight])

ax_age_hum_3 = figure.add_axes([0.635, 0.525, panel_width, panel_hight])
ax_age_ml_3 = figure.add_axes([0.635, 0.06, panel_width, panel_hight])

ax_cbar_age = figure.add_axes([0.92, 0.2, 0.015, 0.6])


ax_age_hum_1.imshow(age_map_hum_1.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_hum_2.imshow(age_map_hum_2.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_hum_3.imshow(age_map_hum_3.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)

ax_age_ml_1.imshow(age_map_ml_1.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_ml_2.imshow(age_map_ml_2.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_ml_3.imshow(age_map_ml_3.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)



display_models(ax=ax_age_hum_1, y_color='ub', age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_age_hum_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_hum_3, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_1, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_3, y_color='ub', age_label_fontsize=fontsize+2)

hf.plot_reddening_vect(ax=ax_age_hum_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.1)
hf.plot_reddening_vect(ax=ax_age_hum_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_age_hum_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax_age_ml_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax_age_ml_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_age_ml_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ColorbarBase(ax_cbar_age, orientation='vertical', cmap=cmap_age, norm=norm_age, extend='neither', ticks=None)
ax_cbar_age.set_ylabel(r'log(Age)', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_age.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

pe = [patheffects.withStroke(linewidth=3, foreground="w")]


ax_age_hum_1.set_title('Cluster Class 1 (Hum)', fontsize=fontsize)
ax_age_hum_2.set_title('Cluster Class 2 (Hum)', fontsize=fontsize)
ax_age_hum_3.set_title('Compact Associations (Hum)', fontsize=fontsize)
ax_age_ml_1.set_title('Cluster Class 1 (ML)', fontsize=fontsize)
ax_age_ml_2.set_title('Cluster Class 2 (ML)', fontsize=fontsize)
ax_age_ml_3.set_title('Compact Associations (ML)', fontsize=fontsize)


ax_age_hum_1.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_1.transAxes, path_effects=pe)
ax_age_hum_2.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_2.transAxes, path_effects=pe)
ax_age_hum_3.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_3.transAxes, path_effects=pe)
ax_age_ml_1.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_1.transAxes, path_effects=pe)
ax_age_ml_2.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_2.transAxes, path_effects=pe)
ax_age_ml_3.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_3.transAxes, path_effects=pe)

ax_age_hum_1.set_xlim(x_lim_vi)
ax_age_hum_2.set_xlim(x_lim_vi)
ax_age_hum_3.set_xlim(x_lim_vi)
ax_age_ml_1.set_xlim(x_lim_vi)
ax_age_ml_2.set_xlim(x_lim_vi)
ax_age_ml_3.set_xlim(x_lim_vi)

ax_age_hum_1.set_ylim(y_lim_ub)
ax_age_hum_2.set_ylim(y_lim_ub)
ax_age_hum_3.set_ylim(y_lim_ub)
ax_age_ml_1.set_ylim(y_lim_ub)
ax_age_ml_2.set_ylim(y_lim_ub)
ax_age_ml_3.set_ylim(y_lim_ub)

ax_age_hum_2.set_yticklabels([])
ax_age_hum_3.set_yticklabels([])
ax_age_ml_2.set_yticklabels([])
ax_age_ml_3.set_yticklabels([])

ax_age_hum_1.set_xticklabels([])
ax_age_hum_2.set_xticklabels([])
ax_age_hum_3.set_xticklabels([])


ax_age_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_age_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_age_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_age_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_age_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_age_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

ax_age_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
plt.savefig('plot_output/color_color_age_sed_fix.png')
plt.savefig('plot_output/color_color_age_sed_fix.pdf')
plt.clf()
plt.cla()



figure = plt.figure(figsize=(30, 20))
fontsize = 35

panel_width = 0.28
panel_hight = 0.43
ax_age_hum_1 = figure.add_axes([0.065, 0.525, panel_width, panel_hight])
ax_age_ml_1 = figure.add_axes([0.065, 0.06, panel_width, panel_hight])

ax_age_hum_2 = figure.add_axes([0.35, 0.525, panel_width, panel_hight])
ax_age_ml_2 = figure.add_axes([0.35, 0.06, panel_width, panel_hight])

ax_age_hum_3 = figure.add_axes([0.635, 0.525, panel_width, panel_hight])
ax_age_ml_3 = figure.add_axes([0.635, 0.06, panel_width, panel_hight])

ax_cbar_age = figure.add_axes([0.92, 0.2, 0.015, 0.6])


ax_age_hum_1.imshow(age_map_hum_1_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_hum_2.imshow(age_map_hum_2_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_hum_3.imshow(age_map_hum_3_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)

ax_age_ml_1.imshow(age_map_ml_1_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_ml_2.imshow(age_map_ml_2_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)
ax_age_ml_3.imshow(age_map_ml_3_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_age, norm=norm_age)



display_models(ax=ax_age_hum_1, y_color='ub', age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_age_hum_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_hum_3, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_1, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_age_ml_3, y_color='ub', age_label_fontsize=fontsize+2)

hf.plot_reddening_vect(ax=ax_age_hum_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.1)
hf.plot_reddening_vect(ax=ax_age_hum_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_age_hum_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax_age_ml_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax_age_ml_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_age_ml_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ColorbarBase(ax_cbar_age, orientation='vertical', cmap=cmap_age, norm=norm_age, extend='neither', ticks=None)
ax_cbar_age.set_ylabel(r'log(Age)', labelpad=-4, fontsize=fontsize + 5)
ax_cbar_age.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

pe = [patheffects.withStroke(linewidth=3, foreground="w")]

ax_age_hum_1.set_title('Cluster Class 1 (Hum)', fontsize=fontsize)
ax_age_hum_2.set_title('Cluster Class 2 (Hum)', fontsize=fontsize)
ax_age_hum_3.set_title('Compact Associations (Hum)', fontsize=fontsize)
ax_age_ml_1.set_title('Cluster Class 1 (ML)', fontsize=fontsize)
ax_age_ml_2.set_title('Cluster Class 2 (ML)', fontsize=fontsize)
ax_age_ml_3.set_title('Compact Associations (ML)', fontsize=fontsize)


ax_age_hum_1.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_1.transAxes, path_effects=pe)
ax_age_hum_2.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_2.transAxes, path_effects=pe)
ax_age_hum_3.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_age_hum_3.transAxes, path_effects=pe)
ax_age_ml_1.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_1.transAxes, path_effects=pe)
ax_age_ml_2.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_2.transAxes, path_effects=pe)
ax_age_ml_3.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_age_ml_3.transAxes, path_effects=pe)

ax_age_hum_1.set_xlim(x_lim_vi)
ax_age_hum_2.set_xlim(x_lim_vi)
ax_age_hum_3.set_xlim(x_lim_vi)
ax_age_ml_1.set_xlim(x_lim_vi)
ax_age_ml_2.set_xlim(x_lim_vi)
ax_age_ml_3.set_xlim(x_lim_vi)

ax_age_hum_1.set_ylim(y_lim_ub)
ax_age_hum_2.set_ylim(y_lim_ub)
ax_age_hum_3.set_ylim(y_lim_ub)
ax_age_ml_1.set_ylim(y_lim_ub)
ax_age_ml_2.set_ylim(y_lim_ub)
ax_age_ml_3.set_ylim(y_lim_ub)

ax_age_hum_2.set_yticklabels([])
ax_age_hum_3.set_yticklabels([])
ax_age_ml_2.set_yticklabels([])
ax_age_ml_3.set_yticklabels([])

ax_age_hum_1.set_xticklabels([])
ax_age_hum_2.set_xticklabels([])
ax_age_hum_3.set_xticklabels([])


ax_age_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_age_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_age_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_age_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_age_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_age_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

ax_age_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_age_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
plt.savefig('plot_output/color_color_age_ir4.png')
plt.savefig('plot_output/color_color_age_ir4.pdf')
plt.clf()
plt.cla()


#######################################
## EBV

cmap_ebv = matplotlib.cm.get_cmap('rainbow')
norm_ebv = matplotlib.colors.Normalize(vmin=0, vmax=0.5)

figure = plt.figure(figsize=(30, 20))
fontsize = 35

panel_width = 0.28
panel_hight = 0.43
ax_ebv_hum_1 = figure.add_axes([0.065, 0.525, panel_width, panel_hight])
ax_ebv_ml_1 = figure.add_axes([0.065, 0.06, panel_width, panel_hight])

ax_ebv_hum_2 = figure.add_axes([0.35, 0.525, panel_width, panel_hight])
ax_ebv_ml_2 = figure.add_axes([0.35, 0.06, panel_width, panel_hight])

ax_ebv_hum_3 = figure.add_axes([0.635, 0.525, panel_width, panel_hight])
ax_ebv_ml_3 = figure.add_axes([0.635, 0.06, panel_width, panel_hight])

ax_cbar_ebv = figure.add_axes([0.92, 0.2, 0.015, 0.6])


ax_ebv_hum_1.imshow(ebv_map_hum_1.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_2.imshow(ebv_map_hum_2.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_3.imshow(ebv_map_hum_3.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)

ax_ebv_ml_1.imshow(ebv_map_ml_1.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_2.imshow(ebv_map_ml_2.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_3.imshow(ebv_map_ml_3.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)



display_models(ax=ax_ebv_hum_1, y_color='ub', age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_ebv_hum_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_hum_3, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_1, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_3, y_color='ub', age_label_fontsize=fontsize+2)

hf.plot_reddening_vect(ax=ax_ebv_hum_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.1)
hf.plot_reddening_vect(ax=ax_ebv_hum_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_ebv_hum_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax_ebv_ml_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax_ebv_ml_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_ebv_ml_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ColorbarBase(ax_cbar_ebv, orientation='vertical', cmap=cmap_ebv, norm=norm_ebv, extend='neither', ticks=None)
ax_cbar_ebv.set_ylabel(r'E(B-V) [mag]', labelpad=10, fontsize=fontsize + 5)
ax_cbar_ebv.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

pe = [patheffects.withStroke(linewidth=3, foreground="w")]


ax_ebv_hum_1.set_title('Cluster Class 1 (Hum)', fontsize=fontsize)
ax_ebv_hum_2.set_title('Cluster Class 2 (Hum)', fontsize=fontsize)
ax_ebv_hum_3.set_title('Compact Associations (Hum)', fontsize=fontsize)
ax_ebv_ml_1.set_title('Cluster Class 1 (ML)', fontsize=fontsize)
ax_ebv_ml_2.set_title('Cluster Class 2 (ML)', fontsize=fontsize)
ax_ebv_ml_3.set_title('Compact Associations (ML)', fontsize=fontsize)


ax_ebv_hum_1.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_1.transAxes, path_effects=pe)
ax_ebv_hum_2.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_2.transAxes, path_effects=pe)
ax_ebv_hum_3.text(0.03, 0.97, 'SED fix',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_3.transAxes, path_effects=pe)
ax_ebv_ml_1.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_1.transAxes, path_effects=pe)
ax_ebv_ml_2.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_2.transAxes, path_effects=pe)
ax_ebv_ml_3.text(0.03, 0.97, 'SED fix',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_3.transAxes, path_effects=pe)

ax_ebv_hum_1.set_xlim(x_lim_vi)
ax_ebv_hum_2.set_xlim(x_lim_vi)
ax_ebv_hum_3.set_xlim(x_lim_vi)
ax_ebv_ml_1.set_xlim(x_lim_vi)
ax_ebv_ml_2.set_xlim(x_lim_vi)
ax_ebv_ml_3.set_xlim(x_lim_vi)

ax_ebv_hum_1.set_ylim(y_lim_ub)
ax_ebv_hum_2.set_ylim(y_lim_ub)
ax_ebv_hum_3.set_ylim(y_lim_ub)
ax_ebv_ml_1.set_ylim(y_lim_ub)
ax_ebv_ml_2.set_ylim(y_lim_ub)
ax_ebv_ml_3.set_ylim(y_lim_ub)

ax_ebv_hum_2.set_yticklabels([])
ax_ebv_hum_3.set_yticklabels([])
ax_ebv_ml_2.set_yticklabels([])
ax_ebv_ml_3.set_yticklabels([])

ax_ebv_hum_1.set_xticklabels([])
ax_ebv_hum_2.set_xticklabels([])
ax_ebv_hum_3.set_xticklabels([])


ax_ebv_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_ebv_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_ebv_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_ebv_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

ax_ebv_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
plt.savefig('plot_output/color_color_ebv_sed_fix.png')
plt.savefig('plot_output/color_color_ebv_sed_fix.pdf')
plt.clf()
plt.cla()



figure = plt.figure(figsize=(30, 20))
fontsize = 35

panel_width = 0.28
panel_hight = 0.43
ax_ebv_hum_1 = figure.add_axes([0.065, 0.525, panel_width, panel_hight])
ax_ebv_ml_1 = figure.add_axes([0.065, 0.06, panel_width, panel_hight])

ax_ebv_hum_2 = figure.add_axes([0.35, 0.525, panel_width, panel_hight])
ax_ebv_ml_2 = figure.add_axes([0.35, 0.06, panel_width, panel_hight])

ax_ebv_hum_3 = figure.add_axes([0.635, 0.525, panel_width, panel_hight])
ax_ebv_ml_3 = figure.add_axes([0.635, 0.06, panel_width, panel_hight])

ax_cbar_ebv = figure.add_axes([0.92, 0.2, 0.015, 0.6])


ax_ebv_hum_1.imshow(ebv_map_hum_1_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_2.imshow(ebv_map_hum_2_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_hum_3.imshow(ebv_map_hum_3_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                    interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)

ax_ebv_ml_1.imshow(ebv_map_ml_1_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_2.imshow(ebv_map_ml_2_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)
ax_ebv_ml_3.imshow(ebv_map_ml_3_ir4.T, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                   interpolation='nearest', aspect='auto', cmap=cmap_ebv, norm=norm_ebv)



display_models(ax=ax_ebv_hum_1, y_color='ub', age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax_ebv_hum_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_hum_3, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_1, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_2, y_color='ub', age_label_fontsize=fontsize+2)
display_models(ax=ax_ebv_ml_3, y_color='ub', age_label_fontsize=fontsize+2)

hf.plot_reddening_vect(ax=ax_ebv_hum_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.0, y_text_offset=-0.1)
hf.plot_reddening_vect(ax=ax_ebv_hum_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_ebv_hum_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

hf.plot_reddening_vect(ax=ax_ebv_ml_1, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax_ebv_ml_2, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax_ebv_ml_3, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)


ColorbarBase(ax_cbar_ebv, orientation='vertical', cmap=cmap_ebv, norm=norm_ebv, extend='neither', ticks=None)
ax_cbar_ebv.set_ylabel(r'E(B-V) [mag]', labelpad=10, fontsize=fontsize + 5)
ax_cbar_ebv.tick_params(axis='both', which='both', width=2, direction='in', right=True, labelright=True, labelsize=fontsize)

pe = [patheffects.withStroke(linewidth=3, foreground="w")]

ax_ebv_hum_1.set_title('Cluster Class 1 (Hum)', fontsize=fontsize)
ax_ebv_hum_2.set_title('Cluster Class 2 (Hum)', fontsize=fontsize)
ax_ebv_hum_3.set_title('Compact Associations (Hum)', fontsize=fontsize)
ax_ebv_ml_1.set_title('Cluster Class 1 (ML)', fontsize=fontsize)
ax_ebv_ml_2.set_title('Cluster Class 2 (ML)', fontsize=fontsize)
ax_ebv_ml_3.set_title('Compact Associations (ML)', fontsize=fontsize)


ax_ebv_hum_1.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_1.transAxes, path_effects=pe)
ax_ebv_hum_2.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_2.transAxes, path_effects=pe)
ax_ebv_hum_3.text(0.03, 0.97, 'Standard SED fit',
                  horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                  transform=ax_ebv_hum_3.transAxes, path_effects=pe)
ax_ebv_ml_1.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_1.transAxes, path_effects=pe)
ax_ebv_ml_2.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_2.transAxes, path_effects=pe)
ax_ebv_ml_3.text(0.03, 0.97, 'Standard SED fit',
                 horizontalalignment='left', verticalalignment='top', fontsize=fontsize,
                 transform=ax_ebv_ml_3.transAxes, path_effects=pe)

ax_ebv_hum_1.set_xlim(x_lim_vi)
ax_ebv_hum_2.set_xlim(x_lim_vi)
ax_ebv_hum_3.set_xlim(x_lim_vi)
ax_ebv_ml_1.set_xlim(x_lim_vi)
ax_ebv_ml_2.set_xlim(x_lim_vi)
ax_ebv_ml_3.set_xlim(x_lim_vi)

ax_ebv_hum_1.set_ylim(y_lim_ub)
ax_ebv_hum_2.set_ylim(y_lim_ub)
ax_ebv_hum_3.set_ylim(y_lim_ub)
ax_ebv_ml_1.set_ylim(y_lim_ub)
ax_ebv_ml_2.set_ylim(y_lim_ub)
ax_ebv_ml_3.set_ylim(y_lim_ub)

ax_ebv_hum_2.set_yticklabels([])
ax_ebv_hum_3.set_yticklabels([])
ax_ebv_ml_2.set_yticklabels([])
ax_ebv_ml_3.set_yticklabels([])

ax_ebv_hum_1.set_xticklabels([])
ax_ebv_hum_2.set_xticklabels([])
ax_ebv_hum_3.set_xticklabels([])


ax_ebv_hum_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_ebv_ml_1.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax_ebv_ml_1.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_2.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_ebv_ml_3.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_ebv_hum_1.legend(frameon=False, loc=3, fontsize=fontsize)

ax_ebv_hum_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_hum_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_1.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv_ml_3.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

plt.tight_layout()
plt.savefig('plot_output/color_color_ebv_ir4.png')
plt.savefig('plot_output/color_color_ebv_ir4.pdf')
plt.clf()
plt.cla()



