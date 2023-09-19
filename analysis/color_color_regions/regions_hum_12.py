import numpy as np
import matplotlib.pyplot as plt
from photometry_tools import helper_func
from scipy.spatial import ConvexHull


age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')

color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
detect_u_hum = np.load('../color_color/data_output/detect_u_hum.npy')
detect_b_hum = np.load('../color_color/data_output/detect_b_hum.npy')
detect_v_hum = np.load('../color_color/data_output/detect_v_hum.npy')
detect_i_hum = np.load('../color_color/data_output/detect_i_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')

color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
detect_u_ml = np.load('../color_color/data_output/detect_u_ml.npy')
detect_b_ml = np.load('../color_color/data_output/detect_b_ml.npy')
detect_v_ml = np.load('../color_color/data_output/detect_v_ml.npy')
detect_i_ml = np.load('../color_color/data_output/detect_i_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')

# color range limitations
x_lim_vi = (-0.6, 1.9)
y_lim_ub = (0.9, -2.2)

mask_class_12_hum = (clcl_color_hum == 1) | (clcl_color_hum == 2)
mask_class_12_ml = (clcl_color_ml == 1) | (clcl_color_ml == 2)

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                             (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                            (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                            mask_detect_ubvi_ml)


# get gauss und segmentations
n_bins_ubvi = 90
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_hum_12 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data=color_ub_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_data_err=color_vi_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              y_data_err=color_ub_err_hum[mask_class_12_hum * mask_good_colors_ubvi_hum],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)

gauss_map_hum_12 = gauss_dict_ubvi_hum_12['gauss_map']
seg_map_hum_12 = gauss_dict_ubvi_hum_12['seg_deb_map']
no_seg_map_hum_12 = gauss_map_hum_12.copy()
young_map_hum_12 = gauss_map_hum_12.copy()
cascade_map_hum_12 = gauss_map_hum_12.copy()
gc_map_hum_12 = gauss_map_hum_12.copy()
no_seg_map_hum_12[seg_map_hum_12._data != 0] = np.nan
young_map_hum_12[seg_map_hum_12._data != 1] = np.nan
cascade_map_hum_12[seg_map_hum_12._data != 2] = np.nan
gc_map_hum_12[seg_map_hum_12._data != 3] = np.nan


figure = plt.figure(figsize=(20, 22))
fontsize = 38

ax_cc_reg_hum = figure.add_axes([0.1, 0.08, 0.88, 0.88])

vmax = np.nanmax(gauss_map_hum_12)
ax_cc_reg_hum.imshow(no_seg_map_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(young_map_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Blues', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(cascade_map_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Greens', vmin=0+vmax/10, vmax=vmax/1.1)
ax_cc_reg_hum.imshow(gc_map_hum_12, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                     interpolation='nearest', aspect='auto', cmap='Reds', vmin=0+vmax/10, vmax=vmax/1.2)


ax_cc_reg_hum.scatter([], [], color='white', label=r'N = %i' % (sum(mask_class_12_hum)))



vi_int = 1.2
ub_int = -1.6
av_value = 1

helper_func.plot_reddening_vect(ax=ax_cc_reg_hum, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                                x_text_offset=0.00, y_text_offset=-0.05,
                                linewidth=6, line_color='k', text=True, fontsize=fontsize)

ax_cc_reg_hum.text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
              'Class 1|2 (Human)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

ax_cc_reg_hum.set_title('The PHANGS-HST Bright Star Cluster Sample', fontsize=fontsize)

ax_cc_reg_hum.set_xlim(x_lim_vi)
ax_cc_reg_hum.set_ylim(y_lim_ub)

ax_cc_reg_hum.plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=6, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_cc_reg_hum.plot(model_vi_sol50[age_mod_sol50>500], model_ub_sol50[age_mod_sol50>500],
                   color='k', linewidth=6, linestyle=':', label=r'BC03, Z$_{\odot}/50$ (> 500 Myr)')


age_dots_sol=[1, 5, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000, 11000, 12000, 13000, 13750]
age_dots_sol50=[500,  1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000, 11000, 12000, 13000, 13750]

for age in age_dots_sol:
  ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1], color='darkred', s=200, zorder=30)

for age in age_dots_sol50:
  ax_cc_reg_hum.scatter(model_vi_sol50[age_mod_sol50 == 1], model_ub_sol50[age_mod_sol50 == 1], color='darkred', s=200, zorder=30)

# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1], color='darkred', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 5], model_ub_sol[age_mod_sol == 5], color='darkred', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 10], model_ub_sol[age_mod_sol == 10], color='darkred', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 100], model_ub_sol[age_mod_sol == 100], color='darkred', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 500], model_ub_sol[age_mod_sol == 500], color='darkred', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol[age_mod_sol == 13750], model_ub_sol[age_mod_sol == 13750], color='darkred', s=200, zorder=30)
#
# ax_cc_reg_hum.scatter(model_vi_sol50[age_mod_sol50 == 500], model_ub_sol50[age_mod_sol50 == 500], color='k', s=200, zorder=30)
# ax_cc_reg_hum.scatter(model_vi_sol50[age_mod_sol50 == 13750], model_ub_sol50[age_mod_sol50 == 13750], color='k', s=200, zorder=30)

ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1]-0.1, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 5]+0.1, model_ub_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 10]+0.03, model_ub_sol[age_mod_sol == 10]-0.07, r'10 Myr',
                   horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 100]-0.05, model_ub_sol[age_mod_sol == 100]+0.1, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 500]-0.05, model_ub_sol[age_mod_sol == 500]+0.1, r'500 Myr',
                   horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax_cc_reg_hum.text(model_vi_sol[age_mod_sol == 13750], model_ub_sol[age_mod_sol == 13750]+0.05, r'13 Gyr',
                   horizontalalignment='center', verticalalignment='top', fontsize=fontsize)



ax_cc_reg_hum.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_cc_reg_hum.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_reg_hum.legend(frameon=False, loc=3, fontsize=fontsize)

ax_cc_reg_hum.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.tight_layout()
plt.savefig('plot_output/reg_hum_c12.png')
plt.savefig('plot_output/reg_hum_c12.pdf')


