import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import dust_tools.extinction_tools
from photometry_tools import helper_func

age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('../color_color/data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')


color_vi_hum = np.load('../color_color/data_output/color_vi_hum.npy')
color_ub_hum = np.load('../color_color/data_output/color_ub_hum.npy')
color_bv_hum = np.load('../color_color/data_output/color_bv_hum.npy')
color_nuvu_hum = np.load('../color_color/data_output/color_nuvu_hum.npy')
color_nuvb_hum = np.load('../color_color/data_output/color_nuvb_hum.npy')
color_vi_err_hum = np.load('../color_color/data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('../color_color/data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('../color_color/data_output/color_bv_err_hum.npy')
color_nuvu_err_hum = np.load('../color_color/data_output/color_nuvu_err_hum.npy')
color_nuvb_err_hum = np.load('../color_color/data_output/color_nuvb_err_hum.npy')
detect_vi_hum = np.load('../color_color/data_output/detect_vi_hum.npy')
detect_ub_hum = np.load('../color_color/data_output/detect_ub_hum.npy')
detect_bv_hum = np.load('../color_color/data_output/detect_bv_hum.npy')
detect_nuvu_hum = np.load('../color_color/data_output/detect_nuvu_hum.npy')
detect_nuvb_hum = np.load('../color_color/data_output/detect_nuvb_hum.npy')
clcl_color_hum = np.load('../color_color/data_output/clcl_color_hum.npy')
age_hum = np.load('../color_color/data_output/age_hum.npy')
ebv_hum = np.load('../color_color/data_output/ebv_hum.npy')
color_vi_ml = np.load('../color_color/data_output/color_vi_ml.npy')
color_ub_ml = np.load('../color_color/data_output/color_ub_ml.npy')
color_bv_ml = np.load('../color_color/data_output/color_bv_ml.npy')
color_nuvu_ml = np.load('../color_color/data_output/color_nuvu_ml.npy')
color_nuvb_ml = np.load('../color_color/data_output/color_nuvb_ml.npy')
color_vi_err_ml = np.load('../color_color/data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('../color_color/data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('../color_color/data_output/color_bv_err_ml.npy')
color_nuvu_err_ml = np.load('../color_color/data_output/color_nuvu_err_ml.npy')
color_nuvb_err_ml = np.load('../color_color/data_output/color_nuvb_err_ml.npy')
detect_vi_ml = np.load('../color_color/data_output/detect_vi_ml.npy')
detect_ub_ml = np.load('../color_color/data_output/detect_ub_ml.npy')
detect_bv_ml = np.load('../color_color/data_output/detect_bv_ml.npy')
detect_nuvu_ml = np.load('../color_color/data_output/detect_nuvu_ml.npy')
detect_nuvb_ml = np.load('../color_color/data_output/detect_nuvb_ml.npy')
clcl_color_ml = np.load('../color_color/data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('../color_color/data_output/clcl_qual_color_ml.npy')
age_ml = np.load('../color_color/data_output/age_ml.npy')
ebv_ml = np.load('../color_color/data_output/ebv_ml.npy')
mag_mask_ml = np.load('../color_color/data_output/mag_mask_ml.npy')

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_hum = detect_vi_hum * detect_ub_hum
mask_detect_ubvi_ml = detect_vi_ml * detect_ub_ml

mask_detect_bvvi_hum = detect_vi_hum * detect_bv_hum
mask_detect_bvvi_ml = detect_vi_ml * detect_bv_ml

mask_good_colors_ubvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                             (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
mask_good_colors_ubvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                            (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml
mask_good_colors_bvvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                             (color_bv_hum > -1) & (color_bv_hum < 1.7)) * mask_detect_bvvi_hum
mask_good_colors_bvvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                            (color_bv_ml > -1) & (color_bv_ml < 1.7)) * mask_detect_bvvi_ml

# color range limitations
x_lim_ubvi = (-0.6, 1.9)
y_lim_ubvi = (0.8, -1.9)
n_bins_ubvi = 150
threshold_fact = 2
kernal_std = 8.0

gauss_dict_ubvi_hum_1 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 y_data=color_ub_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 x_data_err=color_vi_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 y_data_err=color_ub_err_hum[mask_class_1_hum * mask_good_colors_ubvi_hum],
                                 x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                 threshold_fact=threshold_fact, kernal_std=kernal_std)
gauss_dict_ubvi_ml_1 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                y_data=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                x_data_err=color_vi_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                y_data_err=color_ub_err_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                                x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                threshold_fact=threshold_fact, kernal_std=kernal_std)

gauss_dict_ubvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                 y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                 x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                 y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                 x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                 threshold_fact=threshold_fact, kernal_std=kernal_std)
gauss_dict_ubvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                threshold_fact=threshold_fact, kernal_std=kernal_std)

gauss_dict_ubvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                 y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                 x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                 y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                 x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                 threshold_fact=threshold_fact, kernal_std=kernal_std)
gauss_dict_ubvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
                                threshold_fact=threshold_fact, kernal_std=kernal_std)



fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(17.5, 12.6))
fontsize = 20


helper_func.plot_reg_map(ax=ax[0, 0], gauss_map=gauss_dict_ubvi_hum_1['gauss_map'], seg_map=gauss_dict_ubvi_hum_1['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             color_1='Greens', color_2='Reds',
             plot_cont_1=True, plot_cont_2=True, save_str_1='ubvi_hum_1_cascade', save_str_2='ubvi_hum_1_gc')
helper_func.plot_reg_map(ax=ax[0, 1], gauss_map=gauss_dict_ubvi_hum_2['gauss_map'], seg_map=gauss_dict_ubvi_hum_2['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             plot_cont_1=True, plot_cont_2=True, save_str_1='ubvi_hum_2_young', save_str_2='ubvi_hum_2_cascade')
helper_func.plot_reg_map(ax=ax[0, 2], gauss_map=gauss_dict_ubvi_hum_3['gauss_map'], seg_map=gauss_dict_ubvi_hum_3['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             plot_cont_1=True, save_str_1='ubvi_hum_3_young')

helper_func.plot_reg_map(ax=ax[1, 0], gauss_map=gauss_dict_ubvi_ml_1['gauss_map'], seg_map=gauss_dict_ubvi_ml_1['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             color_1='Greens', color_2='Reds',
             plot_cont_1=True, plot_cont_2=True, save_str_1='ubvi_ml_1_cascade', save_str_2='ubvi_ml_1_gc')
helper_func.plot_reg_map(ax=ax[1, 1], gauss_map=gauss_dict_ubvi_ml_2['gauss_map'], seg_map=gauss_dict_ubvi_ml_2['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             plot_cont_1=True, plot_cont_2=True, save_str_1='ubvi_ml_2_young', save_str_2='ubvi_ml_2_cascade')
helper_func.plot_reg_map(ax=ax[1, 2], gauss_map=gauss_dict_ubvi_ml_3['gauss_map'], seg_map=gauss_dict_ubvi_ml_3['seg_deb_map'],
             x_lim=x_lim_ubvi, y_lim=y_lim_ubvi, n_bins=n_bins_ubvi,
             plot_cont_1=True, save_str_1='ubvi_ml_3_young')



lin_fit_result_ubvi_c2_hum = helper_func.fit_line(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                     y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                     x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                     y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                     hull_str='ubvi_hum_2_young')
lin_fit_result_ubvi_c3_hum = helper_func.fit_line(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                     y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                     x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                     y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                     hull_str='ubvi_hum_3_young')
lin_fit_result_ubvi_c2_ml = helper_func.fit_line(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                     y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                     x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                     y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                     hull_str='ubvi_ml_2_young')
lin_fit_result_ubvi_c3_ml = helper_func.fit_line(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                     y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                     x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                     y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                     hull_str='ubvi_ml_3_young')

dummy_x_data = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], 100)
dummy_y_data_ubvi_c2_hum = helper_func.lin_func((lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c3_hum = helper_func.lin_func((lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c2_ml = helper_func.lin_func((lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c3_ml = helper_func.lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']), x=dummy_x_data)

ax[0, 1].plot(dummy_x_data, dummy_y_data_ubvi_c2_hum, color='k', linewidth=2, linestyle='--')
ax[0, 2].plot(dummy_x_data, dummy_y_data_ubvi_c3_hum, color='k', linewidth=2, linestyle='--')
ax[1, 1].plot(dummy_x_data, dummy_y_data_ubvi_c2_ml, color='k', linewidth=2, linestyle='--')
ax[1, 2].plot(dummy_x_data, dummy_y_data_ubvi_c3_ml, color='k', linewidth=2, linestyle='--')


x_text_pos = 1.0
text_anle_c2_hum = - np.arctan(lin_fit_result_ubvi_c2_hum['gradient']) * 180/np.pi
text_anle_c3_hum = - np.arctan(lin_fit_result_ubvi_c3_hum['gradient']) * 180/np.pi
text_anle_c2_ml = - np.arctan(lin_fit_result_ubvi_c2_ml['gradient']) * 180/np.pi
text_anle_c3_ml = - np.arctan(lin_fit_result_ubvi_c3_ml['gradient']) * 180/np.pi

ax[0, 1].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['intersect']),
                       x=x_text_pos)+0.12,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['gradient_err']),
              horizontalalignment='left', verticalalignment='center',
              rotation=text_anle_c2_hum, fontsize=fontsize - 5)
ax[0, 2].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['intersect']),
                       x=x_text_pos)+0.12,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['gradient_err']),
              horizontalalignment='left', verticalalignment='center',
              rotation=text_anle_c3_hum, fontsize=fontsize - 5)
ax[1, 1].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['intersect']),
                       x=x_text_pos)+0.12,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['gradient_err']),
              horizontalalignment='left', verticalalignment='center',
              rotation=text_anle_c2_ml, fontsize=fontsize - 5)
ax[1, 2].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']),
                       x=x_text_pos)+0.12,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['gradient_err']),
              horizontalalignment='left', verticalalignment='center',
              rotation=text_anle_c3_ml, fontsize=fontsize - 5)


x_bins = np.linspace(x_lim_ubvi[0], x_lim_ubvi[1], n_bins_ubvi)
kernal_rad = (x_bins[1] - x_bins[0]) * kernal_std
# plot_kernel_std
# ellipse = Ellipse(xy=(-0.35, 0.7), width=kernal_rad, height=kernal_rad, angle=0, edgecolor='r', fc='None', lw=2, zorder=10)
ellipse = Ellipse(xy=(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.09,
                      y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.07),
                  width=kernal_rad, height=kernal_rad, angle=0, edgecolor='r', fc='None', lw=2, zorder=10)

ax[0, 0].add_patch(ellipse)
# ax[0, 0].text(-0.2, 0.75, 'Smoothing kernel', horizontalalignment='left', verticalalignment='center', fontsize=fontsize, zorder=10)


ax[0, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}/50$')
ax[0, 0].plot(np.nan, np.nan, color='white', linewidth=2, linestyle='-', label=r'Smoothing kernel')

ax[0, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[0, 2].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[0, 2].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[1, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[1, 0].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[1, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[1, 1].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[1, 2].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[1, 2].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')



catalog_access = photometry_tools.data_access.CatalogAccess()
vi_int = 1.1
ub_int = -1.4
max_av = 1
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)

slope_av_vector = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector = - np.arctan(slope_av_vector) * 180/np.pi

print('slope_av_vector ', slope_av_vector)
print('angle_av_vector ', angle_av_vector)

ax[0, 0].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax[0, 1].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax[0, 2].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax[1, 0].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax[1, 1].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))
ax[1, 2].annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
            xytext=(vi_int, ub_int), fontsize=fontsize,
            textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))

ax[0, 0].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)
ax[0, 1].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)
ax[0, 2].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)

ax[1, 0].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)
ax[1, 1].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)
ax[1, 2].text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = 1', horizontalalignment='left', verticalalignment='bottom',
              rotation=angle_av_vector, fontsize=fontsize)


ax[0, 0].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[0, 1].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[0, 2].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Compact associations (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[1, 0].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 1 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[1, 1].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Class 2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
ax[1, 2].text(x_lim_ubvi[0] + (x_lim_ubvi[1]-x_lim_ubvi[0])*0.95, y_lim_ubvi[0] + (y_lim_ubvi[1]-y_lim_ubvi[0])*0.93,
              'Compact associations (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)


ax[0, 0].set_xlim(x_lim_ubvi)
ax[0, 1].set_xlim(x_lim_ubvi)
ax[0, 2].set_xlim(x_lim_ubvi)
ax[0, 0].set_ylim(y_lim_ubvi)
ax[0, 1].set_ylim(y_lim_ubvi)
ax[0, 2].set_ylim(y_lim_ubvi)

ax[1, 0].set_xlim(x_lim_ubvi)
ax[1, 1].set_xlim(x_lim_ubvi)
ax[1, 2].set_xlim(x_lim_ubvi)
ax[1, 0].set_ylim(y_lim_ubvi)
ax[1, 1].set_ylim(y_lim_ubvi)
ax[1, 2].set_ylim(y_lim_ubvi)

ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[0, 2].set_xticklabels([])

ax[0, 1].set_yticklabels([])
ax[1, 1].set_yticklabels([])
ax[0, 2].set_yticklabels([])
ax[1, 2].set_yticklabels([])

ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)


ax[0, 0].legend(frameon=False, loc=3, fontsize=fontsize-3)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/parametrize_ubvi.png')
plt.savefig('plot_output/parametrize_ubvi.pdf')



catalog_access = photometry_tools.data_access.CatalogAccess()
vi_int = 1.1
ub_int = -1.4
max_av = 1
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi_ccm89 = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub_ccm89 = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_vi_f99 = dust_tools.extinction_tools.ExtinctionTools.color_ext_f99_av(wave1=v_wave, wave2=i_wave, av=max_av)
max_color_ext_ub_f99 = dust_tools.extinction_tools.ExtinctionTools.color_ext_f99_av(wave1=u_wave, wave2=b_wave, av=max_av)

slope_av_vector_ccm89 = ((ub_int + max_color_ext_ub_ccm89) - ub_int) / ((vi_int + max_color_ext_vi_ccm89) - vi_int)
slope_av_vector_f99 = ((ub_int + max_color_ext_ub_f99) - ub_int) / ((vi_int + max_color_ext_vi_f99) - vi_int)



print('CCM (1989) Milky Way & %.2f \\\\ ' % slope_av_vector_ccm89)
print('F (1999) & %.2f \\\\ ' % slope_av_vector_f99)
print('Class 2 (Hum) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['gradient_err']))
print('Class 2 (ML) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['gradient_err']))
print('Compact associations (Hum) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['gradient_err']))
print('Compact associations (ML) & $%.2f\\pm%.2f$ \\\\ ' % (lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['gradient_err']))



