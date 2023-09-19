import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import dust_tools.extinction_tools
from photometry_tools import helper_func

from astropy.io import fits
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
y_lim_ub = (0.9, -1.9)

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_hum = detect_u_hum * detect_b_hum * detect_v_hum * detect_i_hum
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_good_colors_ubvi_hum = ((color_vi_hum > (x_lim_vi[0] - 1)) & (color_vi_hum < (x_lim_vi[1] + 1)) &
                             (color_ub_hum > (y_lim_ub[1] - 1)) & (color_ub_hum < (y_lim_ub[0] + 1)) &
                             mask_detect_ubvi_hum)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                            (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                            mask_detect_ubvi_ml)


convex_hull_young_hum = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_hum_12.fits')[1].data
convex_hull_young_ml = fits.open('../color_color_regions/data_output/convex_hull_ubvi_young_ml_12.fits')[1].data

convex_hull_vi_young_hum = convex_hull_young_hum['vi']
convex_hull_ub_young_hum = convex_hull_young_hum['ub']
convex_hull_vi_young_ml = convex_hull_young_ml['vi']
convex_hull_ub_young_ml = convex_hull_young_ml['ub']

hull_young_hum = ConvexHull(np.array([convex_hull_vi_young_hum, convex_hull_ub_young_hum]).T)
hull_young_ml = ConvexHull(np.array([convex_hull_vi_young_ml, convex_hull_ub_young_ml]).T)

in_hull_young_hum = helper_func.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_young_hum)
in_hull_young_ml = helper_func.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)

# get gauss und segmentations
n_bins_ubvi = 120
threshold_fact = 3
kernal_std = 1.0
contrast = 0.01

gauss_dict_ubvi_hum_2 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_hum_3 = helper_func.calc_seg(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum],
                                             x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                             threshold_fact=5, kernal_std=kernal_std, contrast=contrast)




gauss_dict_ubvi_ml_2 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)
gauss_dict_ubvi_ml_3 = helper_func.calc_seg(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                                              x_lim=x_lim_vi, y_lim=y_lim_ub, n_bins=n_bins_ubvi,
                                              threshold_fact=5, kernal_std=kernal_std, contrast=contrast)


x_bins_vi = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_ubvi)
y_bins_ub = np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_ubvi)
x_mesh_vi, y_mesh_ub = np.meshgrid(x_bins_vi, y_bins_ub)

gauss_map_hum_2_bkg = gauss_dict_ubvi_hum_2['gauss_map'].copy()
gauss_map_hum_2_young = gauss_dict_ubvi_hum_2['gauss_map'].copy()
gauss_map_hum_3_bkg = gauss_dict_ubvi_hum_3['gauss_map'].copy()
gauss_map_hum_3_young = gauss_dict_ubvi_hum_3['gauss_map'].copy()

gauss_map_ml_2_bkg = gauss_dict_ubvi_ml_2['gauss_map'].copy()
gauss_map_ml_2_young = gauss_dict_ubvi_ml_2['gauss_map'].copy()
gauss_map_ml_3_bkg = gauss_dict_ubvi_ml_3['gauss_map'].copy()
gauss_map_ml_3_young = gauss_dict_ubvi_ml_3['gauss_map'].copy()

in_hull_map_hum = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_ub.flatten()]).T, hull_young_hum), dtype=bool)
in_hull_map_ml = np.array(helper_func.points_in_hull(np.array([x_mesh_vi.flatten(), y_mesh_ub.flatten()]).T, hull_young_ml), dtype=bool)
in_hull_map_hum = np.reshape(in_hull_map_hum, newshape=(n_bins_ubvi, n_bins_ubvi))
in_hull_map_ml = np.reshape(in_hull_map_ml, newshape=(n_bins_ubvi, n_bins_ubvi))


gauss_map_hum_2_bkg[in_hull_map_hum] = np.nan
gauss_map_hum_2_young[np.invert(in_hull_map_hum)] = np.nan
gauss_map_hum_3_bkg[in_hull_map_hum] = np.nan
gauss_map_hum_3_young[np.invert(in_hull_map_hum)] = np.nan

gauss_map_ml_2_bkg[in_hull_map_ml] = np.nan
gauss_map_ml_2_young[np.invert(in_hull_map_ml)] = np.nan
gauss_map_ml_3_bkg[in_hull_map_ml] = np.nan
gauss_map_ml_3_young[np.invert(in_hull_map_ml)] = np.nan

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
fontsize = 23

vmax_hum_2 = np.nanmax(gauss_dict_ubvi_hum_2['gauss_map']) / 1.2
ax[0, 0].imshow(gauss_map_hum_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_2)
ax[0, 0].imshow(gauss_map_hum_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_2)

vmax_hum_3 = np.nanmax(gauss_dict_ubvi_hum_3['gauss_map']) / 1.2
ax[0, 1].imshow(gauss_map_hum_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_hum_3)
ax[0, 1].imshow(gauss_map_hum_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_hum_3)

vmax_ml_2 = np.nanmax(gauss_dict_ubvi_ml_2['gauss_map']) / 1.2
ax[1, 0].imshow(gauss_map_ml_2_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_2)
ax[1, 0].imshow(gauss_map_ml_2_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_2)

vmax_ml_3 = np.nanmax(gauss_dict_ubvi_ml_3['gauss_map']) / 1.2
ax[1, 1].imshow(gauss_map_ml_3_bkg, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Greys', vmin=0, vmax=vmax_ml_3)
ax[1, 1].imshow(gauss_map_ml_3_young, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
          interpolation='nearest', aspect='auto', cmap='Blues', vmin=0, vmax=vmax_ml_3)





lin_fit_result_ubvi_c2_hum = helper_func.fit_line(x_data=color_vi_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     y_data=color_ub_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     x_data_err=color_vi_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     y_data_err=color_ub_err_hum[mask_class_2_hum * mask_good_colors_ubvi_hum * in_hull_young_hum])


lin_fit_result_ubvi_c3_hum = helper_func.fit_line(x_data=color_vi_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     y_data=color_ub_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     x_data_err=color_vi_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],
                                     y_data_err=color_ub_err_hum[mask_class_3_hum * mask_good_colors_ubvi_hum * in_hull_young_hum],)
lin_fit_result_ubvi_c2_ml = helper_func.fit_line(x_data=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     x_data_err=color_vi_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data_err=color_ub_err_ml[mask_class_2_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                                 )
lin_fit_result_ubvi_c3_ml = helper_func.fit_line(x_data=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     x_data_err=color_vi_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml],
                                     y_data_err=color_ub_err_ml[mask_class_3_ml * mask_good_colors_ubvi_ml * in_hull_young_ml])

dummy_x_data = np.linspace(x_lim_vi[0], x_lim_vi[1], 100)
dummy_y_data_ubvi_c2_hum = helper_func.lin_func((lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c3_hum = helper_func.lin_func((lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c2_ml = helper_func.lin_func((lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['intersect']), x=dummy_x_data)
dummy_y_data_ubvi_c3_ml = helper_func.lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']), x=dummy_x_data)

ax[0, 0].plot(dummy_x_data, dummy_y_data_ubvi_c2_hum, color='k', linewidth=2, linestyle='--')
ax[0, 1].plot(dummy_x_data, dummy_y_data_ubvi_c3_hum, color='k', linewidth=2, linestyle='--')
ax[1, 0].plot(dummy_x_data, dummy_y_data_ubvi_c2_ml, color='k', linewidth=2, linestyle='--')
ax[1, 1].plot(dummy_x_data, dummy_y_data_ubvi_c3_ml, color='k', linewidth=2, linestyle='--')


x_text_pos = 1.0
text_anle_c2_hum = np.arctan(lin_fit_result_ubvi_c2_hum['gradient']) * 180/np.pi
text_anle_c3_hum = np.arctan(lin_fit_result_ubvi_c3_hum['gradient']) * 180/np.pi
text_anle_c2_ml = np.arctan(lin_fit_result_ubvi_c2_ml['gradient']) * 180/np.pi
text_anle_c3_ml = np.arctan(lin_fit_result_ubvi_c3_ml['gradient']) * 180/np.pi

ax[0, 0].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['intersect']),
                       x=x_text_pos)+0.05,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c2_hum['gradient'], lin_fit_result_ubvi_c2_hum['gradient_err']),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=text_anle_c2_hum, fontsize=fontsize - 5)
ax[0, 1].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['intersect']),
                       x=x_text_pos)+0.05,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c3_hum['gradient'], lin_fit_result_ubvi_c3_hum['gradient_err']),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=text_anle_c3_hum, fontsize=fontsize - 5)
ax[1, 0].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['intersect']),
                       x=x_text_pos)+0.05,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c2_ml['gradient'], lin_fit_result_ubvi_c2_ml['gradient_err']),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=text_anle_c2_ml, fontsize=fontsize - 5)
ax[1, 1].text(x_text_pos-0.12,
              helper_func.lin_func((lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['intersect']),
                       x=x_text_pos)+0.05,
              r'slope = %.2f$\,\pm\,$%.2f' %
              (lin_fit_result_ubvi_c3_ml['gradient'], lin_fit_result_ubvi_c3_ml['gradient_err']),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=text_anle_c3_ml, fontsize=fontsize - 5)



ax[0, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-', label=r'BC03, Z$_{\odot}/50$')

ax[0, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[1, 0].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[1, 0].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

ax[1, 1].plot(model_vi_sol, model_ub_sol, color='darkred', linewidth=2, linestyle='-')
ax[1, 1].plot(model_vi_sol50, model_ub_sol50, color='darkorange', linewidth=2, linestyle='-')

vi_int = 1.1
ub_int = -1.6
av_value = 1

helper_func.plot_reddening_vect(ax=ax[0, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=True, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)
helper_func.plot_reddening_vect(ax=ax[0, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=False, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)
helper_func.plot_reddening_vect(ax=ax[1, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=False, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)
helper_func.plot_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                                x_color_int=vi_int, y_color_int=ub_int, av_val=av_value,
                                linewidth=3, line_color='k',
                                text=False, fontsize=fontsize - 2, text_color='k', x_text_offset=0.0, y_text_offset=-0.1)


catalog_access = photometry_tools.data_access.CatalogAccess()
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=av_value)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=av_value)

slope_av_vector = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
angle_av_vector = np.arctan(slope_av_vector) * 180/np.pi
print('slope_av_vector ', slope_av_vector)
print('angle_av_vector ', angle_av_vector)

ax[0, 0].text(vi_int-0.12, ub_int+0.05,
              r'slope = %.2f' % (slope_av_vector),
              horizontalalignment='left', verticalalignment='center', transform_rotates_text=True, rotation_mode='anchor',
              rotation=angle_av_vector, fontsize=fontsize - 5)





ax[0, 0].set_title('Class 2 (Human)', fontsize=fontsize)
ax[0, 1].set_title('Compact Associations (Human)', fontsize=fontsize)
ax[1, 1].set_title('Class 2 (ML)', fontsize=fontsize)
ax[1, 0].set_title('Compact Associations (ML)', fontsize=fontsize)

ax[0, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N = %i, (%i fitted)' %
              (sum(mask_class_2_hum), sum(mask_class_2_hum*in_hull_young_hum*mask_good_colors_ubvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N = %i, (%i fitted)' %
              (sum(mask_class_3_hum), sum(mask_class_3_hum*in_hull_young_hum*mask_good_colors_ubvi_hum)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N = %i, (%i fitted)' %
              (sum(mask_class_2_ml), sum(mask_class_2_ml*in_hull_young_ml*mask_good_colors_ubvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N = %i, (%i fitted)' %
              (sum(mask_class_3_ml), sum(mask_class_3_ml*in_hull_young_ml*mask_good_colors_ubvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


#
# ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 2 (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[0, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Compact associations (Hum)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 1 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Class 2 (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
# ax[1, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.95, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.93,
#               'Compact associations (ML)', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)


ax[0, 0].set_xlim(x_lim_vi)
ax[0, 1].set_xlim(x_lim_vi)
ax[0, 0].set_ylim(y_lim_ub)
ax[0, 1].set_ylim(y_lim_ub)

ax[1, 0].set_xlim(x_lim_vi)
ax[1, 1].set_xlim(x_lim_vi)
ax[1, 0].set_ylim(y_lim_ub)
ax[1, 1].set_ylim(y_lim_ub)

ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])

ax[0, 1].set_yticklabels([])
ax[1, 1].set_yticklabels([])

ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0, 0].legend(frameon=False, loc=3, bbox_to_anchor=(0, 0.05), fontsize=fontsize-3)

ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


# plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/fit_reddening_vect.png')
plt.savefig('plot_output/fit_reddening_vect.pdf')



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



