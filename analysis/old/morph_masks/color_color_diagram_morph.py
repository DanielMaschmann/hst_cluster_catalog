""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import photometry_tools
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize, LogNorm
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from scipy.stats import gaussian_kde


def contours(ax, x, y, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    origin = 'lower'
    cs = ax.contour(xi, yi, zi, levels=levels,
                    linewidths=(2,),
                    origin=origin)

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)



def two_contours(ax, x_1, y_1, x_2, y_2, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values_1 = np.invert(((np.isnan(x_1) | np.isnan(y_1)) | (np.isinf(x_1) | np.isinf(y_1))))
    good_values_2 = np.invert(((np.isnan(x_2) | np.isnan(y_2)) | (np.isinf(x_2) | np.isinf(y_2))))

    x_1 = x_1[good_values_1]
    y_1 = y_1[good_values_1]

    x_2 = x_2[good_values_2]
    y_2 = y_2[good_values_2]

    k_1 = gaussian_kde(np.vstack([x_1, y_1]))
    xi_1, yi_1 = np.mgrid[x_1.min():x_1.max():x_1.size**0.5*1j,y_1.min():y_1.max():y_1.size**0.5*1j]
    zi_1 = k_1(np.vstack([xi_1.flatten(), yi_1.flatten()]))

    k_2 = gaussian_kde(np.vstack([x_2, y_2]))
    xi_2, yi_2 = np.mgrid[x_2.min():x_2.max():x_2.size**0.5*1j,y_2.min():y_2.max():y_2.size**0.5*1j]
    zi_2 = k_2(np.vstack([xi_2.flatten(), yi_2.flatten()]))

    #set zi to 0-1 scale
    zi_1 = (zi_1-zi_1.min())/(zi_1.max() - zi_1.min())
    zi_1 = zi_1.reshape(xi_1.shape)

    zi_2 = (zi_2-zi_2.min())/(zi_2.max() - zi_2.min())
    zi_2 = zi_2.reshape(xi_2.shape)

    origin = 'lower'
    cs_1 = ax.contour(xi_1, yi_1, zi_1, levels=levels, linewidths=(2,), origin=origin, colors='r')
    cs_2 = ax.contour(xi_2, yi_2, zi_2, levels=levels, linewidths=(2,), origin=origin, colors='b')

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs_1.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
mag_v_sol = helper_func.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = helper_func.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol = helper_func.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = helper_func.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol


target_list = catalog_access.target_hst_cc
# target_name_list = catalog_access.phangs_galaxy_list

# catalog_access.load_morph_mask_target_list(target_list=target_list)
# np.save('data_output/morph_mask_data.npy', catalog_access.morph_mask_data)
catalog_access.morph_mask_data = np.load('morph_mask_data.npy', allow_pickle=True).item()


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


class_array = np.array([])
class_qual_array = np.array([])
age_array = np.array([])
ui_array = np.array([])
color_vi_array = np.array([])
color_ub_array = np.array([])

mask_ring_array = np.array([], dtype=bool)
mask_center_array = np.array([], dtype=bool)
mask_bar_array = np.array([], dtype=bool)
mask_arm_array = np.array([], dtype=bool)
mask_inter_arm_array = np.array([], dtype=bool)
mask_floc_disc_array = np.array([], dtype=bool)


for target in target_list:

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    print('target ', target)

    color_ub_12_ml = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_12_ml = catalog_access.get_hst_color_vi(target=target, classify='ml')
    color_ub_3_ml = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_3_ml = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    color_vi_array = np.concatenate([color_vi_array, color_vi_12_ml, color_vi_3_ml])
    color_ub_array = np.concatenate([color_ub_array, color_ub_12_ml, color_ub_3_ml])

    class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    class_array = np.concatenate([class_array, class_12_ml, class_3_ml])

    class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    class_qual_array = np.concatenate([class_qual_array, class_qual_12_ml, class_qual_3_ml])

    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    age_array = np.concatenate([age_array, age_12_ml, age_3_ml])

    ra_12_ml, dec_12_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_3_ml, dec_3_ml = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    ra_ml = np.concatenate([ra_12_ml, ra_3_ml])
    dec_ml = np.concatenate([dec_12_ml, dec_3_ml])
    pos_mask_dict_ml = catalog_access.get_morph_locations(target=target, ra=ra_ml, dec=dec_ml)

    mask_ring_ml = pos_mask_dict_ml['pos_mask_ring']
    mask_center_ml = pos_mask_dict_ml['pos_mask_center'] * ~mask_ring_ml
    mask_bar_ml = ((pos_mask_dict_ml['pos_mask_bar'] + pos_mask_dict_ml['pos_mask_lens']) *
                   ~mask_center_ml * ~mask_ring_ml)
    mask_arm_ml = pos_mask_dict_ml['pos_mask_arm'] * ~mask_center_ml * ~mask_ring_ml * ~mask_bar_ml
    mask_disc_ml = (pos_mask_dict_ml['pos_mask_disc'] * ~mask_arm_ml * ~mask_center_ml *
                    ~mask_ring_ml * ~mask_bar_ml)

    presence_arm = catalog_access.morph_mask_data[galaxy_name]['presence_arm']
    presence_bar = catalog_access.morph_mask_data[galaxy_name]['presence_bar']
    presence_disc = catalog_access.morph_mask_data[galaxy_name]['presence_disc']

    print('presence_arm ', presence_arm)
    print('presence_bar ', presence_bar)
    print('presence_disc ', presence_disc)


    mask_ring_array = np.concatenate([mask_ring_array, mask_ring_ml])
    mask_center_array = np.concatenate([mask_center_array, mask_center_ml])
    mask_bar_array = np.concatenate([mask_bar_array, mask_bar_ml])
    mask_arm_array = np.concatenate([mask_arm_array, mask_arm_ml])
    if presence_arm:
        mask_inter_arm_array = np.concatenate([mask_inter_arm_array, mask_disc_ml])
    else:
        mask_inter_arm_array = np.concatenate([mask_inter_arm_array, np.zeros(len(mask_disc_ml), dtype=bool)])
    if presence_disc & ~presence_arm & presence_bar:
        mask_floc_disc_array = np.concatenate([mask_floc_disc_array, mask_disc_ml])
    else:
        mask_floc_disc_array = np.concatenate([mask_floc_disc_array, np.zeros(len(mask_disc_ml), dtype=bool)])

class_1 = class_array == 1
class_2 = class_array == 2
class_3 = class_array == 3

good_colors = (color_vi_array > -2) & (color_vi_array < 3) & (color_ub_array > -3) & (color_ub_array < 2)



fig, ax = plt.subplots(ncols=5, nrows=3, sharex=True, sharey=True, figsize=(27, 14))
fontsize = 16

ax[0, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[1, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[2, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[0, 0].scatter(color_vi_array[mask_ring_array*class_1], color_ub_array[mask_ring_array*class_1], c='r', s=1)
ax[1, 0].scatter(color_vi_array[mask_ring_array*class_2], color_ub_array[mask_ring_array*class_2], c='b', s=1)
ax[2, 0].scatter(color_vi_array[mask_ring_array*class_3], color_ub_array[mask_ring_array*class_3], c='b', s=1)
# two_contours(ax=ax[0, 0], x_1=color_vi_array[mask_ring_array*good_colors*class_1],
#              y_1=color_ub_array[mask_ring_array*good_colors*class_1],
#              x_2=color_vi_array[mask_ring_array*good_colors*class_2],
#              y_2=color_ub_array[mask_ring_array*good_colors*class_2])
contours(ax=ax[0, 0], x=color_vi_array[mask_ring_array*good_colors*class_1],
         y=color_ub_array[mask_ring_array*good_colors*class_1])
contours(ax=ax[1, 0], x=color_vi_array[mask_ring_array*good_colors*class_2],
         y=color_ub_array[mask_ring_array*good_colors*class_2])
contours(ax=ax[2, 0], x=color_vi_array[mask_ring_array*good_colors*class_3],
         y=color_ub_array[mask_ring_array*good_colors*class_3])
ax[0, 0].set_title('Central ring', fontsize=fontsize)


ax[0, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[1, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[2, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[0, 1].scatter(color_vi_array[mask_bar_array*class_1], color_ub_array[mask_bar_array*class_1], c='r', s=1)
ax[1, 1].scatter(color_vi_array[mask_bar_array*class_2], color_ub_array[mask_bar_array*class_2], c='b', s=1)
ax[2, 1].scatter(color_vi_array[mask_bar_array*class_3], color_ub_array[mask_bar_array*class_3], c='b', s=1)
# two_contours(ax=ax[0, 1], x_1=color_vi_array[mask_bar_array*good_colors*class_1],
#              y_1=color_ub_array[mask_bar_array*good_colors*class_1],
#              x_2=color_vi_array[mask_bar_array*good_colors*class_2],
#              y_2=color_ub_array[mask_bar_array*good_colors*class_2])
contours(ax=ax[0, 1], x=color_vi_array[mask_bar_array*good_colors*class_1],
         y=color_ub_array[mask_bar_array*good_colors*class_1])
contours(ax=ax[1, 1], x=color_vi_array[mask_bar_array*good_colors*class_2],
         y=color_ub_array[mask_bar_array*good_colors*class_2])
contours(ax=ax[2, 1], x=color_vi_array[mask_bar_array*good_colors*class_3],
         y=color_ub_array[mask_bar_array*good_colors*class_3])
ax[0, 1].set_title('Bar', fontsize=fontsize)

ax[0, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[1, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[2, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[0, 2].scatter(color_vi_array[mask_arm_array*class_1], color_ub_array[mask_arm_array*class_1], c='r', s=1)
ax[1, 2].scatter(color_vi_array[mask_arm_array*class_2], color_ub_array[mask_arm_array*class_2], c='b', s=1)
ax[2, 2].scatter(color_vi_array[mask_arm_array*class_2], color_ub_array[mask_arm_array*class_2], c='b', s=1)
# two_contours(ax=ax[2], x_1=color_vi_array[mask_arm_array*good_colors*class_1],
#              y_1=color_ub_array[mask_arm_array*good_colors*class_1],
#              x_2=color_vi_array[mask_arm_array*good_colors*class_2],
#              y_2=color_ub_array[mask_arm_array*good_colors*class_2])
contours(ax=ax[0, 2], x=color_vi_array[mask_arm_array*good_colors*class_1],
         y=color_ub_array[mask_arm_array*good_colors*class_1])
contours(ax=ax[1, 2], x=color_vi_array[mask_arm_array*good_colors*class_2],
         y=color_ub_array[mask_arm_array*good_colors*class_2])
contours(ax=ax[2, 2], x=color_vi_array[mask_arm_array*good_colors*class_3],
         y=color_ub_array[mask_arm_array*good_colors*class_3])
ax[0, 2].set_title('Arm', fontsize=fontsize)

ax[0, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[1, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[2, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[0, 3].scatter(color_vi_array[mask_inter_arm_array*class_1], color_ub_array[mask_inter_arm_array*class_1], c='r', s=1)
ax[1, 3].scatter(color_vi_array[mask_inter_arm_array*class_2], color_ub_array[mask_inter_arm_array*class_2], c='b', s=1)
ax[2, 3].scatter(color_vi_array[mask_inter_arm_array*class_2], color_ub_array[mask_inter_arm_array*class_2], c='b', s=1)
# two_contours(ax=ax[3], x_1=color_vi_array[mask_inter_arm_array*good_colors*class_1],
#              y_1=color_ub_array[mask_inter_arm_array*good_colors*class_1],
#              x_2=color_vi_array[mask_inter_arm_array*good_colors*class_2],
#              y_2=color_ub_array[mask_inter_arm_array*good_colors*class_2])
contours(ax=ax[0, 3], x=color_vi_array[mask_inter_arm_array*good_colors*class_1],
         y=color_ub_array[mask_inter_arm_array*good_colors*class_1])
contours(ax=ax[1, 3], x=color_vi_array[mask_inter_arm_array*good_colors*class_2],
         y=color_ub_array[mask_inter_arm_array*good_colors*class_2])
contours(ax=ax[2, 3], x=color_vi_array[mask_inter_arm_array*good_colors*class_3],
         y=color_ub_array[mask_inter_arm_array*good_colors*class_3])
ax[0, 3].set_title('Inter arm', fontsize=fontsize)

ax[0, 4].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[1, 4].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[2, 4].plot(model_vi_sol, model_ub_sol, color='red', linewidth=1.2)
ax[0, 4].scatter(color_vi_array[mask_floc_disc_array*class_1], color_ub_array[mask_floc_disc_array*class_1], c='r', s=1)
ax[1, 4].scatter(color_vi_array[mask_floc_disc_array*class_2], color_ub_array[mask_floc_disc_array*class_2], c='b', s=1)
ax[2, 4].scatter(color_vi_array[mask_floc_disc_array*class_2], color_ub_array[mask_floc_disc_array*class_2], c='b', s=1)
# two_contours(ax=ax[4], x_1=color_vi_array[mask_floc_disc_array*good_colors*class_1],
#              y_1=color_ub_array[mask_floc_disc_array*good_colors*class_1],
#              x_2=color_vi_array[mask_floc_disc_array*good_colors*class_2],
#              y_2=color_ub_array[mask_floc_disc_array*good_colors*class_2])
contours(ax=ax[0, 4], x=color_vi_array[mask_floc_disc_array*good_colors*class_1],
         y=color_ub_array[mask_floc_disc_array*good_colors*class_1])
contours(ax=ax[1, 4], x=color_vi_array[mask_floc_disc_array*good_colors*class_2],
         y=color_ub_array[mask_floc_disc_array*good_colors*class_2])
contours(ax=ax[2, 4], x=color_vi_array[mask_floc_disc_array*good_colors*class_3],
         y=color_ub_array[mask_floc_disc_array*good_colors*class_3])
ax[0, 4].set_title('Flocculant disc', fontsize=fontsize)


ax[0, 0].set_ylim(1.25, -2.2)
ax[0, 0].set_xlim(-1.0, 2.3)
# fig.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
# fig.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)

plt.savefig('plot_output/color_color_morph_1.png')

exit()



mask_ring_array
mask_center_array
mask_bar_array
mask_arm_array
mask_inter_arm_array
mask_floc_disc_array



print('!!!!')
print(np.sum(n_total_hum) / np.sum(total_surf))
print('n_total_hum ', n_total_hum)
print('n_total_ml ', n_total_ml)
print('total_surf ', total_surf)

print('n_young_hum ', n_young_hum)
print('n_inter_hum ', n_inter_hum)
print('n_old_hum ', n_old_hum)
print('n_class_1_hum ', n_class_1_hum)
print('n_class_2_hum ', n_class_2_hum)
print('n_class_3_hum ', n_class_3_hum)

print('n_young_ml ', n_young_ml)
print('n_inter_ml ', n_inter_ml)
print('n_old_ml ', n_old_ml)
print('n_class_1_ml ', n_class_1_ml)
print('n_class_2_ml ', n_class_2_ml)
print('n_class_3_ml ', n_class_3_ml)


ax[0].scatter(pos_array, n_total_hum / total_surf, color='k')
ax[0].plot(pos_array, n_total_hum / total_surf, linewidth=2, color='k', label='total')
ax[0].plot([0, 6], [np.sum(n_total_hum) / np.sum(total_surf), np.sum(n_total_hum) / np.sum(total_surf)], color='grey', linewidth=2, linestyle='--')
ax[0].text(4.5, np.sum(n_total_hum) / np.sum(total_surf) + 0.1, 'Mean', color='grey', fontsize=fontsize)


ax[0].scatter(pos_array, n_young_hum / total_surf)
ax[0].plot(pos_array, n_young_hum / total_surf, linewidth=2, label='young')
ax[0].scatter(pos_array, n_inter_hum / total_surf)
ax[0].plot(pos_array, n_inter_hum / total_surf, linewidth=2, label='inter')
ax[0].scatter(pos_array, n_old_hum / total_surf)
ax[0].plot(pos_array, n_old_hum / total_surf, linewidth=2, label='old')

ax[0].scatter(pos_array, n_class_1_hum / total_surf)
ax[0].plot(pos_array, n_class_1_hum / total_surf, linewidth=2, label='class 1')
ax[0].scatter(pos_array, n_class_2_hum / total_surf)
ax[0].plot(pos_array, n_class_2_hum / total_surf, linewidth=2, label='class 2')
ax[0].scatter(pos_array, n_class_3_hum / total_surf)
ax[0].plot(pos_array, n_class_3_hum / total_surf, linewidth=2, label='class 3')

ax[0].set_xticks(pos_array)
ax[0].xaxis.set_tick_params(rotation=45)
ax[0].set_xticklabels(name_array)
ax[0].set_title('Human', fontsize=fontsize)
ax[0].legend(frameon=False, fontsize=fontsize)
ax[0].set_ylabel(r'N/$\Sigma$ [kp$^{-2}$]', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=3, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0].set_yscale('log')

ax[1].scatter(pos_array, n_total_ml / total_surf, color='k')
ax[1].plot(pos_array, n_total_ml / total_surf, linewidth=2, color='k', label='total')
ax[1].plot([0, 6], [np.sum(n_total_ml) / np.sum(total_surf), np.sum(n_total_ml) / np.sum(total_surf)], color='grey', linewidth=2, linestyle='--')
ax[1].text(4.5, np.sum(n_total_ml) / np.sum(total_surf) + 0.1, 'Mean', color='grey', fontsize=fontsize)



ax[1].scatter(pos_array, n_young_ml / total_surf)
ax[1].plot(pos_array, n_young_ml / total_surf, linewidth=2, label='young')
ax[1].scatter(pos_array, n_inter_ml / total_surf)
ax[1].plot(pos_array, n_inter_ml / total_surf, linewidth=2, label='inter')
ax[1].scatter(pos_array, n_old_ml / total_surf)
ax[1].plot(pos_array, n_old_ml / total_surf, linewidth=2, label='old')

ax[1].scatter(pos_array, n_class_1_ml / total_surf)
ax[1].plot(pos_array, n_class_1_ml / total_surf, linewidth=2, label='class 1')
ax[1].scatter(pos_array, n_class_2_ml / total_surf)
ax[1].plot(pos_array, n_class_2_ml / total_surf, linewidth=2, label='class 2')
ax[1].scatter(pos_array, n_class_3_ml / total_surf)
ax[1].plot(pos_array, n_class_3_ml / total_surf, linewidth=2, label='class 3')

ax[1].set_xticks(pos_array)
ax[1].xaxis.set_tick_params(rotation=45)
ax[1].set_xticklabels(name_array)
ax[1].set_title('ML', fontsize=fontsize)
ax[1].legend(frameon=False, fontsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=3, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].set_yscale('log')




# plt.show()
plt.savefig('plot_output/cluster_dens_overview.png')
