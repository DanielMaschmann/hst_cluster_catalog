import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from xgaltool import analysis_tools
import matplotlib

from matplotlib import cm

from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects

# import os
from astropy.convolution import convolve

from photutils.segmentation import make_2dgaussian_kernel
from astropy.table import Table
from matplotlib.legend_handler import HandlerTuple
# os.system('firefox https://www.youtube.com/watch?v=F8AIydIusUQ')


def contours(ax, x, y, levels=None, axis_offse=(-0.2, 0.1, -0.55, 0.6)):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]

    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min()+axis_offse[0]:x.max()+axis_offse[1]:x.size**0.5*1j,
             y.min()+axis_offse[2]:y.max()+axis_offse[3]:y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    # ax[0].scatter(xi.flatten(), yi.flatten(), c=zi)
    cmap = cm.get_cmap('viridis')
    cs = ax.contour(xi, yi, zi, levels=levels,
                    cmap=cmap,
                    linewidths=(3,),
                    origin='lower')



# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)


age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')


target_list = catalog_access.target_hst_cc

catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='human')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')


x_lim_low = 9.1
x_lim_high = 11.4
y_lim_low = -0.9
y_lim_high = 1.4

len_x = x_lim_high - x_lim_low
len_y = y_lim_high - y_lim_low

# print('len_x ', len_x)
# print('len_y ', len_y)
fig = plt.figure(figsize=(20, 20 * len_y / len_x))
fontsize = 26

ms_x_pos = 0.06
ms_y_pos = 0.06
ms_x_len = 0.93
ms_y_len = 0.93

contour_width = 0.10
contour_hight = 0.10

ax_ms = fig.add_axes([ms_x_pos, ms_y_pos, ms_x_len, ms_y_len])

ax_ms.set_xlim(x_lim_low, x_lim_high)
ax_ms.set_ylim(y_lim_low, y_lim_high)

# plot contours
gswlc_access = analysis_tools.AnalysisTools(object_type='gswlc_a_v1', writable_table=True)
# gswlc_access = analysis_tools.AnalysisTools(object_type='mpa', writable_table=True)
gswlc_access.cross_match_table(cross_match='rcsed')
log_sfr = gswlc_access.get_log_sed_sfr()
log_stellar_mass = gswlc_access.get_log_sed_stellar_mass()
flag_sed_fit = (gswlc_access.table['FLAG_SED'] == 0) & (gswlc_access.table['FLAG_MGS'] == 1) & (gswlc_access.table['REDCHISQ'] < 5)
# log_sfr = gswlc_access.get_log_mpa_sfr()
# log_stellar_mass = gswlc_access.get_log_mpa_stellar_mass()

redshift = gswlc_access.get_redshift()
good_values = (log_sfr > -2.5) & (log_sfr < 3.2) & (log_stellar_mass > 6.5) & (log_stellar_mass < 12.5) & (redshift < 0.05)
good_values_sloan = (log_sfr > -2.5) & (log_sfr < 3.2) & (log_stellar_mass > 6.5) & (log_stellar_mass < 12.5) & (redshift < 0.15)
# good_values_sloan = (log_sfr > -2.5) & (log_sfr < 3.2) & (log_stellar_mass > 6.5) & (log_stellar_mass < 12.5) & (flag_sed_fit) & (redshift > 0.01) & (redshift < 0.2)

# plot contours
xcoldgass_access = analysis_tools.AnalysisTools(object_type='xcoldgass', writable_table=True)
log_sfr_xcoldgass = xcoldgass_access.table['LOGSFR_SED']
log_stellar_mass_xcoldgass = xcoldgass_access.table['LOGMSTAR']
redshift_xcoldgass = xcoldgass_access.table['Z_SDSS']


# leroy_table = Table.read('J_ApJS_244_24/table4.dat', readme='J_ApJS_244_24/ReadMe', format="ascii.cds")
# dist_leroy = leroy_table['Dist']
# log_stellar_mass_leroy = leroy_table['logM*']
# logsfr_leroy = leroy_table['logSFR']

# get all Phangs alma
log_stellar_mass_phangs_alma = []
log_sfr_phangs_alma = []
catalog_access.load_sample_table()
for phangs_target in catalog_access.sample_table['target_names']:
    # if phangs_target in catalog_access.phangs_galaxy_list:
    #     continue
    sfr = catalog_access.get_target_sfr(target=phangs_target)
    mstar = catalog_access.get_target_mstar(target=phangs_target)
    log_stellar_mass_phangs_alma.append(np.log10(mstar))
    log_sfr_phangs_alma.append(np.log10(sfr))




dummy_m_star = np.linspace(7, 13, 50)
# take from Leroy 2019

dummy_sfr = -0.32 * (dummy_m_star - 10.0) - 10.17 + dummy_m_star
dummy_sfr_polynome = -2.332 * dummy_m_star + 0.4156 * dummy_m_star**2 - 0.01828 * dummy_m_star**3


# taken from Catinella 2018
dumy_sfr_std = 0.088*(dummy_m_star - 9) + 0.188

upper_dummy_sfr = dummy_sfr + dumy_sfr_std
lower_dummy_sfr = dummy_sfr - dumy_sfr_std




hist, x_bin, y_bin = np.histogram2d(log_stellar_mass[good_values_sloan], log_sfr[good_values_sloan],
                                    bins=(np.linspace(x_lim_low, x_lim_high, 70),
                                          np.linspace(y_lim_low, y_lim_high, 70)))

ax_ms.fill_between(dummy_m_star, upper_dummy_sfr, lower_dummy_sfr, color='grey', alpha=0.3)
ax_ms.plot(dummy_m_star, dummy_sfr, linewidth=2, color='k', linestyle='--')
# ax_ms.plot(dummy_m_star, dummy_sfr_polynome, linewidth=2, color='g', linestyle='-')

kernel = make_2dgaussian_kernel(2.0, size=9)  # FWHM = 3.0
hist = convolve(hist, kernel)

# hist[hist < 10] = np.nan
cmap_ms = matplotlib.cm.get_cmap('Purples')
norm_ms = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(hist)/1.1)


ax_ms.imshow(hist.T, extent=(x_lim_low, x_lim_high, y_lim_low, y_lim_high), origin='lower', cmap=cmap_ms, norm=norm_ms)
# ax_ms.scatter(log_stellar_mass_phangs_alma, log_sfr_phangs_alma, color='dodgerblue', linewidth=3, marker='o', s=130, facecolor='None')
# ax_ms.scatter(log_stellar_mass_xcoldgass, log_sfr_xcoldgass, color='tab:orange', marker='o', s=80)
ax_ms.set_xlabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize+5)
ax_ms.set_ylabel('log(SFR/M$_{\odot}$ yr$^{-1}$)', labelpad=-33, fontsize=fontsize+5)
ax_ms.tick_params(axis='both', which='both',
                  width=5, length=10, right=True, top=True, direction='in', labelsize=fontsize+5)

#
# p1 = ax_ms.scatter([], [], color='r', s=80)
# p2 = ax_ms.scatter([], [], color='r', marker='*', s=200)
# p3 = ax_ms.scatter([], [], color='dodgerblue', linewidth=3, marker='o', s=130, facecolor='None')
# # p4 = ax_ms.scatter([], [], color='tab:orange', marker='o', s=80)

# Assign two of the handles to the same legend entry by putting them in a tuple
# and using a generic handler map (which would be used for any additional
# tuples of handles like (p1, p3)).
# l = ax_ms.legend([(p1, p2), p3], ['PHANGS-HST', 'PHANGS-ALMA'],
#                  handler_map={tuple: HandlerTuple(ndivide=None)}, loc=2, fontsize=fontsize)




offset_dict = {
    'ic1954': [-0.05, 0.05, 1.5, 0.5],
    'ic5332': [0.0, 0.05, 1.0, 0.5],
    'ngc0628e': [-0.09, 0.0, 1.5, -1.0],
    'ngc0628c': [-0.04, 0.05, 1.5, 0.5],
    'ngc0685': [-0.07, -0.02, 1.5, -1.3],
    'ngc1087': [-0.05, 0.0, 1.5, -0.5],
    'ngc1097': [+0.06, 0.1,  0.3, -0.3],
    'ngc1300': [-0.05, -0.0, 1.1, -1.3],
    'ngc1317': [-0.05, 0.05, 1.5, 0.5],
    'ngc1365': [0, 0, 0, 0],
    'ngc1385': [-0.05, 0.0, 1.5, -0.5],
    'ngc1433': [-0.0, -0.08, 0.5, -2.5],
    'ngc1512': [0, 0, 0, 0],
    'ngc1559': [-0.14, 0.11, 1.5, 0.5],
    'ngc1566': [+0.12, 0.06, 0.3, -0.3],
    'ngc1672': [0, 0, 0, 0],
    'ngc1792': [+0.025, 0.035, 0.3, -0.3],
    'ngc2775': [0, 0, 0, 0],
    'ngc2835': [0, 0, 0, 0],
    'ngc2903': [-0.06, 0.04, 1.5, 0.5],
    'ngc3351': [0, 0, 0, 0],
    'ngc3621': [+0.06, -0.04, -0.5, -1.5],
    'ngc3627': [+0.11, 0.005, 0.3, -0.3],
    'ngc4254': [-0.14, 0.05, 1.5, -0.5],
    'ngc4298': [-0.05, 0.05, 1.5, 0.5],
    'ngc4303': [-0.05, 0.05, 1.5, 0.5],
    'ngc4321': [+0.09, -0.04, 1., 0.5],
    'ngc4535': [0, 0, 0, 0],
    'ngc4536': [-0.08, 0.11, 1.5, 0.5],
    'ngc4548': [0.0, 0.05, 1.0, 0.5],
    'ngc4569': [+0.03, 0.05, 0.3, 0.2],
    'ngc4571': [0, 0, 0, 0],
    'ngc4654': [0.0, 0.15, 1.0, 0.8],
    'ngc4689': [0, 0, 0, 0],
    'ngc4826': [0, 0, 0, 0],
    'ngc5068': [0, 0, 0, 0],
    'ngc5248': [0, 0, 0, 0],
    'ngc6744': [0, 0, 0, 0],
    'ngc7496': [-0.02, 0.06, 1.0, 0.6],
}


log_sfr_list = []
log_stellar_mass_list = []

# for index in range(2):
for index in range(len(target_list)):
    target = target_list[index]

    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_name_str = target_list[index][0:3] + ' ' +  target_list[index][4:]
    elif target_list[index][0:2] == 'ic':
        target_name_str = target_list[index][0:2] + ' ' +  target_list[index][2:]
    elif target_list[index][0:3] == 'ngc':
        target_name_str = target_list[index][0:3] + ' ' +  target_list[index][3:]
    else:
        target_name_str = target_list[index]
    target_name_str = target_name_str.upper()

    print('target ', target_name_str)

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') > 0))
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))

    if (target == 'ngc0628e') | (target == 'ngc0628c'):
        target_str = 'ngc0628'
    else:
        target_str = target

    sfr = catalog_access.get_target_sfr(target=target_str)
    sfr_err = catalog_access.get_target_sfr_err(target=target_str)
    mstar = catalog_access.get_target_mstar(target=target_str)
    mstar_err = catalog_access.get_target_mstar_err(target=target_str)


    ms_x_pos_cc = ms_x_pos + (ms_x_len / (x_lim_high - x_lim_low)) * (np.log10(mstar) - x_lim_low) - contour_width/2
    ms_y_pos_cc = ms_y_pos + (ms_y_len / (y_lim_high - y_lim_low)) * (np.log10(sfr) - y_lim_low) - contour_hight / 2
    # print('ms_x_pos_cc ', ms_x_pos_cc)
    # print('ms_y_pos_cc ', ms_y_pos_cc)

    ax1 = fig.add_axes([ms_x_pos_cc + offset_dict[target][0], ms_y_pos_cc + offset_dict[target][1], contour_width, contour_hight])
    ax1.patch.set_alpha(0.5)
    if offset_dict[target] != [0, 0, 0, 0]:
        ax_ms.scatter(np.log10(mstar), np.log10(sfr), color='r', s=80)
        con_spec_1 = ConnectionPatch(
        xyA=(np.log10(np.array(mstar)),
             np.log10(np.array(sfr))), coordsA=ax_ms.transData,
        xyB=(offset_dict[target][2], offset_dict[target][3]), coordsB=ax1.transData,
        arrowstyle="<-",
        linewidth=2,
        color='k',
        mutation_scale=40
        )
        fig.add_artist(con_spec_1)
    else:
        log_stellar_mass_list.append(np.log10(mstar))
        log_sfr_list.append(np.log10(sfr))


    good_colors = (color_ub_ml_12 < 2) & (color_ub_ml_12 > -5.0) & (color_vi_ml_12 > -1.5) & (color_vi_ml_12 < 2.6)
    contours(ax=ax1, x=color_vi_ml_12[good_colors * detect_ub_ml_12 * detect_vi_ml_12],
             y=color_ub_ml_12[good_colors * detect_ub_ml_12 * detect_vi_ml_12],
             levels=[0.3, 0.5, 0.7, 0.8, 0.95, 0.99])
    pe = [patheffects.withStroke(linewidth=2, foreground="w")]
    ax1.text(-0.5, -2, target_name_str, fontsize=fontsize - 8, color='k', path_effects=pe)

    ax1.plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=3,  path_effects=pe)
    ax1.set_xlim(-1.0, 2.3)
    ax1.set_ylim(1.25, -2.5)
    ax1.axis('off')


pseudo_ax_ms = fig.add_axes([ms_x_pos, ms_y_pos, ms_x_len, ms_y_len])
#
pseudo_ax_ms.set_xlim(ax_ms.get_xlim())
pseudo_ax_ms.set_ylim(ax_ms.get_ylim())
#
pseudo_ax_ms.axis('off')
#
# pseudo_ax_ms.scatter(log_stellar_mass_list, log_sfr_list, color='dodgerblue', linewidth=3, marker='o', s=130, facecolor='None')
pseudo_ax_ms.scatter(log_stellar_mass_list, log_sfr_list, color='r', marker='*', s=200)

# plt.show()
#
# exit()

plt.savefig('plot_output/ms_cc.png')
plt.savefig('plot_output/ms_cc.pdf')


exit()




ax_hum[0, 0].set_ylim(1.25, -2.2)
ax_hum[0, 0].set_xlim(-1.0, 2.3)
fig_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_hum.text(0.5, 0.89, 'Class 1|2|3 Human', ha='center', fontsize=fontsize)

ax_ml[0, 0].set_ylim(1.25, -2.2)
ax_ml[0, 0].set_xlim(-1.0, 2.3)
fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/ub_vi_hum_1.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_1.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_1.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_1.pdf', bbox_inches='tight', dpi=300)


fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
fig_hum.set_size_inches(16, 18)
fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
fig_ml.set_size_inches(16, 18)
fontsize = 18

row_index = 0
col_index = 0
for index in range(20, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi(target=target, cluster_class='class3')

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_ml_12 == 1) & (cluster_class_qual_ml_12 >= 0.9)
    class_2_ml = (cluster_class_ml_12 == 2) & (cluster_class_qual_ml_12 >= 0.9)
    class_3_ml = (cluster_class_ml_3 == 3) & (cluster_class_qual_ml_3 >= 0.9)

    ax_hum[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_hum[row_index, col_index].scatter(color_vi_hum_3[cluster_class_hum_3 == 3],
                                         color_ub_hum_3[cluster_class_hum_3 == 3], c='royalblue', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 1],
                                         color_ub_hum_12[cluster_class_hum_12 == 1], c='forestgreen', s=1)
    ax_hum[row_index, col_index].scatter(color_vi_hum_12[cluster_class_hum_12 == 2],
                                         color_ub_hum_12[cluster_class_hum_12 == 2], c='darkorange', s=1)

    ax_ml[row_index, col_index].plot(model_vi, model_ub, color='red', linewidth=1.2)
    ax_ml[row_index, col_index].scatter(color_vi_ml_3[class_3_ml], color_ub_ml_3[class_3_ml], c='royalblue', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_1_ml], color_ub_ml_12[class_1_ml], c='forestgreen', s=1)
    ax_ml[row_index, col_index].scatter(color_vi_ml_12[class_2_ml], color_ub_ml_12[class_2_ml], c='darkorange', s=1)

    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)
    else:
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        anchored_left = AnchoredText(target.upper()+'$^*$'+'\nd='+str(dist)+' Mpc',
                                     loc='upper left', borderpad=0.1, frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)

    ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                            direction='in', labelsize=fontsize)

    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0


ax_hum[0, 0].set_ylim(1.25, -2.2)
ax_hum[0, 0].set_xlim(-1.0, 2.3)
fig_hum.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_hum.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_hum.text(0.5, 0.89, 'Class 1|2|3 Human', ha='center', fontsize=fontsize)
ax_hum[4, 3].axis('off')

ax_ml[0, 0].set_ylim(1.25, -2.2)
ax_ml[0, 0].set_xlim(-1.0, 2.3)
fig_ml.text(0.5, 0.08, 'V (F555W) - I (F814W)', ha='center', fontsize=fontsize)
fig_ml.text(0.08, 0.5, 'U (F336W) - B (F438W/F435W'+'$^*$'+')', va='center', rotation='vertical', fontsize=fontsize)
fig_ml.text(0.5, 0.89, 'Class 1|2|3 ML', ha='center', fontsize=fontsize)
ax_ml[4, 3].axis('off')

fig_hum.subplots_adjust(wspace=0, hspace=0)
fig_hum.savefig('plot_output/ub_vi_hum_2.png', bbox_inches='tight', dpi=300)
fig_hum.savefig('plot_output/ub_vi_hum_2.pdf', bbox_inches='tight', dpi=300)

fig_ml.subplots_adjust(wspace=0, hspace=0)
fig_ml.savefig('plot_output/ub_vi_ml_2.png', bbox_inches='tight', dpi=300)
fig_ml.savefig('plot_output/ub_vi_ml_2.pdf', bbox_inches='tight', dpi=300)

