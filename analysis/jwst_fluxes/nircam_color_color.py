"""
Extract Nircam flux of HST clusters if possible
"""
import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func as hf
from photometry_tools.plotting_tools import DensityContours
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib
from matplotlib.colorbar import ColorbarBase
import dust_tools.extinction_tools


def density_with_points(ax, x, y, binx=None, biny=None, threshold=1, kernel_std=2.0, save=False, save_name=''):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    if save:
        np.save('data_output/binx.npy', binx)
        np.save('data_output/biny.npy', biny)
        np.save('data_output/hist_%s_un_smoothed.npy' % save_name, hist)

    kernel = Gaussian2DKernel(x_stddev=kernel_std)
    hist = convolve(hist, kernel)

    if save:
        np.save('data_output/hist_%s_smoothed.npy' % save_name, hist)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask

    hist[hist <= threshold] = np.nan
    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap='inferno', interpolation='nearest')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.')
    ax.set_ylim(ax.get_ylim()[::-1])




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
flux_f275w_sol = data_mod_sol['F275W_UVIS_CHIP2']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
flux_f200w_sol = data_mod_sol['jwst.nircam.F200W']
flux_f300w_sol = data_mod_sol['jwst.nircam.F300M']
mag_vega_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_vega_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_vega_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_vega_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_vega_nuv_sol = hf.conv_mjy2vega(flux=flux_f275w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_ab_v_sol = hf.conv_mjy2ab_mag(flux=flux_f555w_sol)
mag_ab_i_sol = hf.conv_mjy2ab_mag(flux=flux_f814w_sol)
mag_ab_u_sol = hf.conv_mjy2ab_mag(flux=flux_f336w_sol)
mag_ab_b_sol = hf.conv_mjy2ab_mag(flux=flux_f438w_sol)
mag_ab_nuv_sol = hf.conv_mjy2ab_mag(flux=flux_f275w_sol)
ZP_v_200 = 759.59/1e6
ZP_v_300 = 3631./1e6
mag_vega_200_sol = -2.5*np.log10(flux_f200w_sol * 1e-9/ZP_v_200)
mag_vega_300_sol = -2.5*np.log10(flux_f300w_sol * 1e-9/ZP_v_300)
mag_ab_200_sol = hf.conv_mjy2ab_mag(flux_f200w_sol)
mag_ab_300_sol = hf.conv_mjy2ab_mag(flux_f300w_sol)

model_vega_vi_sol = mag_vega_v_sol - mag_vega_i_sol
model_vega_ub_sol = mag_vega_u_sol - mag_vega_b_sol
model_vega_v200_sol = mag_vega_v_sol - mag_vega_200_sol
model_vega_v300_sol = mag_vega_v_sol - mag_vega_300_sol
model_vega_200300_sol = mag_vega_200_sol - mag_vega_300_sol
model_vega_nuvb_sol = mag_vega_nuv_sol - mag_vega_b_sol
model_vega_nuvu_sol = mag_vega_nuv_sol - mag_vega_u_sol

model_ab_vi_sol = mag_ab_v_sol - mag_ab_i_sol
model_ab_ub_sol = mag_ab_u_sol - mag_ab_b_sol
model_ab_v200_sol = mag_ab_v_sol - mag_ab_200_sol
model_ab_v300_sol = mag_ab_v_sol - mag_ab_300_sol
model_ab_200300_sol = mag_ab_200_sol - mag_ab_300_sol
model_ab_nuvb_sol = mag_ab_nuv_sol - mag_ab_b_sol
model_ab_nuvu_sol = mag_ab_nuv_sol - mag_ab_u_sol


hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f275w_sol50 = data_mod_sol50['F275W_UVIS_CHIP2']
flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
flux_f814w_sol50 = data_mod_sol50['F814W_UVIS_CHIP2']
flux_f336w_sol50 = data_mod_sol50['F336W_UVIS_CHIP2']
flux_f438w_sol50 = data_mod_sol50['F438W_UVIS_CHIP2']
flux_f200w_sol50 = data_mod_sol50['jwst.nircam.F200W']
flux_f300w_sol50 = data_mod_sol50['jwst.nircam.F300M']
mag_vega_v_sol50 = hf.conv_mjy2vega(flux=flux_f555w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_vega_i_sol50 = hf.conv_mjy2vega(flux=flux_f814w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_vega_u_sol50 = hf.conv_mjy2vega(flux=flux_f336w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_vega_b_sol50 = hf.conv_mjy2vega(flux=flux_f438w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_vega_nuv_sol50 = hf.conv_mjy2vega(flux=flux_f275w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_ab_v_sol50 = hf.conv_mjy2ab_mag(flux=flux_f555w_sol50)
mag_ab_i_sol50 = hf.conv_mjy2ab_mag(flux=flux_f814w_sol50)
mag_ab_u_sol50 = hf.conv_mjy2ab_mag(flux=flux_f336w_sol50)
mag_ab_b_sol50 = hf.conv_mjy2ab_mag(flux=flux_f438w_sol50)
mag_ab_nuv_sol50 = hf.conv_mjy2ab_mag(flux=flux_f275w_sol50)
ZP_v_200 = 759.59/1e6
ZP_v_300 = 3631./1e6
mag_vega_200_sol50 = -2.5*np.log10(flux_f200w_sol50 * 1e-9/ZP_v_200)
mag_vega_300_sol50 = -2.5*np.log10(flux_f300w_sol50 * 1e-9/ZP_v_300)
mag_ab_200_sol50 = hf.conv_mjy2ab_mag(flux_f200w_sol50)
mag_ab_300_sol50 = hf.conv_mjy2ab_mag(flux_f300w_sol50)

model_vega_vi_sol50 = mag_vega_v_sol50 - mag_vega_i_sol50
model_vega_ub_sol50 = mag_vega_u_sol50 - mag_vega_b_sol50
model_vega_v200_sol50 = mag_vega_v_sol50 - mag_vega_200_sol50
model_vega_v300_sol50 = mag_vega_v_sol50 - mag_vega_300_sol50
model_vega_200300_sol50 = mag_vega_200_sol50 - mag_vega_300_sol50
model_vega_nuvb_sol50 = mag_vega_nuv_sol50 - mag_vega_b_sol50
model_vega_nuvu_sol50 = mag_vega_nuv_sol50 - mag_vega_u_sol50

model_ab_vi_sol50 = mag_ab_v_sol50 - mag_ab_i_sol50
model_ab_ub_sol50 = mag_ab_u_sol50 - mag_ab_b_sol50
model_ab_v200_sol50 = mag_ab_v_sol50 - mag_ab_200_sol50
model_ab_v300_sol50 = mag_ab_v_sol50 - mag_ab_300_sol50
model_ab_200300_sol50 = mag_ab_200_sol50 - mag_ab_300_sol50
model_ab_nuvb_sol50 = mag_ab_nuv_sol50 - mag_ab_b_sol50
model_ab_nuvu_sol50 = mag_ab_nuv_sol50 - mag_ab_u_sol50


np.save('data_output/model_tracks/model_vega_vi_sol.npy', model_vega_vi_sol)
np.save('data_output/model_tracks/model_vega_ub_sol.npy', model_vega_ub_sol)
np.save('data_output/model_tracks/model_vega_v200_sol.npy', model_vega_v200_sol)
np.save('data_output/model_tracks/model_vega_v300_sol.npy', model_vega_v300_sol)
np.save('data_output/model_tracks/model_vega_200300_sol.npy', model_vega_200300_sol)
np.save('data_output/model_tracks/model_vega_nuvb_sol.npy', model_vega_nuvb_sol)
np.save('data_output/model_tracks/model_vega_nuvu_sol.npy', model_vega_nuvu_sol)
np.save('data_output/model_tracks/model_ab_vi_sol.npy', model_ab_vi_sol)
np.save('data_output/model_tracks/model_ab_ub_sol.npy', model_ab_ub_sol)
np.save('data_output/model_tracks/model_ab_v200_sol.npy', model_ab_v200_sol)
np.save('data_output/model_tracks/model_ab_v300_sol.npy', model_ab_v300_sol)
np.save('data_output/model_tracks/model_ab_200300_sol.npy', model_ab_200300_sol)
np.save('data_output/model_tracks/model_ab_nuvb_sol.npy', model_ab_nuvb_sol)
np.save('data_output/model_tracks/model_ab_nuvu_sol.npy', model_ab_nuvu_sol)
np.save('data_output/model_tracks/model_vega_vi_sol50.npy', model_vega_vi_sol50)
np.save('data_output/model_tracks/model_vega_ub_sol50.npy', model_vega_ub_sol50)
np.save('data_output/model_tracks/model_vega_v200_sol50.npy', model_vega_v200_sol50)
np.save('data_output/model_tracks/model_vega_v300_sol50.npy', model_vega_v300_sol50)
np.save('data_output/model_tracks/model_vega_200300_sol50.npy', model_vega_200300_sol50)
np.save('data_output/model_tracks/model_vega_nuvb_sol50.npy', model_vega_nuvb_sol50)
np.save('data_output/model_tracks/model_vega_nuvu_sol50.npy', model_vega_nuvu_sol50)
np.save('data_output/model_tracks/model_ab_vi_sol50.npy', model_ab_vi_sol50)
np.save('data_output/model_tracks/model_ab_ub_sol50.npy', model_ab_ub_sol50)
np.save('data_output/model_tracks/model_ab_v200_sol50.npy', model_ab_v200_sol50)
np.save('data_output/model_tracks/model_ab_v300_sol50.npy', model_ab_v300_sol50)
np.save('data_output/model_tracks/model_ab_200300_sol50.npy', model_ab_200300_sol50)

exit()


# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc7496',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')


color_vega_v200 = np.array([])
color_vega_ub = np.array([])
color_ab_v200 = np.array([])
color_ab_ub = np.array([])
cluster_class = np.array([])
cluster_age = np.array([])


for target in ['ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672',
               'ngc3627', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496']:
    print(target)
    if target == 'ngc0628':
        target = 'ngc0628c'

    flux_dict = np.load('data_output/flux_dict_%s.npy' % target, allow_pickle=True).item()

    flux_200w = flux_dict['flux_f200w']
    ZP_v_200 = 759.59/1e6
    mag_vega_200 = -2.5*np.log10(flux_200w * 1e-9/ZP_v_200)
    mag_vega_v = flux_dict['mag_vega_v_list']
    color_vega_ub_list = flux_dict['color_vega_ub_list']
    color_vega_v200 = np.concatenate([color_vega_v200, mag_vega_v-mag_vega_200])
    color_vega_ub = np.concatenate([color_vega_ub, color_vega_ub_list])

    # mag_ab_200 = hf.conv_mjy2ab_mag(flux=flux_200w)
    mag_ab_200 = flux_dict['mag_f200w']
    mag_ab_v = flux_dict['mag_ab_v_list']
    color_ab_ub_list = flux_dict['color_ab_ub_list']
    color_ab_v200 = np.concatenate([color_ab_v200, mag_ab_v-mag_ab_200])
    color_ab_ub = np.concatenate([color_ab_ub, color_ab_ub_list])

    cluster_class = np.concatenate([cluster_class, flux_dict['class_list']])
    cluster_age = np.concatenate([cluster_age, flux_dict['age_list']])


class_1 = cluster_class == 1
class_2 = cluster_class == 2
class_3 = cluster_class == 3

young = cluster_age < 10
inter = (cluster_age >= 10) & (cluster_age < 100)
old = cluster_age > 100

# good_data = (color > -1) & (color < 4) & (abs_mag_f300m > -26) & (abs_mag_f300m < -17)

mask_good_colors_ab = ((color_ab_v200 > -10) & (color_ab_v200 < 3) & (color_ab_ub > -3) & (color_ab_ub < 4) &
                       np.invert(((np.isnan(color_ab_v200) | np.isnan(color_ab_ub)) | (np.isinf(color_ab_v200) | np.isinf(color_ab_ub)))))

figure = plt.figure(figsize=(18, 17))
fontsize = 26
ax_cc = figure.add_axes([0.08, 0.07, 0.9, 0.9])
ax_cbar = figure.add_axes([0.75, 0.9, 0.2, 0.015])

cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.Normalize(vmin=6, vmax=9)

ax_cc.scatter(color_ab_v200[(class_1+class_2)*mask_good_colors_ab],
              color_ab_ub[(class_1+class_2)*mask_good_colors_ab],
              c=np.log10(cluster_age[(class_1+class_2)*mask_good_colors_ab])+6, cmap=cmap, norm=norm)


ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_xlabel(r'log(Age)', labelpad=4, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)

# ax_cc.plot(model_v200_sol, model_ub_sol, linewidth=4)
# ax_cc.plot(model_v200_sol50, model_ub_sol50, linewidth=4)
ax_cc.plot(model_ab_v200_sol, model_ab_ub_sol, color='darkred', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}$')
ax_cc.plot(model_ab_v200_sol50, model_ab_ub_sol50, color='darkorange', linewidth=4, linestyle='-', label=r'BC03, Z$_{\odot}/50$')


x_lim = (-9.5, 4)
y_lim = (2.5, -0.5)
ax_cc.set_xlim(x_lim)
ax_cc.set_ylim(y_lim)

ax_cc.set_title('Class 1|2 HUM', fontsize=fontsize)

ax_cc.set_xlabel('V (F555W) - K (F200W) AB', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+') AB', fontsize=fontsize)

ax_cc.legend(frameon=False, loc=3, fontsize=fontsize)
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



v200_int = -8
ub_int = 0.5
max_av = 2
v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
f200w_wave = catalog_access.nircam_bands_mean_wave['F200W']*1e-4
u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
max_color_ext_v200 = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=f200w_wave, av=max_av)
max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)
max_color_ext_v200_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=f200w_wave, av=max_av+0.1)
max_color_ext_ub_arr = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av+0.1)

ax_cc.annotate('', xy=(v200_int + max_color_ext_v200, ub_int + max_color_ext_ub), xycoords='data',
                                      xytext=(v200_int, ub_int), fontsize=fontsize,
                                      textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=2, ls='-'))


# plt.show(
plt.savefig('plot_output/test_color.png')


exit()





fig, ax = plt.subplots(figsize=(15, 11))
fontsize = 17

DensityContours.get_three_contours_percentage(ax=ax,
                                              x_data_1=color[mask_snr * good_data * young],
                                              y_data_1=abs_mag_f335m[mask_snr * good_data * young],
                                              x_data_2=color[mask_snr * good_data * inter],
                                              y_data_2=abs_mag_f335m[mask_snr * good_data * inter],
                                              x_data_3=color[mask_snr * good_data * old],
                                              y_data_3=abs_mag_f335m[mask_snr * good_data * old],
                                              color_1='k', color_2='tab:orange', color_3='c',
                                              percent_1=False, percent_2=False, percent_3=False)


ax.plot([], [], color='k', label='young')
ax.plot([], [], color='tab:orange', label='inter age')
ax.plot([], [], color='c', label='old')
ax.scatter(color_emb[emb==0], abs_mag_f_335m_emb[emb==0], color='r', label='Embedded')
ax.scatter(color_emb[emb==1], abs_mag_f_335m_emb[emb==1], color='green', label='Intermediate')
ax.scatter(color_emb[emb==2], abs_mag_f_335m_emb[emb==2], color='blue', label='Visible')

ax.legend(frameon=False, fontsize=fontsize)

ax.invert_yaxis()
ax.set_xlim(-0.85, 2.2)
ax.set_ylabel('Abs. mag F335M', fontsize=fontsize)
ax.set_xlabel('F300M - F335M', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



plt.savefig('plot_output/color_mag_2.png')


