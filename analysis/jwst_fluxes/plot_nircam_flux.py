"""
Extract Nircam flux of HST clusters if possible
"""
import numpy as np
import photometry_tools
import matplotlib.pyplot as plt

from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func as hf
from photometry_tools.plotting_tools import DensityContours



emb_cluster_cat = '/home/benutzer/data/PHANGS_products/emb_clusters/phot_50per_ngc7496_candidates_v2p4.dat'

cluster_list = np.genfromtxt(emb_cluster_cat, dtype=object)
names = cluster_list[0]

data = np.array(cluster_list[1:], dtype=float)

ra_cat = data[:, names == b'raj2000']
dec_cat = data[:, names == b'dej2000']
emb = data[:, names == b'EMB']

flux_f300m = data[:, names == b'flux_F300M_50'] * 1e9
flux_f335m = data[:, names == b'flux_F335M_50'] * 1e9


# print(flux_f300m)




size_of_cutout = (3, 3)
band_list = ['F300M', 'F335M']
# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc7496',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')


mag_f300m_emb = hf.conv_mjy2ab_mag(flux=flux_f300m)
mag_f335m_emb = hf.conv_mjy2ab_mag(flux=flux_f335m)

color_emb = mag_f300m_emb - mag_f335m_emb
abs_mag_f300m_emb = hf.conv_mag2abs_mag(mag=mag_f300m_emb, dist=phangs_photometry.dist_dict['ngc7496']['dist'])
abs_mag_f_335m_emb = hf.conv_mag2abs_mag(mag=mag_f335m_emb, dist=phangs_photometry.dist_dict['ngc7496']['dist'])



#
# color = np.zeros(len(ra_cat))
# abs_mag = np.zeros(len(ra_cat))
#
# for index in range(len(ra_cat)):
#
#     print('ra_cat[index] ', ra_cat[index])
#     print('dec_cat[index] ', dec_cat[index])
#     phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy', band_list=band_list, load_err=False)
#
#     cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_cat[index][0],
#                                                          dec_cutout=dec_cat[index][0], cutout_size=size_of_cutout,
#                                                          band_list=band_list, include_err=False)
#     source = SkyCoord(ra=ra_cat[index][0], dec=dec_cat[index][0], unit=(u.degree, u.degree), frame='fk5')
#
#
#
#     # compute flux from 50% encircled energy
#     aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
#                                                                           recenter=True, recenter_rad=0.01,
#                                                                           default_ee_rad=50)
#
#     flux_f300m = aperture_dict['aperture_dict_F300M']['flux']
#     flux_f300m_err = aperture_dict['aperture_dict_F300M']['flux_err']
#
#     flux_f335m = aperture_dict['aperture_dict_F335M']['flux']
#     flux_f335m_err = aperture_dict['aperture_dict_F335M']['flux_err']
#
#     mag_f300m = hf.conv_mjy2ab_mag(flux=flux_f300m)
#     mag_f335m = hf.conv_mjy2ab_mag(flux=flux_f335m)
#
#     color[index] = mag_f300m - mag_f335m
#     abs_mag[index] = hf.conv_mag2abs_mag(mag=mag_f300m, dist=phangs_photometry.dist_dict['ngc7496']['dist'])
#
#
#
# print('color ', color)
# print('abs_mag ', abs_mag)
#
# exit()



color = np.array([])
snr_f300m = np.array([])
snr_f335m = np.array([])
abs_mag_f300m = np.array([])
abs_mag_f335m = np.array([])
cluster_class = np.array([])
cluster_age = np.array([])


for target in ['ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672',
               'ngc3627', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496']:
    print(target)

    flux_dict = np.load('data_output/flux_dict_%s.npy' % target, allow_pickle=True).item()

    color = np.concatenate([color, flux_dict['mag_f300m'] - flux_dict['mag_f335m']])
    abs_mag_f300m = np.concatenate([abs_mag_f300m, flux_dict['abs_mag_f300m']])
    abs_mag_f335m = np.concatenate([abs_mag_f335m, flux_dict['abs_mag_f335m']])

    snr_f300m = np.concatenate([snr_f300m, flux_dict['flux_f300m'] / flux_dict['flux_f300m_err']])
    snr_f335m = np.concatenate([snr_f335m, flux_dict['flux_f335m'] / flux_dict['flux_f335m_err']])
    cluster_class = np.concatenate([cluster_class, flux_dict['class_list']])
    cluster_age = np.concatenate([cluster_age, flux_dict['age_list']])

mask_snr = (snr_f300m > 3) & (snr_f335m > 3)


class_1 = cluster_class == 1
class_2 = cluster_class == 2
class_3 = cluster_class == 3

young = cluster_age < 10
inter = (cluster_age >= 10) & (cluster_age < 100)
old = cluster_age > 100

good_data = (color > -1) & (color < 4) & (abs_mag_f300m > -26) & (abs_mag_f300m < -17)


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


