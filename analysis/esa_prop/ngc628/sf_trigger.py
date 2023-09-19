
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from matplotlib.patches import ConnectionPatch


target_name_ml = np.load('../../color_color/data_output/target_name_ml.npy')
x_ml = np.load('../../color_color/data_output/x_ml.npy')
y_ml = np.load('../../color_color/data_output/y_ml.npy')
ra_ml = np.load('../../color_color/data_output/ra_ml.npy')
dec_ml = np.load('../../color_color/data_output/dec_ml.npy')
age_ml = np.load('../../color_color/data_output/age_ml.npy')
mass_ml = np.load('../../color_color/data_output/mass_ml.npy')

age_hum = np.load('../../color_color/data_output/age_hum.npy')
mass_hum = np.load('../../color_color/data_output/mass_hum.npy')


print(sum((mass_ml > 1e5) & (age_ml < 10)))
print(sum((mass_hum > 1e5) & (age_hum < 10)))

exit()


target = 'ngc0628c'
galaxy_name = 'ngc0628'

target_mask = (target_name_ml == target)
mask_old_ml = (age_ml > 10000) & target_mask
mask_mid_ml = ((age_ml > 30) & (age_ml < 100)) & target_mask
mask_young_ml = (age_ml < 5) & target_mask

# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
# band_list = ['F336W', 'F435W', 'F555W']
# band_list = ['F336W', 'F435W', 'F555W', 'F658N', 'F814W', 'F200W', 'F335M', 'F360M', 'F770W', 'F1000W', 'F2100W']
band_list = ['F336W', 'F435W', 'F555W', 'F658N', 'F200W', 'F335M', 'F360M']
# get the center of the region of interest
simbad_table = Simbad.query_object(galaxy_name)
central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                               unit=(u.hourangle, u.deg))
cutout_size = (90, 90)

phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=band_list, load_err=False)
# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg) - 10*u.arcsec,
                                                               dec_cutout=central_coordinates.dec.to(u.deg) + 10*u.arcsec,
                                                               cutout_size=cutout_size, include_err=False,
                                                               band_list=band_list)
red_color_hst = '#EE4B2B'
green_color_hst = '#FFF9DB'
blue_color_hst = '#0096FF'
ha_color_hst = '#FF00FF'
# red_color_hst = '#FF2400'
# green_color_hst = '#0FFF50'
# blue_color_hst = '#0096FF'
# ha_color_hst = '#FF00FF'

red_color_nircam = '#FF2400'
green_color_nircam = '#0FFF50'
blue_color_nircam = '#0096FF'
# ha_color_nircam = '#FF00FF'

rgb_hst_img = helper_func.create3color_rgb_img(img_r=cutout_dict_large_rgb['F555W_img_cutout'].data,
                                                         img_g=cutout_dict_large_rgb['F435W_img_cutout'].data,
                                                         img_b=cutout_dict_large_rgb['F336W_img_cutout'].data,
                                                         gamma=2.2, min_max=[0.01, 99.7], gamma_rgb=2.2,
                                                         red_color=red_color_hst, green_color=green_color_hst,
                                                         blue_color=blue_color_hst,
                                                         mask_no_coverage=True)


# coords_obj_world = SkyCoord(ra=24.163632754905947*u.deg, dec=15.795263780630586*u.deg + 1*u.arcsec,)

coords_obj_world = SkyCoord('1h36m39.15s 15d47m43.0s', unit=(u.hourangle, u.deg))


coords_obj_pix = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(coords_obj_world)
# load all data
cutout_size_small = (9, 9)
# get a cutout
cutout_dict_zoom_in = phangs_photometry.get_band_cutout_dict(ra_cutout=coords_obj_world.ra.to(u.deg),
                                                             dec_cutout=coords_obj_world.dec.to(u.deg),
                                                             cutout_size=cutout_size_small, include_err=False,
                                                             band_list=band_list)

alma_hdu = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' %
                     (galaxy_name, galaxy_name))
cutout_alma = helper_func.get_img_cutout(img=alma_hdu[0].data, wcs=WCS(alma_hdu[0].header),
                                         coord=coords_obj_world, cutout_size=cutout_size_small)
alma_zoom_in = plotting_tools.reproject_image(data=cutout_alma.data,
                                              wcs=cutout_alma.wcs,
                                              new_wcs=cutout_dict_zoom_in['F555W_img_cutout'].wcs,
                                              new_shape=cutout_dict_zoom_in['F555W_img_cutout'].data.shape)


rgb_hst_img_zoom_in = helper_func.create4color_rgb_img(img_r=cutout_dict_zoom_in['F555W_img_cutout'].data,
                                                         img_g=cutout_dict_zoom_in['F435W_img_cutout'].data,
                                                         img_b=cutout_dict_zoom_in['F336W_img_cutout'].data,
                                                         img_p=cutout_dict_zoom_in['F658N_img_cutout'].data,
                                                         gamma=2.8, min_max=[0.3, 99.5], gamma_rgb=2.8,
                                                         red_color=red_color_hst, green_color=green_color_hst,
                                                         blue_color=blue_color_hst, pink_color=ha_color_hst,
                                                         mask_no_coverage=False)

new_wcs = cutout_dict_zoom_in['F200W_img_cutout'].wcs
new_shape = cutout_dict_zoom_in['F200W_img_cutout'].data.shape
nircam_img_r = plotting_tools.reproject_image(data=cutout_dict_zoom_in['F200W_img_cutout'].data,
                                              wcs=cutout_dict_zoom_in['F200W_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_g = plotting_tools.reproject_image(data=cutout_dict_zoom_in['F335M_img_cutout'].data,
                                              wcs=cutout_dict_zoom_in['F335M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_b = plotting_tools.reproject_image(data=cutout_dict_zoom_in['F360M_img_cutout'].data,
                                              wcs=cutout_dict_zoom_in['F360M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)

rgb_nircam_img_zoom_in = helper_func.create3color_rgb_img(img_r=nircam_img_r,
                                                            img_g=nircam_img_g,
                                                            img_b=nircam_img_b,
                                                            gamma=2.8, min_max=[0.3, 99.5], gamma_rgb=2.8,
                                                            red_color=red_color_nircam, green_color=green_color_nircam,
                                                            blue_color=blue_color_nircam,
                                                            mask_no_coverage=False)


coords_young_1 = SkyCoord('1h36m39.070s 15d47m43.61s', unit=(u.hourangle, u.deg))
coords_young_2 = SkyCoord('1h36m39.094s 15d47m42.24s', unit=(u.hourangle, u.deg))
coords_young_3 = SkyCoord('1h36m39.109s 15d47m41.27s', unit=(u.hourangle, u.deg))

coords_embedded_1 = SkyCoord('1h36m39.050s 15d47m44.21s', unit=(u.hourangle, u.deg))
coords_embedded_2 = SkyCoord('1h36m38.99s 15d47m44.5s', unit=(u.hourangle, u.deg))
coords_embedded_3 = SkyCoord('1h36m39.0s 15d47m44.2s', unit=(u.hourangle, u.deg))
coords_embedded_4 = SkyCoord('1h36m39.019s 15d47m41.42s', unit=(u.hourangle, u.deg))
coords_embedded_5 = SkyCoord('1h36m39.0s 15d47m44.2s', unit=(u.hourangle, u.deg))
coords_embedded_6 = SkyCoord('1h36m39.02s 15d47m43.6s', unit=(u.hourangle, u.deg))
coords_embedded_7 = SkyCoord('1h36m39.019s 15d47m42.2s', unit=(u.hourangle, u.deg))
coords_embedded_8 = SkyCoord('1h36m39.03s 15d47m40.8s', unit=(u.hourangle, u.deg))
coords_embedded_9 = SkyCoord('1h36m39.07s 15d47m40.6s', unit=(u.hourangle, u.deg))
coords_embedded_10 = SkyCoord('1h36m39.019s 15d47m42.2s', unit=(u.hourangle, u.deg))
coords_embedded_11 = SkyCoord('1h36m39.10s 15d47m40.5s', unit=(u.hourangle, u.deg))

coords_mid_1 = SkyCoord('1h36m39.18s 15d47m43.6s', unit=(u.hourangle, u.deg))
coords_mid_2 = SkyCoord('1h36m39.274s 15d47m42.96s', unit=(u.hourangle, u.deg))
coords_mid_3 = SkyCoord('1h36m39.29s 15d47m42.4s', unit=(u.hourangle, u.deg))


figure = plt.figure(figsize=(43, 15))
fontsize = 40
ax_hst_overview = figure.add_axes([0.03, 0.04, 0.31, 0.95], projection=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
ax_zoom_in_hst = figure.add_axes([0.36, 0.04, 0.31, 0.95], projection=cutout_dict_zoom_in['F555W_img_cutout'].wcs)
ax_zoom_in_nircam = figure.add_axes([0.68, 0.04, 0.31, 0.95], projection=new_wcs)

ax_hst_overview.imshow(rgb_hst_img)
ax_zoom_in_hst.imshow(rgb_hst_img_zoom_in)
ax_zoom_in_hst.contour(alma_zoom_in, linewidth=2)
ax_zoom_in_nircam.imshow(rgb_nircam_img_zoom_in)


coords_mid_1_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_mid_1)
coords_mid_2_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_mid_2)
coords_mid_3_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_mid_3)
ax_zoom_in_hst.annotate(' ', xy=(coords_mid_1_pix[0] - 6, coords_mid_1_pix[1] + 6),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.25, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.70),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
ax_zoom_in_hst.annotate(' ', xy=(coords_mid_2_pix[0], coords_mid_2_pix[1] + 8),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.25, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.70),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
ax_zoom_in_hst.annotate(' ', xy=(coords_mid_3_pix[0], coords_mid_3_pix[1] + 8),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.25, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.70),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
t = ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.25, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.70,
                        r'No H$\alpha$ ~7 Myr', horizontalalignment='center', verticalalignment='center',
                        color='cyan', fontsize=fontsize, zorder=40)
t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='k'))


coords_young_1_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_young_1)
coords_young_2_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_young_2)
coords_young_3_pix = cutout_dict_zoom_in['F555W_img_cutout'].wcs.world_to_pixel(coords_young_3)
ax_zoom_in_hst.annotate(' ', xy=(coords_young_1_pix[0] - 6, coords_young_1_pix[1]),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.35, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.25),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
ax_zoom_in_hst.annotate(' ', xy=(coords_young_2_pix[0] - 6, coords_young_2_pix[1]),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.35, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.25),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
ax_zoom_in_hst.annotate(' ', xy=(coords_young_3_pix[0] - 6, coords_young_3_pix[1]),
                        xytext=(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.35, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.25),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='k', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='-|>', color='k', lw=3, ls='-'))
t = ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.35, rgb_hst_img_zoom_in[:,:,0].shape[1]*0.25,
                        r'H$\alpha$ ~1 Myr', horizontalalignment='right', verticalalignment='center',
                        color='blue', fontsize=fontsize, zorder=40)
t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='k'))


t = ax_zoom_in_nircam.text(rgb_nircam_img_zoom_in[:, :, 0].shape[0]*0.70, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.75,
                        r'embedded < 1 Myr', horizontalalignment='center', verticalalignment='center',
                        color='red', fontsize=fontsize, zorder=40)
t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='k'))

ax_zoom_in_nircam.annotate(' ', xy=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.57, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.55),
                        xytext=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.50, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.55),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='cyan', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='fancy', color='cyan', lw=3, ls='-'))
ax_zoom_in_nircam.annotate(' ', xy=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.62, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.48),
                        xytext=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.55, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.50),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='cyan', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='fancy', color='cyan', lw=3, ls='-'))
ax_zoom_in_nircam.annotate(' ', xy=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.58, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.35),
                        xytext=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.53, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.38),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='cyan', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='fancy', color='cyan', lw=3, ls='-'))
ax_zoom_in_nircam.annotate(' ', xy=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.52, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.25),
                        xytext=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.47, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.32),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='cyan', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='fancy', color='cyan', lw=3, ls='-'))
ax_zoom_in_nircam.annotate(' ', xy=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.44, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.23),
                        xytext=(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.42, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.31),
                        fontsize=fontsize, xycoords='data', textcoords='data', color='cyan', ha='center', va='center',
                        zorder=30, arrowprops=dict(arrowstyle='fancy', color='cyan', lw=3, ls='-'))

t = ax_zoom_in_nircam.text(rgb_nircam_img_zoom_in[:, :, 0].shape[0]*0.40, rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.40,
                        'SF \n Feedback', horizontalalignment='center', verticalalignment='center',
                        color='cyan', fontsize=fontsize, zorder=40,
                           rotation=40)
t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='k'))



plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_young_1, rad=0.3, color='blue', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_young_2, rad=0.3, color='blue', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_young_3, rad=0.3, color='blue', linewidth=3)

plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_1, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_2, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_3, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_4, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_5, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_6, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_7, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_8, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_9, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_10, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_embedded_11, rad=0.3, color='red', linewidth=3)

plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_mid_1, rad=0.3, color='cyan', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_mid_2, rad=0.3, color='cyan', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_hst, pos=coords_mid_3, rad=0.3, color='cyan', linewidth=3)



plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_young_1, rad=0.3, color='blue', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_young_2, rad=0.3, color='blue', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_young_3, rad=0.3, color='blue', linewidth=3)

plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_1, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_2, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_3, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_4, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_5, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_6, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_7, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_8, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_9, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_10, rad=0.3, color='red', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_embedded_11, rad=0.3, color='red', linewidth=3)

plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_mid_1, rad=0.3, color='cyan', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_mid_2, rad=0.3, color='cyan', linewidth=3)
plotting_tools.plot_coord_circle(ax=ax_zoom_in_nircam, pos=coords_mid_3, rad=0.3, color='cyan', linewidth=3)






plotting_tools.draw_box(ax=ax_hst_overview, wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs,
                        coord=coords_obj_world, box_size=cutout_size_small,
                        color='white', linewidth=2, linestyle='--')
pos_edge_1 = SkyCoord(ra=coords_obj_world.ra.to(u.deg) + (cutout_size_small[0]*u.arcsec / 2)/np.cos(coords_obj_world.dec.degree*np.pi/180),
                      dec=coords_obj_world.dec.to(u.deg) - cutout_size_small[1]*u.arcsec / 2)
pos_pix_edge_1 = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_edge_1)
con_spec_edge_1 = ConnectionPatch(
    xyA=(pos_pix_edge_1[0], pos_pix_edge_1[1]), coordsA=ax_hst_overview.transData,
    xyB=(ax_zoom_in_hst.get_xlim()[0], ax_zoom_in_hst.get_ylim()[0]), coordsB=ax_zoom_in_hst.transData,
    linestyle="-", linewidth=3, color='grey')
figure.add_artist(con_spec_edge_1)

pos_edge_2 = SkyCoord(ra=coords_obj_world.ra.to(u.deg) + (cutout_size_small[0]*u.arcsec / 2)/np.cos(coords_obj_world.dec.degree*np.pi/180),
                      dec=coords_obj_world.dec.to(u.deg) + cutout_size_small[1]*u.arcsec / 2)
pos_pix_edge_2 = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_edge_2)
con_spec_edge_2 = ConnectionPatch(
    xyA=(pos_pix_edge_2[0], pos_pix_edge_2[1]), coordsA=ax_hst_overview.transData,
    xyB=(ax_zoom_in_hst.get_xlim()[0], ax_zoom_in_hst.get_ylim()[1]), coordsB=ax_zoom_in_hst.transData,
    linestyle="-", linewidth=3, color='grey')
figure.add_artist(con_spec_edge_2)


ax_hst_overview.set_title('NGC 628  HST rgb Image', fontsize=fontsize)
ax_zoom_in_hst.set_title(r'HST + H$\alpha$ + CO(2-1)', fontsize=fontsize)
ax_zoom_in_nircam.set_title('JWST NIRCAM', fontsize=fontsize)


ax_hst_overview.text(rgb_hst_img[:,:,0].shape[0]*0.05,
                rgb_hst_img[:,:,0].shape[1]*0.95,
                'U F336W', color=blue_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_hst_overview.text(rgb_hst_img[:,:,0].shape[0]*0.05,
                rgb_hst_img[:,:,0].shape[1]*0.91,
                'B F438W', color=green_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_hst_overview.text(rgb_hst_img[:,:,0].shape[0]*0.05,
                rgb_hst_img[:,:,0].shape[1]*0.87,
                'V F555W', color=red_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.75,
                rgb_hst_img_zoom_in[:,:,0].shape[1]*0.95,
                'U F336W', color=blue_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.75,
                rgb_hst_img_zoom_in[:,:,0].shape[1]*0.91,
                'B F438W', color=green_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.75,
                rgb_hst_img_zoom_in[:,:,0].shape[1]*0.87,
                'V F555W', color=red_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_zoom_in_hst.text(rgb_hst_img_zoom_in[:,:,0].shape[0]*0.75,
                rgb_hst_img_zoom_in[:,:,0].shape[1]*0.83,
                r'H$\alpha$ F658N', color=ha_color_hst, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


ax_zoom_in_nircam.text(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.05,
                rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.95,
                'K F200W', color=blue_color_nircam, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_zoom_in_nircam.text(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.05,
                rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.91,
                'F335M PAH 3.3 $\mu$m', color=green_color_nircam, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax_zoom_in_nircam.text(rgb_nircam_img_zoom_in[:,:,0].shape[0]*0.05,
                rgb_nircam_img_zoom_in[:,:,0].shape[1]*0.87,
                'F360M', color=red_color_nircam, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


plotting_tools.arr_axis_params(ax=ax_hst_overview, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.3, dec_minpad=0.0, tick_color='white', label_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_zoom_in_hst, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                    ra_minpad=0.3, dec_minpad=0.4, tick_color='white', label_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)
plotting_tools.arr_axis_params(ax=ax_zoom_in_nircam, ra_tick_label=True, dec_tick_label=False,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                    ra_minpad=0.3, dec_minpad=0.4, tick_color='white', label_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)

ax_zoom_in_hst.grid(color='grey', ls='solid')
ax_zoom_in_nircam.grid(color='grey', ls='solid')

# plt.show()

plt.savefig('plot_output/sf_trigger.png')
plt.savefig('plot_output/sf_trigger.pdf')

exit()



