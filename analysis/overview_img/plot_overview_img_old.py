


import multicolorfits as mcf







exit()


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

from astropy.visualization import make_lupton_rgb
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad


########### load data
# get hst and JWST data
target = 'ngc0628'
galaxy_name = 'ngc0628'
# get the center of the region of interest
simbad_table = Simbad.query_object(galaxy_name)
central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                               unit=(u.hourangle, u.deg))

cutout_size = (130, 130)


band_list = ['F275W', 'F336W', 'F435W', 'F555W', 'F658N', 'F814W', 'F200W', 'F335M', 'F360M', 'F770W', 'F1000W', 'F2100W']

# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=band_list, load_err=False)

# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
                                                               dec_cutout=central_coordinates.dec.to(u.deg),
                                                               cutout_size=cutout_size, include_err=False,
                                                               band_list=band_list)


hdu_muse = fits.open('data/NGC0628_MAPS.fits')
cutout_muse_r = helper_func.get_img_cutout(img=hdu_muse['HA6562_FLUX'].data, wcs=WCS(hdu_muse['HA6562_FLUX'].header),
                                           coord=central_coordinates, cutout_size=cutout_size)
cutout_muse_g = helper_func.get_img_cutout(img=hdu_muse['NII6583_FLUX'].data, wcs=WCS(hdu_muse['NII6583_FLUX'].header),
                                           coord=central_coordinates, cutout_size=cutout_size)
cutout_muse_b = helper_func.get_img_cutout(img=hdu_muse['OIII5006_FLUX'].data,
                                           wcs=WCS(hdu_muse['OIII5006_FLUX'].header),
                                           coord=central_coordinates, cutout_size=cutout_size)


alma_hdu = fits.open('data/ngc0628_12m+7m+tp_co21_broad_tpeak.fits')
cutout_alma = helper_func.get_img_cutout(img=alma_hdu[0].data, wcs=WCS(alma_hdu[0].header),
                                         coord=central_coordinates, cutout_size=cutout_size)


# new_wcs = cutout_alma.wcs
# new_shape = cutout_alma.data.shape
new_wcs = cutout_dict_large_rgb['F200W_img_cutout'].wcs
new_shape = cutout_dict_large_rgb['F200W_img_cutout'].data.shape

# # reproject everything to F200W image
hst_img_r = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F814W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F814W_img_cutout'].wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)
hst_img_g = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F336W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F336W_img_cutout'].wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)
hst_img_b = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F275W_img_cutout'].data,
                                           wcs=cutout_dict_large_rgb['F275W_img_cutout'].wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)

hst_h_alpha_r = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F658N_img_cutout'].data,
                                               wcs=cutout_dict_large_rgb['F658N_img_cutout'].wcs,
                                               new_wcs=new_wcs,
                                               new_shape=new_shape)
hst_h_alpha_g = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F555W_img_cutout'].data,
                                               wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs,
                                               new_wcs=new_wcs,
                                               new_shape=new_shape)
hst_h_alpha_b = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F435W_img_cutout'].data,
                                               wcs=cutout_dict_large_rgb['F435W_img_cutout'].wcs,
                                               new_wcs=new_wcs,
                                               new_shape=new_shape)


nircam_img_r = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F360M_img_cutout'].data,
                                              wcs=cutout_dict_large_rgb['F360M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_g = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F335M_img_cutout'].data,
                                              wcs=cutout_dict_large_rgb['F335M_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)
nircam_img_b = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F200W_img_cutout'].data,
                                              wcs=cutout_dict_large_rgb['F200W_img_cutout'].wcs,
                                              new_wcs=new_wcs,
                                              new_shape=new_shape)

miri_img_r = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F2100W_img_cutout'].data,
                                            wcs=cutout_dict_large_rgb['F2100W_img_cutout'].wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)
miri_img_g = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F1000W_img_cutout'].data,
                                            wcs=cutout_dict_large_rgb['F1000W_img_cutout'].wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)
miri_img_b = plotting_tools.reproject_image(data=cutout_dict_large_rgb['F770W_img_cutout'].data,
                                            wcs=cutout_dict_large_rgb['F770W_img_cutout'].wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)

muse_img_r = plotting_tools.reproject_image(data=cutout_muse_r.data, wcs=cutout_muse_r.wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)
muse_img_g = plotting_tools.reproject_image(data=cutout_muse_g.data, wcs=cutout_muse_g.wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)
muse_img_b = plotting_tools.reproject_image(data=cutout_muse_b.data, wcs=cutout_muse_b.wcs,
                                            new_wcs=new_wcs,
                                            new_shape=new_shape)

alma_img = plotting_tools.reproject_image(data=cutout_alma.data, wcs=cutout_alma.wcs,
                                          new_wcs=new_wcs,
                                          new_shape=new_shape)

q = 10
stretch = 1

rgb_alma_image = make_lupton_rgb(alma_img,
                                 np.zeros(alma_img.shape),
                                 np.zeros(alma_img.shape),
                                 Q=q, stretch=stretch, minimum=-0.0)

rgb_miri_image = make_lupton_rgb(miri_img_r,
                                   miri_img_g,
                                   miri_img_b,
                                   Q=q, stretch=stretch, minimum=-0.0)
rgb_nircam_image = make_lupton_rgb(nircam_img_r*2,
                                   nircam_img_g,
                                   nircam_img_b,
                                   Q=q, stretch=stretch, minimum=-0.0)
rgb_hst_image = make_lupton_rgb(hst_img_r * 0.5,
                                   hst_img_g,
                                   hst_img_b,
                                   Q=q, stretch=stretch, minimum=-0.0)

rgb_hst_h_alpha_image = make_lupton_rgb(hst_h_alpha_r,
                                   hst_h_alpha_g,
                                   hst_h_alpha_b,
                                   Q=q, stretch=stretch, minimum=-0.0)

rgb_muse_image = make_lupton_rgb(muse_img_r * 0.5,
                                   muse_img_g * 1,
                                   muse_img_b * 1.5,
                                   Q=q, stretch=stretch, minimum=-0.0)




# get masks

x_line = np.linspace(1, new_shape[0],
                     new_shape[0])
y_line = np.linspace(1, new_shape[1],
                     new_shape[1])
x_data, y_data = np.meshgrid(x_line, y_line)

x_half_length = new_shape[0] / 2
y_half_length = new_shape[1] / 2

# central_radius
central_rad_arcsec = 10
central_rad_pix = helper_func.transform_world2pix_scale(length_in_arcsec=central_rad_arcsec,
                                                        wcs=new_wcs)
central_pixel = new_wcs.world_to_pixel(central_coordinates)
mask_central = np.sqrt((x_data - central_pixel[0]) ** 2 + (y_data - central_pixel[1]) ** 2) < central_rad_pix

# pizza masks
l_1_inter = x_half_length * (1 - np.sqrt(3))
l_1_slope = (central_pixel[1] - l_1_inter) / central_pixel[0]

l_2_inter = x_half_length * (1 + np.sqrt(3))
l_2_slope = (central_pixel[1] - l_2_inter) / central_pixel[0]

mask_p1 = (y_data <= y_half_length) & (y_data > (l_1_slope * x_data + l_1_inter)) & ~mask_central
mask_p2 = ((y_data < (l_1_slope * x_data + l_1_inter)) & (y_data < (l_2_slope * x_data + l_2_inter))) & ~mask_central
mask_p3 = ((y_data <= y_half_length) & (y_data > (l_2_slope * x_data + l_2_inter))) & ~mask_central
mask_p4 = ((y_data > y_half_length) & (y_data < (l_1_slope * x_data + l_1_inter))) & ~mask_central
mask_p5 = ((y_data > (l_1_slope * x_data + l_1_inter)) & (y_data > (l_2_slope * x_data + l_2_inter))) & ~mask_central
mask_p6 = ((y_data < (l_2_slope * x_data + l_2_inter)) & (y_data > y_half_length)) & ~mask_central


test_mask_1 = (y_data < (l_1_slope * x_data + l_1_inter))
test_mask_2 = (y_data < (l_2_slope * x_data + l_2_inter))

dummy_y_1 = l_1_slope * x_line + l_1_inter
dummy_y_2 = l_2_slope * x_line + l_2_inter


display_img = np.zeros(rgb_alma_image.shape, dtype=int)

display_img[mask_p5 + mask_central] = rgb_miri_image[mask_p5 + mask_central]
display_img[mask_p4] = rgb_nircam_image[mask_p4]
display_img[mask_p6] = rgb_alma_image[mask_p6]
display_img[mask_p1] = rgb_hst_h_alpha_image[mask_p1]
display_img[mask_p2] = rgb_hst_image[mask_p2]
display_img[mask_p3] = rgb_muse_image[mask_p3]


figure = plt.figure(figsize=(30, 30))
fontsize = 40
ax_img = figure.add_axes([0.06, 0.04, 0.93, 0.93], projection=cutout_dict_large_rgb['F200W_img_cutout'].wcs)


ax_img.imshow(display_img)


# plot lines
# horizontal lines
plt.plot([0, x_half_length-central_rad_pix], [y_half_length, y_half_length], color='white', linewidth=4)
plt.plot([x_half_length+central_rad_pix, 2*x_half_length], [y_half_length, y_half_length], color='white', linewidth=4)

# plot circles
circ_angle_1 = np.linspace(0, 60, 50)
circ_angle_2 = np.linspace(120, 360, 200)
x_dots_1 = central_rad_pix * np.cos(circ_angle_1 * np.pi / 180) + central_pixel[0]
y_dots_1 = central_rad_pix * np.sin(circ_angle_1 * np.pi / 180) + central_pixel[1]
x_dots_2 = central_rad_pix * np.cos(circ_angle_2 * np.pi / 180) + central_pixel[0]
y_dots_2 = central_rad_pix * np.sin(circ_angle_2 * np.pi / 180) + central_pixel[1]
plt.plot(x_dots_1, y_dots_1, color='white', linewidth=4)
plt.plot(x_dots_2, y_dots_2, color='white', linewidth=4)

plt.plot([central_rad_pix * np.cos(60 * np.pi / 180) + central_pixel[0],
          display_img.shape[0] * np.cos(60 * np.pi / 180) + central_pixel[0]],
         [central_rad_pix * np.sin(60 * np.pi / 180) + central_pixel[0],
          display_img.shape[1] * np.sin(60 * np.pi / 180) + central_pixel[0]],
         color='white', linewidth=4)
plt.plot([central_rad_pix * np.cos(120 * np.pi / 180) + central_pixel[0],
          display_img.shape[0] * np.cos(120 * np.pi / 180) + central_pixel[0]],
         [central_rad_pix * np.sin(120 * np.pi / 180) + central_pixel[0],
          display_img.shape[1] * np.sin(120 * np.pi / 180) + central_pixel[0]],
         color='white', linewidth=4)
plt.plot([central_rad_pix * np.cos(240 * np.pi / 180) + central_pixel[0],
          display_img.shape[0] * np.cos(240 * np.pi / 180) + central_pixel[0]],
         [central_rad_pix * np.sin(240 * np.pi / 180) + central_pixel[0],
          display_img.shape[1] * np.sin(240 * np.pi / 180) + central_pixel[0]],
         color='white', linewidth=4)
plt.plot([central_rad_pix * np.cos(300 * np.pi / 180) + central_pixel[0],
          display_img.shape[0] * np.cos(300 * np.pi / 180) + central_pixel[0]],
         [central_rad_pix * np.sin(300 * np.pi / 180) + central_pixel[0],
          display_img.shape[1] * np.sin(300 * np.pi / 180) + central_pixel[0]],
         color='white', linewidth=4)


ax_img.set_xlim(0, display_img.shape[0] - 1)
ax_img.set_ylim(0, display_img.shape[1] - 1)


ax_img.text(display_img.shape[0] * 0.02, display_img.shape[1] * 0.97, 'ALMA CO(2-1)',
            horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize, color='white')
ax_img.text(display_img.shape[0] * 0.5, display_img.shape[1] * 0.97, 'JWST MIRI',
            horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize, color='white')
ax_img.text(display_img.shape[0] * 0.98, display_img.shape[1] * 0.97, 'JWST NIRCAM',
            horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize, color='white')

ax_img.text(display_img.shape[0] * 0.02, display_img.shape[1] * 0.01, r'HST + H$\alpha$',
            horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize, color='white')
ax_img.text(display_img.shape[0] * 0.5, display_img.shape[1] * 0.01, 'HST',
            horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize, color='white')
ax_img.text(display_img.shape[0] * 0.98, display_img.shape[1] * 0.01, 'MUSE',
            horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize, color='k')

ax_img.set_title('NGC 628', fontsize=fontsize)

plotting_tools.arr_axis_params(ax=ax_img, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)


plt.savefig('plot_output/overview_img.png')
plt.savefig('plot_output/overview_img.pdf')

exit()




















exit()

# create RGB image
rgb_hst_image = make_lupton_rgb(cutout_dict_large_rgb['F658N_img_cutout'].data,
                                cutout_dict_large_rgb['F555W_img_cutout'].data,
                                cutout_dict_large_rgb['F435W_img_cutout'].data,
                                Q=14, stretch=3)


plt.imshow(rgb_hst_image)

plt.show()

exit()






plt.imshow(rgb_muse_image)
plt.show()

