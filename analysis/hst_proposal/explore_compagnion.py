import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
from astropy.io import fits
from photometry_tools import helper_func as hf
from photometry_tools.data_access import CatalogAccess
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.analysis_tools import CigaleModelWrapper
from photometry_tools import plotting_tools
from scipy.constants import c
from astroquery.simbad import Simbad
from astropy.visualization import make_lupton_rgb
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import ConnectionPatch

target = 'ngc1512'
galaxy_name = 'ngc1512'
hst_band_list = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']

# prepare the flux file
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)


# get model
hdu_a_sol = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod_sol = hdu_a_sol[1].data
age_mod_sol = data_mod_sol['sfh.age']
flux_f275w_sol = data_mod_sol['F275W_UVIS_CHIP2']
flux_f555w_sol = data_mod_sol['F555W_UVIS_CHIP2']
flux_f814w_sol = data_mod_sol['F814W_UVIS_CHIP2']
flux_f336w_sol = data_mod_sol['F336W_UVIS_CHIP2']
flux_f438w_sol = data_mod_sol['F438W_UVIS_CHIP2']
mag_v_sol = hf.conv_mjy2vega(flux=flux_f555w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol = hf.conv_mjy2vega(flux=flux_f814w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol = hf.conv_mjy2vega(flux=flux_f336w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol = hf.conv_mjy2vega(flux=flux_f438w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_nuv_sol = hf.conv_mjy2vega(flux=flux_f275w_sol, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_sol = mag_v_sol - mag_i_sol
model_ub_sol = mag_u_sol - mag_b_sol
model_nuvb_sol = mag_nuv_sol - mag_b_sol

# get model
hdu_a_sol50 = fits.open('../cigale_model/sfh2exp/no_dust/sol_met_50/out/models-block-0.fits')
data_mod_sol50 = hdu_a_sol50[1].data
age_mod_sol50 = data_mod_sol50['sfh.age']
flux_f275w_sol50 = data_mod_sol50['F275W_UVIS_CHIP2']
flux_f555w_sol50 = data_mod_sol50['F555W_UVIS_CHIP2']
flux_f814w_sol50 = data_mod_sol50['F814W_UVIS_CHIP2']
flux_f336w_sol50 = data_mod_sol50['F336W_UVIS_CHIP2']
flux_f438w_sol50 = data_mod_sol50['F438W_UVIS_CHIP2']
mag_v_sol50 = hf.conv_mjy2vega(flux=flux_f555w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sol50 = hf.conv_mjy2vega(flux=flux_f814w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
mag_u_sol50 = hf.conv_mjy2vega(flux=flux_f336w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sol50 = hf.conv_mjy2vega(flux=flux_f438w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_nuv_sol50 = hf.conv_mjy2vega(flux=flux_f275w_sol50, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                               vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))

model_vi_sol50 = mag_v_sol50 - mag_i_sol50
model_ub_sol50 = mag_u_sol50 - mag_b_sol50
model_nuvb_sol50 = mag_nuv_sol50 - mag_b_sol50

catalog_access.load_hst_cc_list(target_list=[target], classify='ml')
clcl = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
clcl_qual = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
phangs_id = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml')
age = catalog_access.get_hst_cc_age(target=target, classify='ml')
mstar = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
ebv = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
chi2 = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml')
ra, dec = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb(target=target, classify='ml')
color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')

# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr', band_list=hst_band_list, load_err=False)


# get the center of the region of interest
# simbad_table = Simbad.query_object(galaxy_name)
# central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
#                                unit=(u.hourangle, u.deg))
# img_cent_coords = central_coordinates
img_cent_coords = SkyCoord('4h03m42.0s -43d23m00s', unit=(u.hourangle, u.deg))


# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=img_cent_coords.ra.to(u.deg),
                                                               dec_cutout=img_cent_coords.dec.to(u.deg),
                                                               cutout_size=(500, 700), include_err=False,
                                                               # cutout_size=(20, 20), include_err=False,
                                                               band_list=hst_band_list)

# create RGB image
# r*5,g*0.75,b*8,Q=0.001,stretch=300,
rgb_image = make_lupton_rgb(
    cutout_dict_large_rgb['F814W_img_cutout'].data * 0.8,
    (cutout_dict_large_rgb['F438W_img_cutout'].data + cutout_dict_large_rgb['F555W_img_cutout'].data) * 0.55,
    (cutout_dict_large_rgb['F275W_img_cutout'].data + cutout_dict_large_rgb['F336W_img_cutout'].data) * 0.6,
    # Q=15, stretch=0.1, minimum=-0.01)
    Q=10, stretch=0.5, minimum=-0.0)
# rgb_image = make_lupton_rgb(
#     cutout_dict_large_rgb['F814W_img_cutout'].data * 0.8,
#     (cutout_dict_large_rgb['F438W_img_cutout'].data + cutout_dict_large_rgb['F555W_img_cutout'].data) * 0.55,
#     (cutout_dict_large_rgb['F275W_img_cutout'].data + cutout_dict_large_rgb['F336W_img_cutout'].data) * 0.6,
#     Q=10, stretch=0.5, minimum=-0.0)


mask_no_data = (rgb_image[:, :, 0] == 25) & (rgb_image[:, :, 1] == 25) & (rgb_image[:, :, 2] == 25)
rgb_image[:, :, 0][mask_no_data] = 255
rgb_image[:, :, 1][mask_no_data] = 255
rgb_image[:, :, 2][mask_no_data] = 255


figure = plt.figure(figsize=(35, 17))
fontsize = 26
ax_rgb = figure.add_axes([0.08, 0.08, 0.908, 0.905], projection=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
ax_rgb.imshow(rgb_image)

# add names for colors
# ax_rgb.text(100, 150, r'R: F814W', color='lightcoral', fontsize=fontsize)
# ax_rgb.text(100, 100, 'G: F555W', color='springgreen', fontsize=fontsize)
# ax_rgb.text(100, 50, 'B: F336W', color='royalblue', fontsize=fontsize)
# ax_rgb.text(60, rgb_image.shape[0]-100, 'NGC 1512', color='white', fontsize=fontsize+5)

for ra, dec in zip(ra, dec):
    pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    circle = SphericalCircle(pos, 0.5 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                             alpha=1.0, transform=ax_rgb.get_transform('fk5'))
    ax_rgb.add_patch(circle)



# plt.show()
plt.savefig('plot_output/overview.png')

