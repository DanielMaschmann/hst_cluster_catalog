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


target = 'ngc1097'
galaxy_name = 'ngc1097'
hst_band_list = ['F275W', 'F336W', 'F438W', 'F555W', 'F657N', 'F814W']
hst_band_list_cigale = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']

# crate wrapper class object
cigale_wrapper = CigaleModelWrapper()

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

flux_f275w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=hst_band_list[0])
flux_f336w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=hst_band_list[1])
flux_f438w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=hst_band_list[2])
flux_f555w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=hst_band_list[3])
flux_f814w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=hst_band_list[5])

flux_f275w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=hst_band_list[0])
flux_f336w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=hst_band_list[1])
flux_f438w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=hst_band_list[2])
flux_f555w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=hst_band_list[3])
flux_f814w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=hst_band_list[5])


def fit_cigale(index, met, ebv=None):
    # get flux dict
    flux_dict = {'n_obj': 1, 'flux_F275W': [flux_f275w[index]], 'flux_F336W': [flux_f336w[index]],
                 'flux_F438W': [flux_f438w[index]], 'flux_F555W': [flux_f555w[index]], 'flux_F814W': [flux_f814w[index]],
                 'flux_F275W_err': [flux_f275w_err[index]], 'flux_F336W_err': [flux_f336w_err[index]],
                 'flux_F438W_err': [flux_f438w_err[index]], 'flux_F555W_err': [flux_f555w_err[index]],
                 'flux_F814W_err': [flux_f814w_err[index]]}
    # fit SED
    cigale_wrapper.create_cigale_flux_file(file_path='data_file.dat',
                                           band_list=hst_band_list_cigale,
                                           target_name=target, flux_dict=flux_dict)
    cigale_wrapper.configurate_hst_cc_fit(data_file='data_file.dat',
                                          band_list=hst_band_list_cigale,
                                          target_name=target, ncores=4, met=met, ebv=ebv)
    cigale_wrapper.run_cigale(save_pdf=False)

    hdu = fits.open('out/results.fits')
    best_age = hdu[1].data['best.sfh.age']
    best_m_star = hdu[1].data['best.stellar.m_star']
    best_ebv = hdu[1].data['best.attenuation.E_BV']

    # print('best_age ', best_age)
    # print('best_m_star ', best_m_star)
    # print('best_ebv ', best_ebv)

    # load fit
    hdu_best_model = fits.open('out/0_best_model.fits')
    data = hdu_best_model[1].data
    header = hdu_best_model[1].header
    wavelength_spec = data["wavelength"] * 1e-3
    surf = 4.0 * np.pi * float(header["universe.luminosity_distance"]) ** 2
    fact = 1e29 * 1e-3 * wavelength_spec ** 2 / c / surf

    spectrum = data['Fnu']
    wavelength = data['wavelength']
    stellar_spectrum = (data["stellar.young"] + data["stellar.old"] +
                        data["attenuation.stellar.young"] + data["attenuation.stellar.old"]) * fact
    stellar_spectrum_unattenuated = (data["stellar.young"] + data["stellar.old"]) * fact

    result_dict = {
        'best_age': best_age,
        'best_m_star': best_m_star,
        'best_ebv': best_ebv,
        'spectrum': spectrum,
        'wavelength': wavelength,
        'stellar_spectrum': stellar_spectrum,
        'stellar_spectrum_unattenuated': stellar_spectrum_unattenuated
    }

    return result_dict


def plot_data_points(ax, index, color, size=40, label=None):

    if flux_f275w_err[index] < 0:
        ax.errorbar(phangs_photometry.get_band_wave(band='F275W'), flux_f275w[index],  yerr=flux_f275w[index],
                    ecolor='grey', elinewidth=5, capsize=10, uplims=True, xlolims=False)
    else:
        ax.errorbar(phangs_photometry.get_band_wave(band='F275W'), flux_f275w[index], yerr=flux_f275w_err[index],
                         fmt='o', c=color, ms=size)

    if flux_f336w_err[index] < 0:
        ax.errorbar(phangs_photometry.get_band_wave(band='F336W'), flux_f336w[index],  yerr=flux_f336w[index],
                    ecolor='grey', elinewidth=5, capsize=10, uplims=True, xlolims=False)
    else:
        ax.errorbar(phangs_photometry.get_band_wave(band='F336W'), flux_f336w[index], yerr=flux_f336w_err[index],
                         fmt='o', c=color, ms=size)

    if flux_f438w_err[index] < 0:
        ax.errorbar(phangs_photometry.get_band_wave(band='F438W'), flux_f438w[index],  yerr=flux_f438w[index],
                    ecolor='grey', elinewidth=5, capsize=10, uplims=True, xlolims=False)
    else:
        ax.errorbar(phangs_photometry.get_band_wave(band='F438W'), flux_f438w[index], yerr=flux_f438w_err[index],
                         fmt='o', c=color, ms=size)

    if flux_f555w_err[index] < 0:
        ax.errorbar(phangs_photometry.get_band_wave(band='F555W'), flux_f555w[index],  yerr=flux_f555w[index],
                    ecolor='grey', elinewidth=5, capsize=10, uplims=True, xlolims=False)
    else:
        ax.errorbar(phangs_photometry.get_band_wave(band='F555W'), flux_f555w[index], yerr=flux_f555w_err[index],
                         fmt='o', c=color, ms=size)

    if flux_f814w_err[index] < 0:
        ax.errorbar(phangs_photometry.get_band_wave(band='F814W'), flux_f814w[index],  yerr=flux_f814w[index],
                    ecolor='grey', elinewidth=5, capsize=10, uplims=True, xlolims=False)
    else:
        ax.errorbar(phangs_photometry.get_band_wave(band='F814W'), flux_f814w[index], yerr=flux_f814w_err[index],
                         fmt='o', c=color, ms=size, label=label)


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
central_coordinates = SkyCoord('2h46m20.0s -30d16m32s', unit=(u.hourangle, u.deg))

# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
                                                               dec_cutout=central_coordinates.dec.to(u.deg),
                                                               cutout_size=(60, 60), include_err=False,
                                                               band_list=hst_band_list)
# create RGB image
rgb_image = make_lupton_rgb(cutout_dict_large_rgb['F657N_img_cutout'].data, cutout_dict_large_rgb['F555W_img_cutout'].data,
                            cutout_dict_large_rgb['F438W_img_cutout'].data, Q=14, stretch=3)


# now identify the objects in this region
all_coordinates = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
separation = central_coordinates.separation(all_coordinates)

close_mask = separation < 10*u.arcsec
# get young objects
selection_young = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age < 10)
                           # & (ebv > 0.3)
                           & (mstar > 1e1) & (chi2 < 1.2) & close_mask
                           )
selection_inter = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age >= 10) & (age < 100)
                           & (ebv > 0.3) & (mstar > 1e1) & (chi2 < 1.2) & close_mask
                           )
selection_old = np.where((clcl == 1) & (clcl_qual >= 0.8) & (age > 1000) & (ebv < 0.1) & (mstar > 1e3) & (chi2 < 1.2) & close_mask)

#indexes = list(selection_young[0]) + list(selection_inter[0]) + list(selection_old[0])
# index_old = 940 # dusty old globular cluster
# index_old = 1329 # also dusty old cluster
# index_old = 997
index_old = 1099
# index_inter = selection_inter[0][0]
index_inter = 1116 # very young dsty cluster
index_yellow = 807
# index_young = selection_young[0][0]
index_young = 1094
print('index_old ', index_old)
print('index_inter ', index_inter)
print('index_young ', index_young)

result_dict_old = fit_cigale(index=index_old, met=0.0004, ebv=[0.0])
result_dict_inter = fit_cigale(index=index_inter, met=0.02)
result_dict_yellow = fit_cigale(index=index_yellow, met=0.02)
result_dict_young = fit_cigale(index=index_young, met=0.02)
print(result_dict_old)
print(result_dict_inter)
print(result_dict_yellow)
print(result_dict_young)


pos_old = SkyCoord(ra=ra[index_old] * u.deg, dec=dec[index_old] * u.deg)
pos_inter = SkyCoord(ra=ra[index_inter] * u.deg, dec=dec[index_inter] * u.deg)
pos_yellow = SkyCoord(ra=ra[index_yellow] * u.deg, dec=dec[index_yellow] * u.deg)
pos_young = SkyCoord(ra=ra[index_young] * u.deg, dec=dec[index_young] * u.deg)

cutout_dict_old = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_old.ra.to(u.deg),
                                                         dec_cutout=pos_old.dec.to(u.deg),
                                                         cutout_size=(4, 4), include_err=False, band_list=hst_band_list)
rgb_old = make_lupton_rgb(cutout_dict_old['F657N_img_cutout'].data, cutout_dict_old['F555W_img_cutout'].data,
                          cutout_dict_old['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_inter = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_inter.ra.to(u.deg),
                                                           dec_cutout=pos_inter.dec.to(u.deg),
                                                           cutout_size=(4, 4), include_err=False,
                                                           band_list=hst_band_list)
rgb_inter = make_lupton_rgb(cutout_dict_inter['F657N_img_cutout'].data, cutout_dict_inter['F555W_img_cutout'].data,
                            cutout_dict_inter['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_yellow = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_yellow.ra.to(u.deg),
                                                            dec_cutout=pos_yellow.dec.to(u.deg),
                                                            cutout_size=(4, 4), include_err=False,
                                                            band_list=hst_band_list)
rgb_yellow = make_lupton_rgb(cutout_dict_yellow['F657N_img_cutout'].data, cutout_dict_yellow['F555W_img_cutout'].data,
                             cutout_dict_yellow['F438W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_young = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_young.ra.to(u.deg),
                                                           dec_cutout=pos_young.dec.to(u.deg),
                                                           cutout_size=(4, 4), include_err=False,
                                                           band_list=hst_band_list)
rgb_young = make_lupton_rgb(cutout_dict_young['F657N_img_cutout'].data, cutout_dict_young['F555W_img_cutout'].data,
                            cutout_dict_young['F438W_img_cutout'].data, Q=14, stretch=3)


figure = plt.figure(figsize=(35, 11))
fontsize = 26
ax_rgb = figure.add_axes([0.02, 0.08, 0.3, 0.905], projection=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
ax_rgb_old = figure.add_axes([-0.03, 0.4, 0.2, 0.2], projection=cutout_dict_old['F555W_img_cutout'].wcs)
ax_rgb_yellow = figure.add_axes([0.19, 0.1, 0.2, 0.2], projection=cutout_dict_inter['F555W_img_cutout'].wcs)
ax_rgb_inter = figure.add_axes([0.05, 0.71, 0.2, 0.2], projection=cutout_dict_inter['F555W_img_cutout'].wcs)
ax_rgb_young = figure.add_axes([0.20, 0.74, 0.2, 0.2], projection=cutout_dict_young['F555W_img_cutout'].wcs)

ax_sed = figure.add_axes([0.36, 0.075, 0.37, 0.905])
ax_cc_vi_ub = figure.add_axes([0.78, 0.54, 0.21, 0.45])
ax_cc_vi_nuvb = figure.add_axes([0.78, 0.075, 0.21, 0.45])

circ_rad_arcsec = 1

ax_rgb.imshow(rgb_image)

# add names for colors
ax_rgb.text(100, 150, r'R: F657N (H$\alpha$)', color='lightcoral', fontsize=fontsize)
ax_rgb.text(100, 100, 'G: F555W', color='springgreen', fontsize=fontsize)
ax_rgb.text(100, 50, 'B: F438W', color='royalblue', fontsize=fontsize)
ax_rgb.text(60, rgb_image.shape[0]-100, 'NGC 1097', color='white', fontsize=fontsize+5)

circle_old = SphericalCircle(pos_old, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                             alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_old)
circle_yellow = SphericalCircle(pos_yellow, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                                alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_yellow)
circle_inter = SphericalCircle(pos_inter, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_inter)
circle_young = SphericalCircle(pos_young, circ_rad_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_young)

ax_rgb_old.imshow(rgb_old)
ax_rgb_yellow.imshow(rgb_yellow)
ax_rgb_inter.imshow(rgb_inter)
ax_rgb_young.imshow(rgb_young)

ax_rgb_old.set_title('C1', fontsize=fontsize+5, color='r')
ax_rgb_yellow.set_title('C3', fontsize=fontsize+5, color='y')
ax_rgb_inter.set_title('C2', fontsize=fontsize+5, color='green')
ax_rgb_young.set_title('C4', fontsize=fontsize+5, color='blue')

plotting_tools.arr_axis_params(ax=ax_rgb, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_old, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_yellow, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_inter, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_rgb_young, ra_tick_label=False, dec_tick_label=False,
                    ra_axis_label=' ', dec_axis_label=' ',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='white',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)

circ_rad_x = hf.transform_world2pix_scale(length_in_arcsec=circ_rad_arcsec,
                                          wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
circ_rad_y = hf.transform_world2pix_scale(length_in_arcsec=circ_rad_arcsec,
                                          wcs=cutout_dict_large_rgb['F555W_img_cutout'].wcs, dim=1)
print('circ_rad_x ', circ_rad_x)
print('circ_rad_y ', circ_rad_y)

pos_pix_circ_old = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_old)
con_spec_old_1 = ConnectionPatch(
    xyA=(pos_pix_circ_old[0] - circ_rad_x, pos_pix_circ_old[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_old.get_xlim()[0], ax_rgb_old.get_ylim()[1]), coordsB=ax_rgb_old.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_old_1)
con_spec_old_2 = ConnectionPatch(
    xyA=(pos_pix_circ_old[0] + circ_rad_x, pos_pix_circ_old[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_old.get_xlim()[1], ax_rgb_old.get_ylim()[1]), coordsB=ax_rgb_old.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_old_2)


pos_pix_circ_yellow = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_yellow)
con_spec_yellow_1 = ConnectionPatch(
    xyA=(pos_pix_circ_yellow[0], pos_pix_circ_yellow[1] - circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_yellow.get_xlim()[0], ax_rgb_yellow.get_ylim()[0]), coordsB=ax_rgb_yellow.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_yellow_1)
con_spec_yellow_2 = ConnectionPatch(
    xyA=(pos_pix_circ_yellow[0], pos_pix_circ_yellow[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_yellow.get_xlim()[0], ax_rgb_yellow.get_ylim()[1]), coordsB=ax_rgb_yellow.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_yellow_2)


pos_pix_circ_inter = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_inter)
con_spec_inter_1 = ConnectionPatch(
    xyA=(pos_pix_circ_inter[0], pos_pix_circ_inter[1] - circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_inter.get_xlim()[1], ax_rgb_inter.get_ylim()[0]), coordsB=ax_rgb_inter.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_inter_1)
con_spec_inter_2 = ConnectionPatch(
    xyA=(pos_pix_circ_inter[0], pos_pix_circ_inter[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_inter.get_xlim()[1], ax_rgb_inter.get_ylim()[1]), coordsB=ax_rgb_inter.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_inter_2)


pos_pix_circ_young = cutout_dict_large_rgb['F555W_img_cutout'].wcs.world_to_pixel(pos_young)
con_spec_young_1 = ConnectionPatch(
    xyA=(pos_pix_circ_young[0] + circ_rad_x, pos_pix_circ_young[1]), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_young.get_xlim()[0], ax_rgb_young.get_ylim()[0]), coordsB=ax_rgb_young.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_young_1)
con_spec_young_2 = ConnectionPatch(
    xyA=(pos_pix_circ_young[0], pos_pix_circ_young[1] + circ_rad_y), coordsA=ax_rgb.transData,
    xyB=(ax_rgb_young.get_xlim()[0], ax_rgb_young.get_ylim()[1]), coordsB=ax_rgb_young.transData,
    linestyle="-", linewidth=3, color='white')
figure.add_artist(con_spec_young_2)



ax_sed.plot(result_dict_old['wavelength'] * 1e-3, result_dict_old['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_yellow['wavelength'] * 1e-3, result_dict_yellow['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_inter['wavelength'] * 1e-3, result_dict_inter['stellar_spectrum'], linewidth=3, color='k')
ax_sed.plot(result_dict_young['wavelength'] * 1e-3, result_dict_young['stellar_spectrum'], linewidth=3, color='k')

plot_data_points(ax=ax_sed, index=index_old, color='r', size=13,
                 label=r'C1, age=%i Gyr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$/50, E(B-V)=%.2f' %
                       (result_dict_old['best_age'][0]/1000, round(result_dict_old['best_m_star'][0]/1e5),
                        result_dict_old['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_yellow, color='y', size=13,
                 label=r'C3, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_yellow['best_age'][0], round(result_dict_yellow['best_m_star'][0]/1e5),
                        result_dict_yellow['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_inter, color='g', size=13,
                  label=r'C2, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_inter['best_age'][0], round(result_dict_inter['best_m_star'][0]/1e5),
                        result_dict_inter['best_ebv'][0]))
plot_data_points(ax=ax_sed, index=index_young, color='b', size=13,
                 label=r'C4, age=%i Myr, M$_{*}$ = %i 10$^{5}$ M$_{\odot}$, Z=Z$_{\odot}$, E(B-V)=%.2f' %
                       (result_dict_young['best_age'][0], round(result_dict_young['best_m_star'][0]/1e5),
                        result_dict_young['best_ebv'][0]))




ax_sed.legend(frameon=False, fontsize=fontsize)
ax_sed.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
ax_sed.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
ax_sed.set_xlim(200 * 1e-3, 1.5)
ax_sed.set_ylim(0.000009, 3e2)
ax_sed.set_xscale('log')
ax_sed.set_yscale('log')
ax_sed.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)


ax_cc_vi_ub.plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='-', alpha=1.0, label='BC03, Z=Z$_{\odot}$')
ax_cc_vi_ub.plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='-', alpha=1.0, label='BC03, Z=Z$_{\odot}$/50')
ax_cc_vi_ub.scatter(color_vi_ml_12[index_old], color_ub_ml_12[index_old], color='r', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_inter], color_ub_ml_12[index_inter], color='g', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_yellow], color_ub_ml_12[index_yellow], color='y', s=150, zorder=10)
ax_cc_vi_ub.scatter(color_vi_ml_12[index_young], color_ub_ml_12[index_young], color='b', s=150, zorder=10)

ax_cc_vi_nuvb.plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=3, linestyle='-', alpha=1.0)
ax_cc_vi_nuvb.plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=3, linestyle='-', alpha=1.0)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_old], color_nuvb_ml_12[index_old], color='r', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_inter], color_nuvb_ml_12[index_inter], color='g', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_yellow], color_nuvb_ml_12[index_yellow], color='y', s=150, zorder=10)
ax_cc_vi_nuvb.scatter(color_vi_ml_12[index_young], color_nuvb_ml_12[index_young], color='b', s=150, zorder=10)

ax_cc_vi_ub.set_ylim(0.9, -1.9)
ax_cc_vi_ub.set_xlim(-0.4, 1.6)
ax_cc_vi_nuvb.set_ylim(2.6, -3.1)
ax_cc_vi_nuvb.set_xlim(-0.4, 1.6)

#ax_cc_vi_ub.set_xticks(ax_cc_vi_nuvb.get_xticks())
ax_cc_vi_ub.set_xticklabels([])
ax_cc_vi_nuvb.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax_cc_vi_ub.set_ylabel('U (F336W) - B (F438W)', fontsize=fontsize)
ax_cc_vi_nuvb.set_ylabel('NUV (F275W) - B (F438W)', labelpad=30, fontsize=fontsize)

ax_cc_vi_ub.legend(frameon=False, fontsize=fontsize, loc=3)

ax_cc_vi_ub.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_cc_vi_nuvb.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)



plt.savefig('plot_output/sed_explain.png')
plt.savefig('plot_output/sed_explain.pdf')

exit()