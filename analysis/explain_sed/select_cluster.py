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


# crate wrapper class object
cigale_wrapper = CigaleModelWrapper()

# prepare the flux file
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)

target = 'ngc1097'
galaxy_name = 'ngc1097'
hst_band_list = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']


catalog_access.load_hst_cc_list(target_list=[target], classify='ml')
clcl = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
clcl_qual = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
phangs_id = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml')
age = catalog_access.get_hst_cc_age(target=target, classify='ml')
mstar = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
ebv = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
chi2 = catalog_access.get_hst_cc_min_chi2(target=target, classify='ml')
ra, dec = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')

# flux_f275w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W')
# flux_f336w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W')
# flux_f435w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F435W')
# flux_f555w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W')
# flux_f814w = catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W')
#
# flux_f275w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F275W')
# flux_f336w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F336W')
# flux_f435w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F435W')
# flux_f555w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F555W')
# flux_f814w_err = catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F814W')


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
central_coordinates = SkyCoord('2h46m18.8s -30d16m13s', unit=(u.hourangle, u.deg))

# get a cutout
cutout_dict_large_rgb = phangs_photometry.get_band_cutout_dict(ra_cutout=central_coordinates.ra.to(u.deg),
                                                               dec_cutout=central_coordinates.dec.to(u.deg),
                                                               cutout_size=(100, 100), include_err=False,
                                                               band_list=hst_band_list)
# create RGB image
rgb_image = make_lupton_rgb(cutout_dict_large_rgb['F814W_img_cutout'].data, cutout_dict_large_rgb['F555W_img_cutout'].data,
                            cutout_dict_large_rgb['F336W_img_cutout'].data, Q=14, stretch=3)


# now identify the objects in this region
all_coordinates = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
# test_coordinates = SkyCoord('2h46m20.18s -30d16m30.9s', unit=(u.hourangle, u.deg))
# test_coordinates = SkyCoord('2h46m19.44s -30d15m52.3s', unit=(u.hourangle, u.deg))
# test_coordinates = SkyCoord('2h46m22.39s -30d16m27.4s', unit=(u.hourangle, u.deg))
# test_coordinates = SkyCoord('2h46m21.87s -30d16m19.2s', unit=(u.hourangle, u.deg))
# test_coordinates = SkyCoord('2h46m18.67s -30d16m19.8s', unit=(u.hourangle, u.deg))
test_coordinates = SkyCoord('2h46m19.46s -30d16m40.9s', unit=(u.hourangle, u.deg))

separation = test_coordinates.separation(all_coordinates)

close_mask = separation < 0.5*u.arcsec
print(np.where(close_mask))


# get young objects
selection_young = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age < 10)
                           # & (ebv > 0.3)
                           & (mstar > 1e1) & (chi2 < 1.2)
                           )
selection_inter = np.where(((clcl == 2) | (clcl == 3))
                           & (clcl_qual >= 0.8)
                           & (age >= 10) & (age < 100)
                           & (ebv > 0.3) & (mstar > 1e1) & (chi2 < 1.2)
                           )
selection_old = np.where((clcl == 1) & (clcl_qual >= 0.8) & (age > 1000) & (ebv < 0.1) & (mstar > 1e5) & (chi2 < 1.2))

#indexes = list(selection_young[0]) + list(selection_inter[0]) + list(selection_old[0])
print('selection_old ', selection_old)
print('selection_inter ', selection_inter)
print('selection_young ', selection_young)


index_old = selection_old[0][0]
index_inter = selection_inter[0][0]
index_young = selection_young[0][0]
print('index_old ', index_old)
print('index_inter ', index_inter)
print('index_young ', index_young)


pos_old = SkyCoord(ra=ra[index_old] * u.deg, dec=dec[index_old] * u.deg)
pos_inter = SkyCoord(ra=ra[index_inter] * u.deg, dec=dec[index_inter] * u.deg)
pos_young = SkyCoord(ra=ra[index_young] * u.deg, dec=dec[index_young] * u.deg)

cutout_dict_old = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_old.ra.to(u.deg),
                                                         dec_cutout=pos_old.dec.to(u.deg),
                                                         cutout_size=(4, 4), include_err=False, band_list=hst_band_list)
rgb_old = make_lupton_rgb(cutout_dict_old['F814W_img_cutout'].data, cutout_dict_old['F555W_img_cutout'].data,
                          cutout_dict_old['F336W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_inter = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_inter.ra.to(u.deg),
                                                         dec_cutout=pos_inter.dec.to(u.deg),
                                                         cutout_size=(4, 4), include_err=False, band_list=hst_band_list)
rgb_inter = make_lupton_rgb(cutout_dict_inter['F814W_img_cutout'].data, cutout_dict_inter['F555W_img_cutout'].data,
                            cutout_dict_inter['F336W_img_cutout'].data, Q=14, stretch=3)
cutout_dict_young = phangs_photometry.get_band_cutout_dict(ra_cutout=pos_young.ra.to(u.deg),
                                                         dec_cutout=pos_young.dec.to(u.deg),
                                                         cutout_size=(4, 4), include_err=False, band_list=hst_band_list)
rgb_young = make_lupton_rgb(cutout_dict_young['F814W_img_cutout'].data, cutout_dict_young['F555W_img_cutout'].data,
                            cutout_dict_young['F336W_img_cutout'].data, Q=14, stretch=3)


figure = plt.figure(figsize=(20, 10))
fontsize = 18
ax_rgb = figure.add_axes([0.03, 0.07, 0.45, 0.905], projection=cutout_dict_large_rgb['F555W_img_cutout'].wcs)
ax_rgb_old = figure.add_axes([0.55, 0.7, 0.2, 0.2], projection=cutout_dict_old['F555W_img_cutout'].wcs)
ax_rgb_inter = figure.add_axes([0.65, 0.05, 0.2, 0.2], projection=cutout_dict_inter['F555W_img_cutout'].wcs)
ax_rgb_young = figure.add_axes([0.75, 0.35, 0.2, 0.2], projection=cutout_dict_young['F555W_img_cutout'].wcs)


ax_rgb.imshow(rgb_image)
circle_old = SphericalCircle(pos_old, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                             alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_old)
circle_inter = SphericalCircle(pos_inter, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_inter)
circle_young = SphericalCircle(pos_young, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                             alpha=1.0, transform=ax_rgb.get_transform('fk5'))
ax_rgb.add_patch(circle_young)

ax_rgb_old.imshow(rgb_old)
ax_rgb_inter.imshow(rgb_inter)
ax_rgb_young.imshow(rgb_young)

plt.show()

# plt.savefig('plot_output/select_cluster.png')


exit()



index = 116
print(age[index])
print(mstar[index])
print(ebv[index])
print('phangs_id ', phangs_id[index])
# prepare CIGALE fit
flux_dict = {'n_obj': 1, 'flux_F275W': [flux_f275w[index]], 'flux_F336W': [flux_f336w[index]],
             'flux_F435W': [flux_f435w[index]], 'flux_F555W': [flux_f555w[index]], 'flux_F814W': [flux_f814w[index]],
             'flux_F275W_err': [flux_f275w_err[index]], 'flux_F336W_err': [flux_f336w_err[index]],
             'flux_F435W_err': [flux_f435w_err[index]], 'flux_F555W_err': [flux_f555w_err[index]],
             'flux_F814W_err': [flux_f814w_err[index]]}
cigale_wrapper.create_cigale_flux_file(file_path='data_file.dat',
                                       band_list=['F275W', 'F336W', 'F435W', 'F555W', 'F814W'],
                                       target_name=target, flux_dict=flux_dict)
cigale_wrapper.configurate_hst_cc_fit(data_file='data_file.dat',
                                      band_list=['F275W', 'F336W', 'F435W', 'F555W', 'F814W'],
                                      target_name=target, ncores=4)
cigale_wrapper.run_cigale(save_pdf=False)


hdu = fits.open('out/results.fits')
print(hdu.info())
print(hdu[1].data.names)
best_age = hdu[1].data['best.sfh.age']
best_m_star = hdu[1].data['best.stellar.m_star']
best_ebv = hdu[1].data['best.attenuation.E_BV']


# ebv = np.load('out/0_attenuation.E_BV_chi2-block-0.npy', encoding='latin1', allow_pickle=True).item()
agedata = np.memmap('out/0_sfh.age_chi2-block-0.npy', dtype=np.float64)
agedata = np.memmap('out/0_sfh.age_chi2-block-0.npy', dtype=np.float64, shape=(2, agedata.size // 2))
ebvdata = np.memmap('out/0_attenuation.E_BV_chi2-block-0.npy', dtype=np.float64)
ebvdata = np.memmap('out/0_attenuation.E_BV_chi2-block-0.npy', dtype=np.float64, shape=(2, ebvdata.size // 2))


likelihood_age = np.exp(-agedata[0, :] / 2.0)
likelihood_ebv = np.exp(-ebvdata[0, :] / 2.0)

model_variable_age = agedata[1, :]
model_variable_ebv = ebvdata[1, :]

w_age = np.where(np.isfinite(likelihood_age) & np.isfinite(model_variable_age))
w_ebv = np.where(np.isfinite(likelihood_ebv) & np.isfinite(model_variable_ebv))

likelihood_age = likelihood_age[w_age]
model_variable_age = model_variable_age[w_age]

likelihood_ebv = likelihood_ebv[w_ebv]
model_variable_ebv = model_variable_ebv[w_ebv]

Npdf = 100
min_hist_age = np.min(model_variable_age)
max_hist_age = np.max(model_variable_age)
Nhist_age = min(Npdf, len(np.unique(model_variable_age)))
min_hist_ebv = np.min(model_variable_ebv)
max_hist_ebv = np.max(model_variable_ebv)
Nhist_ebv = min(Npdf, len(np.unique(model_variable_ebv)))

print('best_age ', best_age)
print('best_m_star ', best_m_star)
print('best_ebv ', best_ebv)

age_bins = cigale_wrapper.sfh2exp_params['age']
ebv_bins = cigale_wrapper.dustext_params['E_BV']


pdf_prob_age, pdf_grid_age = np.histogram(model_variable_age, bins=age_bins,
                                          weights=likelihood_age, density=True)
pdf_prob_ebv, pdf_grid_ebv = np.histogram(model_variable_ebv, bins=ebv_bins,
                                          weights=likelihood_ebv, density=True)

age_log_bins = np.logspace(0, 5)
ebv_bins = np.linspace(min(ebv_bins), max(ebv_bins), 50)

# pdf_x_ebv = (pdf_grid_ebv[1:] + pdf_grid_ebv[:-1]) / 2.0
# plt.plot(pdf_x_ebv, pdf_prob_ebv)
# plt.show()
#
# pdf_x_age = (pdf_grid_age[1:] + pdf_grid_age[:-1]) / 2.0
# plt.plot(pdf_x_age, pdf_prob_age)
# plt.show()


hist_age, _ = np.histogram(model_variable_age, bins=age_log_bins, weights=likelihood_age)
hist_ebv, _ = np.histogram(model_variable_ebv, bins=ebv_bins, weights=likelihood_ebv)

age_log_bins_center = (age_log_bins[1:] + age_log_bins[:-1]) / 2
age_log_bin_width = (age_log_bins[1:] - age_log_bins[:-1])
ebv_bins_center = (ebv_bins[1:] + ebv_bins[:-1]) / 2
ebv_bin_width = (ebv_bins[1:] - ebv_bins[:-1])

# plt.plot(age_log_bins_center, hist_age)
# plt.xscale('log')
# plt.show()
#
# plt.plot(age_log_bins_center, hist_age/ age_log_bin_width)
# plt.xscale('log')
# plt.show()
# plt.plot(ebv_bins_center, hist_ebv)
# plt.show()



hist_age_ebv, x, y = np.histogram2d(model_variable_age, model_variable_ebv, bins=(age_log_bins, ebv_bins),
                                    weights=likelihood_age)



# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc0628',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy', band_list=hst_band_list)


# size of image
size_of_cutout = (2, 2)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra[index], dec_cutout=dec[index],
                                                     cutout_size=size_of_cutout, include_err=True,
                                                     band_list=hst_band_list)
source = SkyCoord(ra=ra[index], dec=dec[index], unit=(u.degree, u.degree), frame='fk5')

# compute flux from 50% encircled energy
aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                      recenter=True, recenter_rad=0.001,
                                                                      default_ee_rad=50)
phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr', band_list=hst_band_list)


# plotting
figure = plt.figure(figsize=(30, 10))
fontsize = 27

ax_sed = figure.add_axes([0.06, 0.09, 0.6, 0.905])
ax_age_ebv = figure.add_axes([0.715, 0.09, 0.2, 0.7])
ax_age_hist = figure.add_axes([0.715, 0.805, 0.2, 0.17])
ax_ebv_hist = figure.add_axes([0.92, 0.09, 0.07, 0.7])


ax_f275w = figure.add_axes([0.02 + 0 * 0.065, 0.71, 0.19, 0.19], projection=cutout_dict['F275W_img_cutout'].wcs)
ax_f336w = figure.add_axes([0.02 + 1 * 0.065, 0.71, 0.19, 0.19], projection=cutout_dict['F336W_img_cutout'].wcs)
ax_f435w = figure.add_axes([0.02 + 2 * 0.065, 0.71, 0.19, 0.19], projection=cutout_dict['F435W_img_cutout'].wcs)
ax_f555w = figure.add_axes([0.02 + 3 * 0.065, 0.71, 0.19, 0.19], projection=cutout_dict['F555W_img_cutout'].wcs)
ax_f814w = figure.add_axes([0.02 + 4 * 0.065, 0.71, 0.19, 0.19], projection=cutout_dict['F814W_img_cutout'].wcs)

plotting_tools.plot_postage_stamps(ax=ax_f275w, cutout=cutout_dict['F275W_img_cutout'], fontsize=fontsize,
                                   show_ax_label=True, title='F275W')
plotting_tools.plot_postage_stamps(ax=ax_f336w, cutout=cutout_dict['F336W_img_cutout'], fontsize=fontsize,
                                   show_ax_label=False, title='F336W')
plotting_tools.plot_postage_stamps(ax=ax_f435w, cutout=cutout_dict['F435W_img_cutout'], fontsize=fontsize,
                                   show_ax_label=False, title='F435W')
plotting_tools.plot_postage_stamps(ax=ax_f555w, cutout=cutout_dict['F555W_img_cutout'], fontsize=fontsize,
                                   show_ax_label=False, title='F555W')
plotting_tools.plot_postage_stamps(ax=ax_f814w, cutout=cutout_dict['F814W_img_cutout'], fontsize=fontsize,
                                   show_ax_label=False, title='F814W')


ax_sed.errorbar(phangs_photometry.get_band_wave(band='F275W'), flux_f275w[index], yerr=flux_f275w_err[index],
                fmt='o', c='k', ms=10)
ax_sed.errorbar(phangs_photometry.get_band_wave(band='F336W'), flux_f336w[index], yerr=flux_f336w_err[index],
                fmt='o', c='k', ms=10)
ax_sed.errorbar(phangs_photometry.get_band_wave(band='F435W'), flux_f435w[index], yerr=flux_f435w_err[index],
                fmt='o', c='k', ms=10)
ax_sed.errorbar(phangs_photometry.get_band_wave(band='F555W'), flux_f555w[index], yerr=flux_f555w_err[index],
                fmt='o', c='k', ms=10)
ax_sed.errorbar(phangs_photometry.get_band_wave(band='F814W'), flux_f814w[index], yerr=flux_f814w_err[index],
                fmt='o', c='k', ms=10)

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

ax_sed.plot(data['wavelength'] * 1e-3, stellar_spectrum, linewidth=3, color='k', label='Stellar attenuated')
ax_sed.plot(data['wavelength'] * 1e-3, stellar_spectrum_unattenuated, linewidth=3, linestyle='--', color='b', label='Stellar unattenuated')

ax_sed.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
ax_sed.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
ax_sed.set_xlim(200 * 1e-3, 1.5)
ax_sed.set_ylim(0.0000009, 3e4)
ax_sed.set_xscale('log')
ax_sed.set_yscale('log')
ax_sed.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)


likelyhood_scan = (hist_age_ebv / ebv_bin_width).T / age_log_bin_width
likelyhood_scan /= np.max(likelyhood_scan)
likelyhood_scan[likelyhood_scan < 1e-4] = np.nan

ax_age_ebv.pcolormesh(age_log_bins, ebv_bins, np.log10(likelyhood_scan))
ax_age_ebv.scatter(best_age[0], best_ebv[0], s=150, c='r', marker='x')
ax_age_ebv.set_xscale('log')

ax_age_ebv.set_xlabel(r'Age, [Myr]', labelpad=0, fontsize=fontsize)
ax_age_ebv.set_ylabel(r'E(B-V)', labelpad=0, fontsize=fontsize)
ax_age_ebv.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)

ax_age_hist.plot(age_log_bins_center, hist_age / age_log_bin_width, color='k', linewidth=2)
ax_age_hist.set_xscale('log')
ax_age_hist.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
ax_age_hist.set_xticklabels([])
ax_age_hist.set_xlim(np.min(age_log_bins), np.max(age_log_bins))

ax_ebv_hist.plot(hist_ebv / ebv_bin_width, ebv_bins_center , color='k', linewidth=2)
ax_ebv_hist.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
ax_ebv_hist.set_yticklabels([])
ax_ebv_hist.set_ylim(np.min(ebv_bins), np.max(ebv_bins))

figure.savefig('plot_output/hst_sed.png')





exit()


# get a list of all the hst bands in the correct order
hst_bands = phangs_photometry.sort_band_list(
    band_list=(phangs_photometry.hst_targets[phangs_photometry.target_name]['acs_wfc1_observed_bands'] +
               phangs_photometry.hst_targets[phangs_photometry.target_name]['wfc3_uvis_observed_bands']))

figure, ax = PlotPhotometry.plot_cigale_sed_panel(
        hst_band_list=hst_bands,
        nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
        miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
        cutout_dict=cutout_dict, aperture_dict=aperture_dict)

figure.savefig('plot_output/test_sed.png')
figure.clf()

plt.imshow(hist_age_ebv, extent=(min_hist_age, max_hist_age, min_hist_ebv, max_hist_ebv),
           interpolation='nearest', aspect='auto')
# plt.show()
plt.savefig('plot_output/age_ebv')

exit()





# make the plot to inspect the aperture extraction
fig = PlotPhotometry.plot_circ_flux_extraction(
        hst_band_list=hst_bands,
        nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
        miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
        cutout_dict=cutout_dict, aperture_dict=aperture_dict,
        # vmax_vmin_hst=(0.05, 1.5), vmax_vmin_nircam=(0.1, 21), vmax_vmin_miri=(0.1, 21),
        # cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
        cmap_hst='Greys', cmap_nircam='Greys', cmap_miri='Greys',
        log_scale=True, axis_length=axis_length)










pdf_x_ebv = (pdf_grid_ebv[1:] + pdf_grid_ebv[:-1]) / 2.0
plt.plot(pdf_x_ebv, pdf_prob_ebv)
plt.show()






exit()





exit()




print(ebvdata[1])

chi2 = ebvdata[0]

plt.scatter(ebvdata[1], ebvdata[0])

plt.show()

exit()



plt.hist(ebvdata[1], bins=np.linspace(0, 1.5, 50))
plt.show()

exit()



h, xbin, ybin = np.histogram2d(age, ebv, bins=(age_bins, ebv_bins))

print(h)

plt.imshow(h)
plt.show()



exit()



# run cigale
cigale_wrapper.run_cigale()

exit()



cigale_wrapper_obj.sed_modules_params['dustext']['E_BV'] = [ebv]
cigale_wrapper_obj.sed_modules_params['sfh2exp']['age'] = [age]
parameter_list = ['attenuation.E_BV']
# run cigale
cigale_wrapper_obj.create_cigale_model()
# load model into constructor
cigale_wrapper_obj.load_cigale_model_block()

