import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from photometry_tools import helper_func as hf
from photometry_tools.data_access import CatalogAccess
from cigale_helper import cigale_wrapper as cw
from matplotlib.patches import ConnectionPatch
import matplotlib
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.collections import LineCollection
import dust_tools.extinction_tools
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.analysis_tools import CigaleModelWrapper
from photometry_tools import plotting_tools
from scipy.constants import c

target = 'ngc0628c'
band_list = ['F275W', 'F336W', 'F435W', 'F555W', 'F658N', 'F814W', 'F200W', 'F300M', 'F335M', 'F360M']
# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc0628',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy', band_list=band_list, load_err=False)


ra_1 = 24.16348
dec_1 = 15.76589

# size of image
size_of_cutout = (2, 2)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_1, dec_cutout=dec_1,
                                                     cutout_size=size_of_cutout, include_err=False,
                                                     band_list=band_list)
source = SkyCoord(ra=ra_1, dec=dec_1, unit=(u.degree, u.degree), frame='fk5')

# compute flux from 50% encircled energy
aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                      recenter=True, recenter_rad=0.001,
                                                                      default_ee_rad=50)

# prepare CIGALE fit
flux_dict = {'n_obj': 1,
             'flux_F275W': [aperture_dict['aperture_dict_F275W']['flux']],
             'flux_F275W_err': [aperture_dict['aperture_dict_F275W']['flux'] * 0.1],
             'flux_F336W': [aperture_dict['aperture_dict_F336W']['flux']],
             'flux_F336W_err': [aperture_dict['aperture_dict_F336W']['flux'] * 0.1],
             'flux_F435W': [aperture_dict['aperture_dict_F435W']['flux']],
             'flux_F435W_err': [aperture_dict['aperture_dict_F435W']['flux'] * 0.1],
             'flux_F555W': [aperture_dict['aperture_dict_F555W']['flux']],
             'flux_F555W_err': [aperture_dict['aperture_dict_F555W']['flux'] * 0.1],
             'flux_F658N': [aperture_dict['aperture_dict_F658N']['flux']],
             'flux_F658N_err': [aperture_dict['aperture_dict_F658N']['flux'] * 0.1],
             'flux_F814W': [aperture_dict['aperture_dict_F814W']['flux']],
             'flux_F814W_err': [aperture_dict['aperture_dict_F814W']['flux'] * 0.1],
             'flux_F200W': [aperture_dict['aperture_dict_F200W']['flux']],
             'flux_F200W_err': [aperture_dict['aperture_dict_F200W']['flux'] * 0.1],
             'flux_F300M': [aperture_dict['aperture_dict_F300M']['flux']],
             'flux_F300M_err': [aperture_dict['aperture_dict_F300M']['flux'] * 0.1],
             'flux_F335M': [aperture_dict['aperture_dict_F335M']['flux']],
             'flux_F335M_err': [aperture_dict['aperture_dict_F335M']['flux'] * 0.1],
             'flux_F360M': [aperture_dict['aperture_dict_F360M']['flux']],
             'flux_F360M_err': [aperture_dict['aperture_dict_F360M']['flux'] * 0.1]}

cigale_wrapper = CigaleModelWrapper()

cigale_wrapper.create_cigale_flux_file(file_path='young_hst_nircam/data_file.dat',
                                       band_list=band_list,
                                       target_name=target[:-1], flux_dict=flux_dict)



# prepare CIGALE fit
flux_dict = {'n_obj': 1,
             'flux_F275W': [aperture_dict['aperture_dict_F275W']['flux']],
             'flux_F275W_err': [aperture_dict['aperture_dict_F275W']['flux'] * 0.1],
             'flux_F336W': [aperture_dict['aperture_dict_F336W']['flux']],
             'flux_F336W_err': [aperture_dict['aperture_dict_F336W']['flux'] * 0.1],
             'flux_F435W': [aperture_dict['aperture_dict_F435W']['flux']],
             'flux_F435W_err': [aperture_dict['aperture_dict_F435W']['flux'] * 0.1],
             'flux_F555W': [aperture_dict['aperture_dict_F555W']['flux']],
             'flux_F555W_err': [aperture_dict['aperture_dict_F555W']['flux'] * 0.1],
             'flux_F814W': [aperture_dict['aperture_dict_F814W']['flux']],
             'flux_F814W_err': [aperture_dict['aperture_dict_F814W']['flux'] * 0.1]}


band_list_hst_olny = ['F275W', 'F336W', 'F435W', 'F555W', 'F814W']

cigale_wrapper = CigaleModelWrapper()

cigale_wrapper.create_cigale_flux_file(file_path='young_hst_only/data_file.dat',
                                       band_list=band_list_hst_olny,
                                       target_name=target[:-1], flux_dict=flux_dict)





ra_1 = 24.18709
dec_1 = 15.79751

# size of image
size_of_cutout = (2, 2)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_1, dec_cutout=dec_1,
                                                     cutout_size=size_of_cutout, include_err=False,
                                                     band_list=band_list)
source = SkyCoord(ra=ra_1, dec=dec_1, unit=(u.degree, u.degree), frame='fk5')

# compute flux from 50% encircled energy
aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                      recenter=True, recenter_rad=0.001,
                                                                      default_ee_rad=50)

# prepare CIGALE fit
flux_dict = {'n_obj': 1,
             'flux_F275W': [aperture_dict['aperture_dict_F275W']['flux']],
             'flux_F275W_err': [aperture_dict['aperture_dict_F275W']['flux'] * 0.1],
             'flux_F336W': [aperture_dict['aperture_dict_F336W']['flux']],
             'flux_F336W_err': [aperture_dict['aperture_dict_F336W']['flux'] * 0.1],
             'flux_F435W': [aperture_dict['aperture_dict_F435W']['flux']],
             'flux_F435W_err': [aperture_dict['aperture_dict_F435W']['flux'] * 0.1],
             'flux_F555W': [aperture_dict['aperture_dict_F555W']['flux']],
             'flux_F555W_err': [aperture_dict['aperture_dict_F555W']['flux'] * 0.1],
             'flux_F658N': [aperture_dict['aperture_dict_F658N']['flux']],
             'flux_F658N_err': [aperture_dict['aperture_dict_F658N']['flux'] * 0.1],
             'flux_F814W': [aperture_dict['aperture_dict_F814W']['flux']],
             'flux_F814W_err': [aperture_dict['aperture_dict_F814W']['flux'] * 0.1],
             'flux_F200W': [aperture_dict['aperture_dict_F200W']['flux']],
             'flux_F200W_err': [aperture_dict['aperture_dict_F200W']['flux'] * 0.1],
             'flux_F300M': [aperture_dict['aperture_dict_F300M']['flux']],
             'flux_F300M_err': [aperture_dict['aperture_dict_F300M']['flux'] * 0.1],
             'flux_F335M': [aperture_dict['aperture_dict_F335M']['flux']],
             'flux_F335M_err': [aperture_dict['aperture_dict_F335M']['flux'] * 0.1],
             'flux_F360M': [aperture_dict['aperture_dict_F360M']['flux']],
             'flux_F360M_err': [aperture_dict['aperture_dict_F360M']['flux'] * 0.1]}

cigale_wrapper = CigaleModelWrapper()

cigale_wrapper.create_cigale_flux_file(file_path='old_hst_nircam/data_file.dat',
                                       band_list=band_list,
                                       target_name=target[:-1], flux_dict=flux_dict)



# prepare CIGALE fit
flux_dict = {'n_obj': 1,
             'flux_F275W': [aperture_dict['aperture_dict_F275W']['flux']],
             'flux_F275W_err': [aperture_dict['aperture_dict_F275W']['flux'] * 0.1],
             'flux_F336W': [aperture_dict['aperture_dict_F336W']['flux']],
             'flux_F336W_err': [aperture_dict['aperture_dict_F336W']['flux'] * 0.1],
             'flux_F435W': [aperture_dict['aperture_dict_F435W']['flux']],
             'flux_F435W_err': [aperture_dict['aperture_dict_F435W']['flux'] * 0.1],
             'flux_F555W': [aperture_dict['aperture_dict_F555W']['flux']],
             'flux_F555W_err': [aperture_dict['aperture_dict_F555W']['flux'] * 0.1],
             'flux_F814W': [aperture_dict['aperture_dict_F814W']['flux']],
             'flux_F814W_err': [aperture_dict['aperture_dict_F814W']['flux'] * 0.1]}


band_list_hst_olny = ['F275W', 'F336W', 'F435W', 'F555W', 'F814W']

cigale_wrapper = CigaleModelWrapper()

cigale_wrapper.create_cigale_flux_file(file_path='old_hst_only/data_file.dat',
                                       band_list=band_list_hst_olny,
                                       target_name=target[:-1], flux_dict=flux_dict)
