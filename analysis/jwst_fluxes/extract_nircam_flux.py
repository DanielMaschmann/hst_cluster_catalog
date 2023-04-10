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


size_of_cutout = (3, 3)
band_list = ['F300M', 'F335M']


for target in ['ngc0628', 'ngc1087', 'ngc1300', 'ngc1365', 'ngc1385', 'ngc1433', 'ngc1512', 'ngc1566', 'ngc1672',
               'ngc3627', 'ngc4303', 'ngc4321', 'ngc4535', 'ngc5068', 'ngc7496']:
    print(target)

    # initialize photometry tools
    phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                      nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                      miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                      target_name=target,
                                      hst_data_ver='v1',
                                      nircam_data_ver='v0p7p3',
                                      miri_data_ver='v0p7p3')
    # load data
    phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy', band_list=band_list, load_err=False)

    # load table
    cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
    hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
    catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                                hst_obs_hdr_file_path=hst_obs_hdr_file_path)

    if target == 'ngc0628':
        catalog_access.load_hst_cc_list(target_list=['ngc0628c', 'ngc0628e'])
        catalog_access.load_hst_cc_list(target_list=['ngc0628c', 'ngc0628e'], cluster_class='class3')

        ra_12_c, dec_12_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c')
        ra_12_e, dec_12_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e')
        ra_3_c, dec_3_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c', cluster_class='class3')
        ra_3_e, dec_3_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e', cluster_class='class3')

        class_12_c = catalog_access.get_hst_cc_class_human(target='ngc0628c')
        class_12_e = catalog_access.get_hst_cc_class_human(target='ngc0628e')
        class_3_c = catalog_access.get_hst_cc_class_human(target='ngc0628c', cluster_class='class3')
        class_3_e = catalog_access.get_hst_cc_class_human(target='ngc0628e', cluster_class='class3')

        age_12_c = catalog_access.get_hst_cc_age(target='ngc0628c')
        age_12_e = catalog_access.get_hst_cc_age(target='ngc0628e')
        age_3_c = catalog_access.get_hst_cc_age(target='ngc0628c', cluster_class='class3')
        age_3_e = catalog_access.get_hst_cc_age(target='ngc0628e', cluster_class='class3')

        ra_list = np.concatenate([ra_12_c, ra_12_e, ra_3_c, ra_3_e])
        dec_list = np.concatenate([dec_12_c, dec_12_e, dec_3_c, dec_3_e])
        class_list = np.concatenate([class_12_c, class_12_e, class_3_c, class_3_e])
        age_list = np.concatenate([age_12_c, age_12_e, age_3_c, age_3_e])

    else:
        catalog_access.load_hst_cc_list(target_list=[target])
        catalog_access.load_hst_cc_list(target_list=[target], cluster_class='class3')
        ra_12, dec_12 = catalog_access.get_hst_cc_coords_world(target=target)
        ra_3, dec_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')

        class_12 = catalog_access.get_hst_cc_class_human(target=target)
        class_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')

        age_12 = catalog_access.get_hst_cc_age(target=target)
        age_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')

        ra_list = np.concatenate([ra_12, ra_3])
        dec_list = np.concatenate([dec_12, dec_3])
        class_list = np.concatenate([class_12, class_3])
        age_list = np.concatenate([age_12, age_3])


    flux_f300m = np.zeros(len(ra_list))
    flux_f300m_err = np.zeros(len(ra_list))
    flux_f335m = np.zeros(len(ra_list))
    flux_f335m_err = np.zeros(len(ra_list))
    for cluster_index in range(len(ra_list)):

        cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_list[cluster_index],
                                                             dec_cutout=dec_list[cluster_index], cutout_size=size_of_cutout,
                                                             band_list=band_list, include_err=False)
        source = SkyCoord(ra=ra_list[cluster_index], dec=dec_list[cluster_index], unit=(u.degree, u.degree), frame='fk5')

        if (cutout_dict['F300M_img_cutout'].data is None) | (cutout_dict['F335M_img_cutout'].data is None):
            flux_f300m[cluster_index] = np.nan
            flux_f300m_err[cluster_index] = np.nan
            flux_f335m[cluster_index] = np.nan
            flux_f335m_err[cluster_index] = np.nan
            continue

        # compute flux from 50% encircled energy
        aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                              recenter=True, recenter_rad=0.01,
                                                                              default_ee_rad=50)

        flux_f300m[cluster_index] = aperture_dict['aperture_dict_F300M']['flux']
        flux_f300m_err[cluster_index] = aperture_dict['aperture_dict_F300M']['flux_err']

        flux_f335m[cluster_index] = aperture_dict['aperture_dict_F335M']['flux']
        flux_f335m_err[cluster_index] = aperture_dict['aperture_dict_F335M']['flux_err']


    mag_f300m = hf.conv_mjy2ab_mag(flux=flux_f300m)
    mag_f335m = hf.conv_mjy2ab_mag(flux=flux_f335m)
    abs_mag_f300m = hf.conv_mag2abs_mag(mag=mag_f300m, dist=phangs_photometry.dist_dict[target]['dist'])
    abs_mag_f335m = hf.conv_mag2abs_mag(mag=mag_f335m, dist=phangs_photometry.dist_dict[target]['dist'])

    flux_dict = {
        'flux_f300m': flux_f300m,
        'flux_f300m_err': flux_f300m_err,
        'flux_f335m': flux_f335m,
        'flux_f335m_err': flux_f335m_err,
        'mag_f300m': mag_f300m,
        'mag_f335m': mag_f335m,
        'abs_mag_f300m': abs_mag_f300m,
        'abs_mag_f335m': abs_mag_f335m,
        'class_list': class_list,
        'age_list': age_list
    }

    np.save('data_output/flux_dict_%s.npy' % target, flux_dict)



