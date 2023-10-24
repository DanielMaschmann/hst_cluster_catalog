import os.path

import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.data_access import CatalogAccess
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull


vi_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_hum_1.npy')
ub_hull_ogc_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_hum_1.npy')
vi_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_ogc_ubvi_ml_1.npy')
ub_hull_ogc_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_ogc_ubvi_ml_1.npy')

vi_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_hum_1.npy')
ub_hull_mid_ubvi_hum_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_hum_1.npy')
vi_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/vi_hull_mid_ubvi_ml_1.npy')
ub_hull_mid_ubvi_ml_1 = np.load('../segmentation/data_output/ub_hull_mid_ubvi_ml_1.npy')

vi_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_hum_3.npy')
ub_hull_young_ubvi_hum_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_hum_3.npy')
vi_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/vi_hull_young_ubvi_ml_3.npy')
ub_hull_young_ubvi_ml_3 = np.load('../segmentation/data_output/ub_hull_young_ubvi_ml_3.npy')


hull_gc_hum = ConvexHull(np.array([vi_hull_ogc_ubvi_hum_1, ub_hull_ogc_ubvi_hum_1]).T)
hull_cascade_hum = ConvexHull(np.array([vi_hull_mid_ubvi_hum_1, ub_hull_mid_ubvi_hum_1]).T)
hull_young_hum = ConvexHull(np.array([vi_hull_young_ubvi_hum_3, ub_hull_young_ubvi_hum_3]).T)

hull_gc_ml = ConvexHull(np.array([vi_hull_ogc_ubvi_ml_1, ub_hull_ogc_ubvi_ml_1]).T)
hull_cascade_ml = ConvexHull(np.array([vi_hull_mid_ubvi_ml_1, ub_hull_mid_ubvi_ml_1]).T)
hull_young_ml = ConvexHull(np.array([vi_hull_young_ubvi_ml_3, ub_hull_young_ubvi_ml_3]).T)

model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                               morph_mask_path=morph_mask_path, sample_table_path=sample_table_path)
target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])


catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc)
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml')
catalog_access.load_hst_cc_list(target_list=catalog_access.target_hst_cc, classify='ml', cluster_class='class3')


color_ub_hum = np.array([])
color_vi_hum = np.array([])
detect_ub_hum = np.array([])
detect_vi_hum = np.array([])

color_ub_ml = np.array([])
color_vi_ml = np.array([])
detect_ub_ml = np.array([])
detect_vi_ml = np.array([])


for index in range(len(target_list)):
    target = target_list[index]
    distance = dist_list[index]
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    print(target)
    # get alma data
    if os.path.isfile('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_150pc_broad_mom0.fits' % (galaxy_name, galaxy_name)):
        mom0_file_name = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_150pc_broad_mom0.fits' % (galaxy_name, galaxy_name)
        emom0_file_name = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_150pc_broad_emom0.fits' % (galaxy_name, galaxy_name)
    else:
        print('no 150 pc data')
        mom0_file_name = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_mom0.fits' % (galaxy_name, galaxy_name)
        emom0_file_name = '/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_emom0.fits' % (galaxy_name, galaxy_name)
    alma_hdu_mom0 = fits.open(mom0_file_name)
    alma_hdu_emom0 = fits.open(emom0_file_name)
    alma_wcs_mom0 = WCS(alma_hdu_mom0[0].header)
    alma_wcs_emom0 = WCS(alma_hdu_emom0[0].header)
    alma_data_mom0 = alma_hdu_mom0[0].data
    alma_data_emom0 = alma_hdu_emom0[0].data
    alma_snr_map = alma_data_mom0 / alma_data_emom0

    # get cluster data
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    detect_ub_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0))
    detect_vi_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0))

    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    detect_ub_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0))
    detect_vi_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0))
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    detect_ub_hum = np.concatenate([detect_ub_hum, detect_ub_hum_12, detect_ub_hum_3])
    detect_vi_hum = np.concatenate([detect_vi_hum, detect_vi_hum_12, detect_vi_hum_3])

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    detect_ub_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0))
    detect_vi_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0))
    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    detect_ub_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0))
    detect_vi_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0) &
                          (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0))
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    detect_ub_ml = np.concatenate([detect_ub_ml, detect_ub_ml_12, detect_ub_ml_3])
    detect_vi_ml = np.concatenate([detect_vi_ml, detect_vi_ml_12, detect_vi_ml_3])



in_hull_gc_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_gc_hum)
in_hull_cascade_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_cascade_hum)
in_hull_young_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_young_hum)

in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_gc_ml)
in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_cascade_ml)
in_hull_young_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)


print(sum(in_hull_gc_hum + in_hull_cascade_hum + in_hull_young_hum) / len(color_vi_hum))
print(sum(in_hull_gc_ml + in_hull_cascade_ml + in_hull_young_ml) / len(color_vi_ml))

