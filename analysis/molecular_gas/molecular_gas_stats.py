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


snr_cut = 6
random_offset_pc = 300

mask_young_with_signal_hum_all = np.array([], dtype=bool)
mask_cascade_with_signal_hum_all = np.array([], dtype=bool)
mask_gc_with_signal_hum_all = np.array([], dtype=bool)
mask_no_class_with_signal_hum_all = np.array([], dtype=bool)
mask_young_with_no_signal_hum_all = np.array([], dtype=bool)
mask_cascade_with_no_signal_hum_all = np.array([], dtype=bool)
mask_gc_with_no_signal_hum_all = np.array([], dtype=bool)
mask_no_class_with_no_signal_hum_all = np.array([], dtype=bool)
mask_young_alma_coverage_hum_all = np.array([], dtype=bool)
mask_cascade_alma_coverage_hum_all = np.array([], dtype=bool)
mask_gc_alma_coverage_hum_all = np.array([], dtype=bool)
mask_no_class_alma_coverage_hum_all = np.array([], dtype=bool)


mask_young_with_signal_ml_all = np.array([], dtype=bool)
mask_cascade_with_signal_ml_all = np.array([], dtype=bool)
mask_gc_with_signal_ml_all = np.array([], dtype=bool)
mask_no_class_with_signal_ml_all = np.array([], dtype=bool)
mask_young_with_no_signal_ml_all = np.array([], dtype=bool)
mask_cascade_with_no_signal_ml_all = np.array([], dtype=bool)
mask_gc_with_no_signal_ml_all = np.array([], dtype=bool)
mask_no_class_with_no_signal_ml_all = np.array([], dtype=bool)
mask_young_alma_coverage_ml_all = np.array([], dtype=bool)
mask_cascade_alma_coverage_ml_all = np.array([], dtype=bool)
mask_gc_alma_coverage_ml_all = np.array([], dtype=bool)
mask_no_class_alma_coverage_ml_all = np.array([], dtype=bool)


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
    age_hum = age_hum_12 # np.concatenate([age_hum_12, age_hum_3])
    color_ub_hum = color_ub_hum_12 # np.concatenate([color_ub_hum_12, color_ub_hum_3])
    color_vi_hum = color_vi_hum_12 # np.concatenate([color_vi_hum_12, color_vi_hum_3])
    detect_ub_hum = detect_ub_hum_12 # np.concatenate([detect_ub_hum_12, detect_ub_hum_3])
    detect_vi_hum = detect_vi_hum_12 # np.concatenate([detect_vi_hum_12, detect_vi_hum_3])

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
    age_ml = age_ml_12 # np.concatenate([age_ml_12, age_ml_3])
    color_ub_ml = color_ub_ml_12 # np.concatenate([color_ub_ml_12, color_ub_ml_3])
    color_vi_ml = color_vi_ml_12 # np.concatenate([color_vi_ml_12, color_vi_ml_3])
    detect_ub_ml = detect_ub_ml_12 # np.concatenate([detect_ub_ml_12, detect_ub_ml_3])
    detect_vi_ml = detect_vi_ml_12 # np.concatenate([detect_vi_ml_12, detect_vi_ml_3])

    in_hull_gc_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_gc_hum)
    in_hull_cascade_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_cascade_hum)
    in_hull_young_hum = hf.points_in_hull(np.array([color_vi_hum, color_ub_hum]).T, hull_young_hum)

    in_hull_gc_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_gc_ml)
    in_hull_cascade_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_cascade_ml)
    in_hull_young_ml = hf.points_in_hull(np.array([color_vi_ml, color_ub_ml]).T, hull_young_ml)

    detectable_hum = detect_ub_hum * detect_vi_hum
    detectable_ml = detect_ub_ml * detect_vi_ml

    young_hum = age_hum < 10
    young_ml = age_ml < 10

    mask_gc_hum = in_hull_gc_hum * detectable_hum * np.invert(young_hum)
    mask_cascade_hum = in_hull_cascade_hum * detectable_hum * np.invert(young_hum)
    mask_young_hum = ((detectable_hum * young_hum * (in_hull_gc_hum + in_hull_cascade_hum)) +
                      (in_hull_young_hum * detectable_hum * np.invert(in_hull_cascade_hum)))
    mask_no_class_hum = (np.invert(in_hull_gc_hum + in_hull_cascade_hum + in_hull_young_hum) * detectable_hum)

    mask_gc_ml = in_hull_gc_ml * detectable_ml * np.invert(young_ml)
    mask_cascade_ml = in_hull_cascade_ml * detectable_ml * np.invert(young_ml)
    mask_young_ml = ((detectable_ml * young_ml * (in_hull_gc_ml + in_hull_cascade_ml)) +
                      (in_hull_young_ml * detectable_ml * np.invert(in_hull_cascade_ml)))
    mask_no_class_ml = (np.invert(in_hull_gc_ml + in_hull_cascade_ml + in_hull_young_ml) * detectable_ml)


    # get cluster coordinates
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    ra_hum = ra_hum_12 # np.concatenate([ra_hum_12, ra_hum_3])
    dec_hum = dec_hum_12 # np.concatenate([dec_hum_12, dec_hum_3])
    coords_world_hum = SkyCoord(ra=ra_hum*u.deg, dec=dec_hum*u.deg)
    coords_pix_hum = alma_wcs_mom0.world_to_pixel(coords_world_hum)

    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    ra_ml = ra_ml_12 # np.concatenate([ra_ml_12, ra_ml_3])
    dec_ml = dec_ml_12 # np.concatenate([dec_ml_12, dec_ml_3])
    coords_world_ml = SkyCoord(ra=ra_ml*u.deg, dec=dec_ml*u.deg)
    coords_pix_ml = alma_wcs_mom0.world_to_pixel(coords_world_ml)

    # calculate the pixel offset for perturbation



    # check which cluster is associated with molecular gas
    mask_alma_snr_hum = np.zeros(len(color_ub_hum), dtype=bool)
    x_pixel_coords_hum = np.array(np.rint(coords_pix_hum[1]), dtype=int)
    y_pixel_coords_hum = np.array(np.rint(coords_pix_hum[0]), dtype=int)
    mask_alma_point_coverage_hum = ((coords_pix_hum[1] > 0) & (coords_pix_hum[1] < (alma_snr_map.shape[0]-1)) &
                          (coords_pix_hum[0] > 0) & (coords_pix_hum[0] < (alma_snr_map.shape[1]-1)))
    mask_alma_snr_hum[mask_alma_point_coverage_hum] = (
            alma_snr_map[x_pixel_coords_hum[mask_alma_point_coverage_hum], y_pixel_coords_hum[mask_alma_point_coverage_hum]] > snr_cut)
    mask_alma_no_coverage_hum = np.ones(len(color_ub_hum), dtype=bool)
    mask_alma_no_coverage_hum[mask_alma_point_coverage_hum] = False
    mask_alma_no_coverage_hum[mask_alma_point_coverage_hum] = np.isnan(alma_snr_map[x_pixel_coords_hum[mask_alma_point_coverage_hum], y_pixel_coords_hum[mask_alma_point_coverage_hum]])
    mask_alma_coverage_hum = np.invert(mask_alma_no_coverage_hum)

    # check which cluster is associated with molecular gas
    mask_alma_snr_ml = np.zeros(len(color_ub_ml), dtype=bool)
    x_pixel_coords_ml = np.array(np.rint(coords_pix_ml[1]), dtype=int)
    y_pixel_coords_ml = np.array(np.rint(coords_pix_ml[0]), dtype=int)
    mask_alma_point_coverage_ml = ((coords_pix_ml[1] > 0) & (coords_pix_ml[1] < (alma_snr_map.shape[0]-1)) &
                          (coords_pix_ml[0] > 0) & (coords_pix_ml[0] < (alma_snr_map.shape[1]-1)))
    mask_alma_snr_ml[mask_alma_point_coverage_ml] = (
            alma_snr_map[x_pixel_coords_ml[mask_alma_point_coverage_ml], y_pixel_coords_ml[mask_alma_point_coverage_ml]] > snr_cut)
    mask_alma_no_coverage_ml = np.ones(len(color_ub_ml), dtype=bool)
    mask_alma_no_coverage_ml[mask_alma_point_coverage_ml] = False
    mask_alma_no_coverage_ml[mask_alma_point_coverage_ml] = np.isnan(alma_snr_map[x_pixel_coords_ml[mask_alma_point_coverage_ml], y_pixel_coords_ml[mask_alma_point_coverage_ml]])
    mask_alma_coverage_ml = np.invert(mask_alma_no_coverage_ml)

    mask_young_with_signal_hum = mask_young_hum * mask_alma_snr_hum * mask_alma_coverage_hum
    mask_cascade_with_signal_hum = mask_cascade_hum * mask_alma_snr_hum * mask_alma_coverage_hum
    mask_gc_with_signal_hum = mask_gc_hum * mask_alma_snr_hum * mask_alma_coverage_hum
    mask_no_class_with_signal_hum = mask_no_class_hum * mask_alma_snr_hum * mask_alma_coverage_hum

    mask_young_with_no_signal_hum = mask_young_hum * np.invert(mask_alma_snr_hum) * mask_alma_coverage_hum
    mask_cascade_with_no_signal_hum = mask_cascade_hum * np.invert(mask_alma_snr_hum) * mask_alma_coverage_hum
    mask_gc_with_no_signal_hum = mask_gc_hum * np.invert(mask_alma_snr_hum) * mask_alma_coverage_hum
    mask_no_class_with_no_signal_hum = mask_no_class_hum * np.invert(mask_alma_snr_hum) * mask_alma_coverage_hum

    mask_young_alma_coverage_hum = mask_young_hum * mask_alma_coverage_hum
    mask_cascade_alma_coverage_hum = mask_cascade_hum * mask_alma_coverage_hum
    mask_gc_alma_coverage_hum = mask_gc_hum * mask_alma_coverage_hum
    mask_no_class_alma_coverage_hum = mask_no_class_hum * mask_alma_coverage_hum

    mask_young_with_signal_ml = mask_young_ml * mask_alma_snr_ml * mask_alma_coverage_ml
    mask_cascade_with_signal_ml = mask_cascade_ml * mask_alma_snr_ml * mask_alma_coverage_ml
    mask_gc_with_signal_ml = mask_gc_ml * mask_alma_snr_ml * mask_alma_coverage_ml
    mask_no_class_with_signal_ml = mask_no_class_ml * mask_alma_snr_ml * mask_alma_coverage_ml

    mask_young_with_no_signal_ml = mask_young_ml * np.invert(mask_alma_snr_ml) * mask_alma_coverage_ml
    mask_cascade_with_no_signal_ml = mask_cascade_ml * np.invert(mask_alma_snr_ml) * mask_alma_coverage_ml
    mask_gc_with_no_signal_ml = mask_gc_ml * np.invert(mask_alma_snr_ml) * mask_alma_coverage_ml
    mask_no_class_with_no_signal_ml = mask_no_class_ml * np.invert(mask_alma_snr_ml) * mask_alma_coverage_ml

    mask_young_alma_coverage_ml = mask_young_ml * mask_alma_coverage_ml
    mask_cascade_alma_coverage_ml = mask_cascade_ml * mask_alma_coverage_ml
    mask_gc_alma_coverage_ml = mask_gc_ml * mask_alma_coverage_ml
    mask_no_class_alma_coverage_ml = mask_no_class_ml * mask_alma_coverage_ml

    mask_young_with_signal_hum_all = np.concatenate([mask_young_with_signal_hum_all, mask_young_with_signal_hum])
    mask_cascade_with_signal_hum_all = np.concatenate([mask_cascade_with_signal_hum_all, mask_cascade_with_signal_hum])
    mask_gc_with_signal_hum_all = np.concatenate([mask_gc_with_signal_hum_all, mask_gc_with_signal_hum])
    mask_no_class_with_signal_hum_all = np.concatenate([mask_no_class_with_signal_hum_all, mask_no_class_with_signal_hum])
    mask_young_with_no_signal_hum_all = np.concatenate([mask_young_with_no_signal_hum_all, mask_young_with_no_signal_hum])
    mask_cascade_with_no_signal_hum_all = np.concatenate([mask_cascade_with_no_signal_hum_all, mask_cascade_with_no_signal_hum])
    mask_gc_with_no_signal_hum_all = np.concatenate([mask_gc_with_no_signal_hum_all, mask_gc_with_no_signal_hum])
    mask_no_class_with_no_signal_hum_all = np.concatenate([mask_no_class_with_no_signal_hum_all, mask_no_class_with_no_signal_hum])
    mask_young_alma_coverage_hum_all = np.concatenate([mask_young_alma_coverage_hum_all, mask_young_alma_coverage_hum])
    mask_cascade_alma_coverage_hum_all = np.concatenate([mask_cascade_alma_coverage_hum_all, mask_cascade_alma_coverage_hum])
    mask_gc_alma_coverage_hum_all = np.concatenate([mask_gc_alma_coverage_hum_all, mask_gc_alma_coverage_hum])
    mask_no_class_alma_coverage_hum_all = np.concatenate([mask_no_class_alma_coverage_hum_all, mask_no_class_alma_coverage_hum])

    mask_young_with_signal_ml_all = np.concatenate([mask_young_with_signal_ml_all, mask_young_with_signal_ml])
    mask_cascade_with_signal_ml_all = np.concatenate([mask_cascade_with_signal_ml_all, mask_cascade_with_signal_ml])
    mask_gc_with_signal_ml_all = np.concatenate([mask_gc_with_signal_ml_all, mask_gc_with_signal_ml])
    mask_no_class_with_signal_ml_all = np.concatenate([mask_no_class_with_signal_ml_all, mask_no_class_with_signal_ml])
    mask_young_with_no_signal_ml_all = np.concatenate([mask_young_with_no_signal_ml_all, mask_young_with_no_signal_ml])
    mask_cascade_with_no_signal_ml_all = np.concatenate([mask_cascade_with_no_signal_ml_all, mask_cascade_with_no_signal_ml])
    mask_gc_with_no_signal_ml_all = np.concatenate([mask_gc_with_no_signal_ml_all, mask_gc_with_no_signal_ml])
    mask_no_class_with_no_signal_ml_all = np.concatenate([mask_no_class_with_no_signal_ml_all, mask_no_class_with_no_signal_ml])
    mask_young_alma_coverage_ml_all = np.concatenate([mask_young_alma_coverage_ml_all, mask_young_alma_coverage_ml])
    mask_cascade_alma_coverage_ml_all = np.concatenate([mask_cascade_alma_coverage_ml_all, mask_cascade_alma_coverage_ml])
    mask_gc_alma_coverage_ml_all = np.concatenate([mask_gc_alma_coverage_ml_all, mask_gc_alma_coverage_ml])
    mask_no_class_alma_coverage_ml_all = np.concatenate([mask_no_class_alma_coverage_ml_all, mask_no_class_alma_coverage_ml])


    # figure = plt.figure(figsize=(25, 25))
    # ax_alma = figure.add_axes([0.05, 0.05, 0.9, 0.9], projection=alma_wcs_mom0)
    # ax_alma.imshow(alma_data_mom0)
    # # ax_alma.scatter(coords_pix_hum[0], coords_pix_hum[1])
    # # ax_alma.scatter(coords_pix_hum[0][mask_alma_snr_hum], coords_pix_hum[1][mask_alma_snr_hum])
    # # ax_alma.scatter(coords_pix_hum[0][mask_alma_no_coverage_hum], coords_pix_hum[1][mask_alma_no_coverage_hum], color='r')
    # plt.show()


frac_young_with_signal_hum = sum(mask_young_with_signal_hum_all) / sum(mask_young_alma_coverage_hum_all)
frac_cascade_with_signal_hum = sum(mask_cascade_with_signal_hum_all) / sum(mask_cascade_alma_coverage_hum_all)
frac_gc_with_signal_hum = sum(mask_gc_with_signal_hum_all) / sum(mask_gc_alma_coverage_hum_all)
frac_no_class_with_signal_hum = sum(mask_no_class_with_signal_hum_all) / sum(mask_no_class_alma_coverage_hum_all)

frac_young_with_no_signal_hum = sum(mask_young_with_no_signal_hum_all) / sum(mask_young_alma_coverage_hum_all)
frac_cascade_with_no_signal_hum = sum(mask_cascade_with_no_signal_hum_all) / sum(mask_cascade_alma_coverage_hum_all)
frac_gc_with_no_signal_hum = sum(mask_gc_with_no_signal_hum_all) / sum(mask_gc_alma_coverage_hum_all)
frac_no_class_with_no_signal_hum = sum(mask_no_class_with_no_signal_hum_all) / sum(mask_no_class_alma_coverage_hum_all)

frac_young_with_signal_ml = sum(mask_young_with_signal_ml_all) / sum(mask_young_alma_coverage_ml_all)
frac_cascade_with_signal_ml = sum(mask_cascade_with_signal_ml_all) / sum(mask_cascade_alma_coverage_ml_all)
frac_gc_with_signal_ml = sum(mask_gc_with_signal_ml_all) / sum(mask_gc_alma_coverage_ml_all)
frac_no_class_with_signal_ml = sum(mask_no_class_with_signal_ml_all) / sum(mask_no_class_alma_coverage_ml_all)

frac_young_with_no_signal_ml = sum(mask_young_with_no_signal_ml_all) / sum(mask_young_alma_coverage_ml_all)
frac_cascade_with_no_signal_ml = sum(mask_cascade_with_no_signal_ml_all) / sum(mask_cascade_alma_coverage_ml_all)
frac_gc_with_no_signal_ml = sum(mask_gc_with_no_signal_ml_all) / sum(mask_gc_alma_coverage_ml_all)
frac_no_class_with_no_signal_ml = sum(mask_no_class_with_no_signal_ml_all) / sum(mask_no_class_alma_coverage_ml_all)


print('%.1f' % (frac_young_with_signal_hum * 100), '%.1f' % (frac_young_with_no_signal_hum * 100))
print('%.1f' % (frac_cascade_with_signal_hum * 100), '%.1f' % (frac_cascade_with_no_signal_hum * 100))
print('%.1f' % (frac_gc_with_signal_hum * 100), '%.1f' % (frac_gc_with_no_signal_hum * 100))
print('%.1f' % (frac_no_class_with_signal_hum * 100), '%.1f' % (frac_no_class_with_no_signal_hum * 100))


print('%.1f' % (frac_young_with_signal_ml * 100), '%.1f' % (frac_young_with_no_signal_ml * 100))
print('%.1f' % (frac_cascade_with_signal_ml * 100), '%.1f' % (frac_cascade_with_no_signal_ml * 100))
print('%.1f' % (frac_gc_with_signal_ml * 100), '%.1f' % (frac_gc_with_no_signal_ml * 100))
print('%.1f' % (frac_no_class_with_signal_ml * 100), '%.1f' % (frac_no_class_with_no_signal_ml * 100))

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 7))
fontsize = 25

bar_width = 0.3

pos_young = 1
pos_cascade = 2
pos_gc = 3
pos_no_class = 4

ax.bar(x=pos_young - 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_young - 0.6*bar_width, height=frac_young_with_signal_hum * 100, edgecolor='tab:blue',  color='tab:blue', width=bar_width)
ax.bar(x=pos_young + 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_young + 0.6*bar_width, height=frac_young_with_signal_ml * 100, edgecolor='tab:blue',  color='tab:blue', width=bar_width)

ax.bar(x=pos_cascade - 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_cascade - 0.6*bar_width, height=frac_cascade_with_signal_hum * 100, edgecolor='tab:green',  color='tab:green', width=bar_width)
ax.bar(x=pos_cascade + 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_cascade + 0.6*bar_width, height=frac_cascade_with_signal_ml * 100, edgecolor='tab:green',  color='tab:green', width=bar_width)

ax.bar(x=pos_gc - 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_gc - 0.6*bar_width, height=frac_gc_with_signal_hum * 100, edgecolor='tab:red',  color='tab:red', width=bar_width)
ax.bar(x=pos_gc + 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
ax.bar(x=pos_gc + 0.6*bar_width, height=frac_gc_with_signal_ml * 100, edgecolor='tab:red',  color='tab:red', width=bar_width)

# ax.bar(x=pos_no_class - 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
# ax.bar(x=pos_no_class - 0.6*bar_width, height=frac_no_class_with_signal_hum * 100, edgecolor='gray',  color='gray', width=bar_width)
# ax.bar(x=pos_no_class + 0.6*bar_width, height=100, color='k', edgecolor='k', facecolor='none', lw=2., hatch='//', alpha=0.7, width=bar_width)
# ax.bar(x=pos_no_class + 0.6*bar_width, height=frac_no_class_with_signal_ml * 100, edgecolor='gray',  color='gray', width=bar_width)

ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Young Cluster Locus', 'Middle-Age Plume', 'Old Globular Cluster Clump'],
                   fontsize=fontsize, rotation=0)
ax.set_ylabel(r'% ' + r'Clusters with S/N$_{\rm CO(2-1)}$ > %i' % snr_cut, fontsize=fontsize, labelpad=-5)

ax.text(pos_young - 0.6*bar_width, 100, 'Hum', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)
ax.text(pos_young + 0.6*bar_width, 100, 'ML', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)

ax.text(pos_cascade - 0.6*bar_width, 100, 'Hum', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)
ax.text(pos_cascade + 0.6*bar_width, 100, 'ML', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)

ax.text(pos_gc - 0.6*bar_width, 100, 'Hum', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)
ax.text(pos_gc + 0.6*bar_width, 100, 'ML', horizontalalignment='center', verticalalignment='bottom', color='k', fontsize=fontsize)

ax.set_ylim(0, 108)

fig.subplots_adjust(left=0.07, bottom=0.06, right=0.995, top=0.995, wspace=0.01, hspace=0.01)

plt.savefig('plot_output/molecular_gas_stats_snr_%i.png' % snr_cut)
plt.savefig('plot_output/molecular_gas_stats_snr_%i.pdf' % snr_cut)







