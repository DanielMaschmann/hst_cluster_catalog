""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import photometry_tools
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize, LogNorm
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad


# get access to HST cluster catalog
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

target_list = catalog_access.target_hst_cc
# target_name_list = catalog_access.phangs_galaxy_list

# catalog_access.load_morph_mask_target_list(target_list=target_list)
# np.save('data_output/morph_mask_data.npy', catalog_access.morph_mask_data)
catalog_access.morph_mask_data = np.load('morph_mask_data.npy', allow_pickle=True).item()

# print(catalog_access.morph_mask_data['ic1954'].keys())
#
# exit()

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


total_surf = np.array([0, 0, 0, 0, 0, 0])

n_total_hum = np.array([0, 0, 0, 0, 0, 0])
n_total_ml = np.array([0, 0, 0, 0, 0, 0])

n_young_hum = np.array([0, 0, 0, 0, 0, 0])
n_inter_hum = np.array([0, 0, 0, 0, 0, 0])
n_old_hum = np.array([0, 0, 0, 0, 0, 0])
n_class_1_hum = np.array([0, 0, 0, 0, 0, 0])
n_class_2_hum = np.array([0, 0, 0, 0, 0, 0])
n_class_3_hum = np.array([0, 0, 0, 0, 0, 0])

n_young_ml = np.array([0, 0, 0, 0, 0, 0])
n_inter_ml = np.array([0, 0, 0, 0, 0, 0])
n_old_ml = np.array([0, 0, 0, 0, 0, 0])
n_class_1_ml = np.array([0, 0, 0, 0, 0, 0])
n_class_2_ml = np.array([0, 0, 0, 0, 0, 0])
n_class_3_ml = np.array([0, 0, 0, 0, 0, 0])


for target in target_list:

    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target

    dist = catalog_access.dist_dict[galaxy_name]['dist']

    print('target ', target, ' dist ', dist)


    # load HST image
    phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                      nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                      miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                      target_name=galaxy_name,
                                      hst_data_ver='v1',
                                      nircam_data_ver='v0p4p2',
                                      miri_data_ver='v0p5')
    phangs_photometry.load_hst_band(band='F555W', load_err=False)
    # reproject the mask
    reproject_f555w = photometry_tools.plotting_tools.reproject_image(data=phangs_photometry.hst_bands_data['F555W_data_img'],
                                                                      wcs=phangs_photometry.hst_bands_data['F555W_wcs_img'],
                                                                      new_wcs=catalog_access.morph_mask_data[galaxy_name]['wcs'],
                                                                      new_shape=catalog_access.morph_mask_data[galaxy_name]['morph_map_float'].shape)
    observed_mask = (reproject_f555w != 0) & ~(np.isnan(reproject_f555w))


    # get size of a pixel in kpc square
    simbad_table = Simbad.query_object(galaxy_name)
    central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                                   unit=(u.hourangle, u.deg))

    pixel_surface_area_sq_kpc = helper_func.get_pixel_surface_area_sq_kp(
        wcs=catalog_access.morph_mask_data[galaxy_name]['wcs'], dist=dist)

    # get inclination for aperture correction
    incl = catalog_access.get_target_incl(target=galaxy_name)
    incl_corr_fact = 1 / np.cos(incl * np.pi/180)


    morph_map_str = catalog_access.morph_mask_data[galaxy_name]['morph_map_str']
    classified = morph_map_str != '0000000'
    if catalog_access.morph_mask_data[galaxy_name]['presence_ring']:
        mask_ring = catalog_access.morph_mask_data[galaxy_name]['mask_ring_1']
    else:
        mask_ring = catalog_access.morph_mask_data[galaxy_name]['mask_ring']
    # mask_central_ring = catalog_access.morph_mask_data[galaxy_name]['mask_ring_1']
    mask_center = catalog_access.morph_mask_data[galaxy_name]['mask_center'] * ~mask_ring
    mask_bar = (catalog_access.morph_mask_data[galaxy_name]['mask_bar'] +
                catalog_access.morph_mask_data[galaxy_name]['mask_lens']) * ~mask_center * ~mask_ring
    mask_arm = catalog_access.morph_mask_data[galaxy_name]['mask_arm'] * ~mask_bar * ~mask_center * ~mask_ring
    mask_disc = (catalog_access.morph_mask_data[galaxy_name]['mask_disc'] + catalog_access.morph_mask_data[galaxy_name]['mask_in_disc']) * ~mask_arm * ~mask_bar * ~mask_center * ~mask_ring

    presence_arm = catalog_access.morph_mask_data[galaxy_name]['presence_arm']
    presence_bar = catalog_access.morph_mask_data[galaxy_name]['presence_bar']
    presence_disc = catalog_access.morph_mask_data[galaxy_name]['presence_disc']

    print('presence_arm ', presence_arm)
    print('presence_bar ', presence_bar)
    print('presence_disc ', presence_disc)

    total_classified_all = np.sum(classified)
    frac_center_all = np.sum(mask_center) / total_classified_all
    frac_ring_all = np.sum(mask_ring) / total_classified_all
    frac_bar_all = np.sum(mask_bar) / total_classified_all
    frac_arm_all = np.sum(mask_arm) / total_classified_all
    frac_disc_all = np.sum(mask_disc) / total_classified_all

    total_classified = np.sum(classified * observed_mask)
    frac_center = np.sum(mask_center * observed_mask) / total_classified
    frac_ring = np.sum(mask_ring * observed_mask) / total_classified
    frac_bar = np.sum(mask_bar * observed_mask) / total_classified
    frac_arm = np.sum(mask_arm * observed_mask) / total_classified
    frac_disc = np.sum(mask_disc * observed_mask) / total_classified

    surface_center = np.sum(mask_center * observed_mask) * pixel_surface_area_sq_kpc * incl_corr_fact
    surface_ring = np.sum(mask_ring * observed_mask) * pixel_surface_area_sq_kpc * incl_corr_fact
    surface_bar = np.sum(mask_bar * observed_mask) * pixel_surface_area_sq_kpc * incl_corr_fact
    surface_arm = np.sum(mask_arm * observed_mask) * pixel_surface_area_sq_kpc * incl_corr_fact
    surface_disc = np.sum(mask_disc * observed_mask) * pixel_surface_area_sq_kpc * incl_corr_fact

    print('surface_center ', surface_center)
    print('surface_ring ', surface_ring)
    print('surface_bar ', surface_bar)
    print('surface_arm ', surface_arm)
    print('surface_disc ', surface_disc)

    print('total_classified_all ', total_classified_all)
    print('frac_center_all ', frac_center_all)
    print('frac_ring_all ', frac_ring_all)
    print('frac_bar_all ', frac_bar_all)
    print('frac_arm_all ', frac_arm_all)
    print('frac_disc_all ', frac_disc_all)
    print('sum_all ', frac_center_all + frac_ring_all + frac_bar_all + frac_arm_all + frac_disc_all)
    print(' ', )
    print('total_classified ', total_classified)
    print('frac_center ', frac_center)
    print('frac_ring ', frac_ring)
    print('frac_bar ', frac_bar)
    print('frac_arm ', frac_arm)
    print('frac_disc ', frac_disc)
    print('sum ', frac_center + frac_ring + frac_bar + frac_arm + frac_disc)

    # get number of objects
    # human classified
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ra_hum = np.concatenate([ra_hum_12, ra_hum_3])
    dec_hum = np.concatenate([dec_hum_12, dec_hum_3])
    cluster_class_hum = np.concatenate([cluster_class_hum_12, cluster_class_hum_3])
    age_hum = np.concatenate([age_hum_12, age_hum_3])
    pos_mask_dict_hum = catalog_access.get_morph_locations(target=target, ra=ra_hum, dec=dec_hum)
    # groups
    class_1_hum = cluster_class_hum == 1
    class_2_hum = cluster_class_hum == 2
    class_3_hum = cluster_class_hum == 3
    young_hum = age_hum <= 10
    inter_hum = (age_hum > 10) & (age_hum < 400)
    old_hum = (age_hum >= 400)

    # ML classified
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ra_ml = np.concatenate([ra_ml_12, ra_ml_3])
    dec_ml = np.concatenate([dec_ml_12, dec_ml_3])
    cluster_class_ml = np.concatenate([cluster_class_ml_12, cluster_class_ml_3])
    cluster_class_qual_ml = np.concatenate([cluster_class_qual_ml_12, cluster_class_qual_ml_3])
    age_ml = np.concatenate([age_ml_12, age_ml_3])
    pos_mask_dict_ml = catalog_access.get_morph_locations(target=target, ra=ra_ml, dec=dec_ml)
    # groups
    class_1_ml = cluster_class_ml == 1
    class_2_ml = cluster_class_ml == 2
    class_3_ml = cluster_class_ml == 3
    young_ml = age_ml <= 10
    inter_ml = (age_ml > 10) & (age_ml < 400)
    old_ml = (age_ml >= 400)
    qual_cut_ml = cluster_class_qual_ml >= 0.9

    # morphological classification
    mask_ring_hum = pos_mask_dict_hum['pos_mask_ring']
    mask_center_hum = pos_mask_dict_hum['pos_mask_center'] * ~mask_ring_hum
    mask_bar_hum = ((pos_mask_dict_hum['pos_mask_bar'] + pos_mask_dict_hum['pos_mask_lens']) *
                    ~mask_center_hum * ~mask_ring_hum)
    mask_arm_hum = pos_mask_dict_hum['pos_mask_arm'] * ~mask_center_hum * ~mask_ring_hum * ~mask_bar_hum
    mask_disc_hum = (pos_mask_dict_hum['pos_mask_disc'] * ~mask_arm_hum * ~mask_center_hum *
                     ~mask_ring_hum * ~mask_bar_hum)

    mask_ring_ml = pos_mask_dict_ml['pos_mask_ring']
    mask_center_ml = pos_mask_dict_ml['pos_mask_center'] * ~mask_ring_ml
    mask_bar_ml = ((pos_mask_dict_ml['pos_mask_bar'] + pos_mask_dict_ml['pos_mask_lens']) *
                   ~mask_center_ml * ~mask_ring_ml)
    mask_arm_ml = pos_mask_dict_ml['pos_mask_arm'] * ~mask_center_ml * ~mask_ring_ml * ~mask_bar_ml
    mask_disc_ml = (pos_mask_dict_ml['pos_mask_disc'] * ~mask_arm_ml * ~mask_center_ml *
                    ~mask_ring_ml * ~mask_bar_ml)

    total_surf[0] += surface_center
    total_surf[1] += surface_ring
    total_surf[2] += surface_bar
    total_surf[3] += surface_arm
    if presence_arm & presence_disc:
        total_surf[4] += surface_disc
    if presence_disc & ~presence_arm & ~presence_bar:
        total_surf[5] += surface_disc

    n_total_hum[0] += np.sum(mask_center_hum)
    n_total_hum[1] += np.sum(mask_ring_hum)
    n_total_hum[2] += np.sum(mask_bar_hum)
    n_total_hum[3] += np.sum(mask_arm_hum)
    if presence_arm & presence_disc:
        n_total_hum[4] += np.sum(mask_disc_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_total_hum[5] += np.sum(mask_disc_hum)

    n_young_hum[0] += np.sum(mask_center_hum * young_hum)
    n_young_hum[1] += np.sum(mask_ring_hum * young_hum)
    n_young_hum[2] += np.sum(mask_bar_hum * young_hum)
    n_young_hum[3] += np.sum(mask_arm_hum * young_hum)
    if presence_arm & presence_disc:
        n_young_hum[4] += np.sum(mask_disc_hum * young_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_young_hum[5] += np.sum(mask_disc_hum * young_hum)

    n_inter_hum[0] += np.sum(mask_center_hum * inter_hum)
    n_inter_hum[1] += np.sum(mask_ring_hum * inter_hum)
    n_inter_hum[2] += np.sum(mask_bar_hum * inter_hum)
    n_inter_hum[3] += np.sum(mask_arm_hum * inter_hum)
    if presence_arm & presence_disc:
        n_inter_hum[4] += np.sum(mask_disc_hum * inter_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_inter_hum[5] += np.sum(mask_disc_hum * inter_hum)

    n_old_hum[0] += np.sum(mask_center_hum * old_hum)
    n_old_hum[1] += np.sum(mask_ring_hum * old_hum)
    n_old_hum[2] += np.sum(mask_bar_hum * old_hum)
    n_old_hum[3] += np.sum(mask_arm_hum * old_hum)
    if presence_arm & presence_disc:
        n_old_hum[4] += np.sum(mask_disc_hum * old_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_old_hum[5] += np.sum(mask_disc_hum * old_hum)

    n_class_1_hum[0] += np.sum(mask_center_hum * class_1_hum)
    n_class_1_hum[1] += np.sum(mask_ring_hum * class_1_hum)
    n_class_1_hum[2] += np.sum(mask_bar_hum * class_1_hum)
    n_class_1_hum[3] += np.sum(mask_arm_hum * class_1_hum)
    if presence_arm & presence_disc:
        n_class_1_hum[4] += np.sum(mask_disc_hum * class_1_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_1_hum[5] += np.sum(mask_disc_hum * class_1_hum)

    n_class_2_hum[0] += np.sum(mask_center_hum * class_2_hum)
    n_class_2_hum[1] += np.sum(mask_ring_hum * class_2_hum)
    n_class_2_hum[2] += np.sum(mask_bar_hum * class_2_hum)
    n_class_2_hum[3] += np.sum(mask_arm_hum * class_2_hum)
    if presence_arm & presence_disc:
        n_class_2_hum[4] += np.sum(mask_disc_hum * class_2_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_2_hum[5] += np.sum(mask_disc_hum * class_2_hum)

    n_class_3_hum[0] += np.sum(mask_center_hum * class_3_hum)
    n_class_3_hum[1] += np.sum(mask_ring_hum * class_3_hum)
    n_class_3_hum[2] += np.sum(mask_bar_hum * class_3_hum)
    n_class_3_hum[3] += np.sum(mask_arm_hum * class_3_hum)
    if presence_arm & presence_disc:
        n_class_3_hum[4] += np.sum(mask_disc_hum * class_3_hum)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_3_hum[5] += np.sum(mask_disc_hum * class_3_hum)

    n_total_ml[0] += np.sum(mask_center_ml)
    n_total_ml[1] += np.sum(mask_ring_ml)
    n_total_ml[2] += np.sum(mask_bar_ml)
    n_total_ml[3] += np.sum(mask_arm_ml)
    if presence_arm & presence_disc:
        n_total_ml[4] += np.sum(mask_disc_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_total_ml[5] += np.sum(mask_disc_ml)

    n_young_ml[0] += np.sum(mask_center_ml * young_ml)
    n_young_ml[1] += np.sum(mask_ring_ml * young_ml)
    n_young_ml[2] += np.sum(mask_bar_ml * young_ml)
    n_young_ml[3] += np.sum(mask_arm_ml * young_ml)
    if presence_arm & presence_disc:
        n_young_ml[4] += np.sum(mask_disc_ml * young_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_young_ml[5] += np.sum(mask_disc_ml * young_ml)

    n_inter_ml[0] += np.sum(mask_center_ml * inter_ml)
    n_inter_ml[1] += np.sum(mask_ring_ml * inter_ml)
    n_inter_ml[2] += np.sum(mask_bar_ml * inter_ml)
    n_inter_ml[3] += np.sum(mask_arm_ml * inter_ml)
    if presence_arm & presence_disc:
        n_inter_ml[4] += np.sum(mask_disc_ml * inter_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_inter_ml[5] += np.sum(mask_disc_ml * inter_ml)

    n_old_ml[0] += np.sum(mask_center_ml * old_ml)
    n_old_ml[1] += np.sum(mask_ring_ml * old_ml)
    n_old_ml[2] += np.sum(mask_bar_ml * old_ml)
    n_old_ml[3] += np.sum(mask_arm_ml * old_ml)
    if presence_arm & presence_disc:
        n_old_ml[4] += np.sum(mask_disc_ml * old_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_old_ml[5] += np.sum(mask_disc_ml * old_ml)

    n_class_1_ml[0] += np.sum(mask_center_ml * class_1_ml)
    n_class_1_ml[1] += np.sum(mask_ring_ml * class_1_ml)
    n_class_1_ml[2] += np.sum(mask_bar_ml * class_1_ml)
    n_class_1_ml[3] += np.sum(mask_arm_ml * class_1_ml)
    if presence_arm & presence_disc:
        n_class_1_ml[4] += np.sum(mask_disc_ml * class_1_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_1_ml[5] += np.sum(mask_disc_ml * class_1_ml)

    n_class_2_ml[0] += np.sum(mask_center_ml * class_2_ml)
    n_class_2_ml[1] += np.sum(mask_ring_ml * class_2_ml)
    n_class_2_ml[2] += np.sum(mask_bar_ml * class_2_ml)
    n_class_2_ml[3] += np.sum(mask_arm_ml * class_2_ml)
    if presence_arm & presence_disc:
        n_class_2_ml[4] += np.sum(mask_disc_ml * class_2_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_2_ml[5] += np.sum(mask_disc_ml * class_2_ml)

    n_class_3_ml[0] += np.sum(mask_center_ml * class_3_ml)
    n_class_3_ml[1] += np.sum(mask_ring_ml * class_3_ml)
    n_class_3_ml[2] += np.sum(mask_bar_ml * class_3_ml)
    n_class_3_ml[3] += np.sum(mask_arm_ml * class_3_ml)
    if presence_arm & presence_disc:
        n_class_3_ml[4] += np.sum(mask_disc_ml * class_3_ml)
    if presence_disc & ~presence_arm & ~presence_bar:
        n_class_3_ml[5] += np.sum(mask_disc_ml * class_3_ml)

pos_array = [1, 2, 3, 4, 5, 6]
name_array = ['center', 'ring', 'bar', 'arm', 'between arms', 'flocculant disc']

fig, ax = plt.subplots(ncols=2, figsize=(18, 11))
fontsize = 16

print('!!!!')
print(np.sum(n_total_hum) / np.sum(total_surf))
print('n_total_hum ', n_total_hum)
print('n_total_ml ', n_total_ml)
print('total_surf ', total_surf)

print('n_young_hum ', n_young_hum)
print('n_inter_hum ', n_inter_hum)
print('n_old_hum ', n_old_hum)
print('n_class_1_hum ', n_class_1_hum)
print('n_class_2_hum ', n_class_2_hum)
print('n_class_3_hum ', n_class_3_hum)

print('n_young_ml ', n_young_ml)
print('n_inter_ml ', n_inter_ml)
print('n_old_ml ', n_old_ml)
print('n_class_1_ml ', n_class_1_ml)
print('n_class_2_ml ', n_class_2_ml)
print('n_class_3_ml ', n_class_3_ml)


ax[0].scatter(pos_array, n_total_hum / total_surf, color='k')
ax[0].plot(pos_array, n_total_hum / total_surf, linewidth=2, color='k', label='total')
ax[0].plot([0, 6], [np.sum(n_total_hum) / np.sum(total_surf), np.sum(n_total_hum) / np.sum(total_surf)], color='grey', linewidth=2, linestyle='--')
ax[0].text(4.5, np.sum(n_total_hum) / np.sum(total_surf) + 0.1, 'Mean', color='grey', fontsize=fontsize)


ax[0].scatter(pos_array, n_young_hum / total_surf)
ax[0].plot(pos_array, n_young_hum / total_surf, linewidth=2, label='young')
ax[0].scatter(pos_array, n_inter_hum / total_surf)
ax[0].plot(pos_array, n_inter_hum / total_surf, linewidth=2, label='inter')
ax[0].scatter(pos_array, n_old_hum / total_surf)
ax[0].plot(pos_array, n_old_hum / total_surf, linewidth=2, label='old')

ax[0].scatter(pos_array, n_class_1_hum / total_surf)
ax[0].plot(pos_array, n_class_1_hum / total_surf, linewidth=2, label='class 1')
ax[0].scatter(pos_array, n_class_2_hum / total_surf)
ax[0].plot(pos_array, n_class_2_hum / total_surf, linewidth=2, label='class 2')
ax[0].scatter(pos_array, n_class_3_hum / total_surf)
ax[0].plot(pos_array, n_class_3_hum / total_surf, linewidth=2, label='class 3')

ax[0].set_xticks(pos_array)
ax[0].xaxis.set_tick_params(rotation=45)
ax[0].set_xticklabels(name_array)
ax[0].set_title('Human', fontsize=fontsize)
ax[0].legend(frameon=False, fontsize=fontsize)
ax[0].set_ylabel(r'N/$\Sigma$ [kp$^{-2}$]', fontsize=fontsize)
ax[0].tick_params(axis='both', which='both', width=3, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0].set_yscale('log')

ax[1].scatter(pos_array, n_total_ml / total_surf, color='k')
ax[1].plot(pos_array, n_total_ml / total_surf, linewidth=2, color='k', label='total')
ax[1].plot([0, 6], [np.sum(n_total_ml) / np.sum(total_surf), np.sum(n_total_ml) / np.sum(total_surf)], color='grey', linewidth=2, linestyle='--')
ax[1].text(4.5, np.sum(n_total_ml) / np.sum(total_surf) + 0.1, 'Mean', color='grey', fontsize=fontsize)



ax[1].scatter(pos_array, n_young_ml / total_surf)
ax[1].plot(pos_array, n_young_ml / total_surf, linewidth=2, label='young')
ax[1].scatter(pos_array, n_inter_ml / total_surf)
ax[1].plot(pos_array, n_inter_ml / total_surf, linewidth=2, label='inter')
ax[1].scatter(pos_array, n_old_ml / total_surf)
ax[1].plot(pos_array, n_old_ml / total_surf, linewidth=2, label='old')

ax[1].scatter(pos_array, n_class_1_ml / total_surf)
ax[1].plot(pos_array, n_class_1_ml / total_surf, linewidth=2, label='class 1')
ax[1].scatter(pos_array, n_class_2_ml / total_surf)
ax[1].plot(pos_array, n_class_2_ml / total_surf, linewidth=2, label='class 2')
ax[1].scatter(pos_array, n_class_3_ml / total_surf)
ax[1].plot(pos_array, n_class_3_ml / total_surf, linewidth=2, label='class 3')

ax[1].set_xticks(pos_array)
ax[1].xaxis.set_tick_params(rotation=45)
ax[1].set_xticklabels(name_array)
ax[1].set_title('ML', fontsize=fontsize)
ax[1].legend(frameon=False, fontsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=3, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].set_yscale('log')




# plt.show()
plt.savefig('plot_output/cluster_dens_overview.png')
