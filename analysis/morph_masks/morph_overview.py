""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import photometry_tools
from photometry_tools.analysis_tools import AnalysisTools
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
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path)
target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
# sort target list after distance
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]
# catalog_access.load_morph_mask_target_list(target_list=target_list)
# np.save('data_output/morph_mask_data.npy', catalog_access.morph_mask_data)
catalog_access.morph_mask_data = np.load('data_output/morph_mask_data.npy', allow_pickle=True).item()

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

dss_cutout_size = {'ngc4826': 18, 'ngc5068': 18, 'ngc3621': 18, 'ic5332': 18,
                   'ngc6744': 18, 'ngc2903': 18, 'ngc0628': 18, 'ngc3351': 18,
                   'ngc3627': 18, 'ngc2835': 18, 'ic1954': 18, 'ngc4254': 18,
                   'ngc1097': 18, 'ngc5248': 18, 'ngc4571': 18, 'ngc4298': 18,
                   'ngc4689': 18, 'ngc4321': 18, 'ngc4569': 18, 'ngc4535': 18,
                   'ngc1087': 18, 'ngc1792': 18, 'ngc4548': 18, 'ngc4536': 18,
                   'ngc4303': 18, 'ngc1385': 18, 'ngc1566': 18, 'ngc1512': 18,
                   'ngc7496': 18, 'ngc1433': 18, 'ngc1300': 18, 'ngc1317': 18,
                   'ngc1672': 18, 'ngc1559': 18, 'ngc1365': 18, 'ngc0685': 18,
                   'ngc4654': 18, 'ngc2775': 18}


fontsize = 17
cma_img = 'Greys'

color_center = 'tab:brown'
color_disc = 'tab:purple'
color_arm = 'tab:olive'
color_bar = 'tab:cyan'
color_ring = 'tab:pink'
color_not_class = 'tab:gray'

color_hst = 'black'

color_c1 = 'forestgreen'
color_c2 = 'darkorange'
color_c3 = 'royalblue'

for target in target_list:
    print('target ', target)
    if target == 'ngc0628c':
        target = 'ngc0628'
    if target == 'ngc0628e':
        continue

    simbad_table = Simbad.query_object(target)
    central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                                   unit=(u.hourangle, u.deg))

    # load DSS image
    paths_dss = SkyView.get_images(position=target, survey='DSS2 IR', radius=(dss_cutout_size[target]+1)*u.arcmin)
    data_dss = paths_dss[0][0].data
    wcs_dss = WCS(paths_dss[0][0].header)

    # load HST image
    phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                      nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                      miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                      target_name=target,
                                      hst_data_ver='v1',
                                      nircam_data_ver='v0p4p2',
                                      miri_data_ver='v0p5')
    phangs_photometry.load_hst_band(band='F555W', load_err=False)
    # reproject the mask
    reproject_f555w = photometry_tools.plotting_tools.reproject_image(data=phangs_photometry.hst_bands_data['F555W_data_img'],
                                                                      wcs=phangs_photometry.hst_bands_data['F555W_wcs_img'],
                                                                      new_wcs=catalog_access.morph_mask_data[target]['wcs'],
                                                                      new_shape=catalog_access.morph_mask_data[target]['morph_map_float'].shape)
    observed_mask = (reproject_f555w != 0) & ~(np.isnan(reproject_f555w))

    # reproject the mask
    reproject_dss = photometry_tools.plotting_tools.reproject_image(data=data_dss,
                                                                    wcs=wcs_dss,
                                                                    new_wcs=catalog_access.morph_mask_data[target]['wcs'],
                                                                    new_shape=catalog_access.morph_mask_data[target]['morph_map_float'].shape)

    if target == 'ngc0628':
        ra_hum_12_c, dec_hum_12_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c')
        ra_hum_3_c, dec_hum_3_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c', cluster_class='class3')
        ra_hum_12_e, dec_hum_12_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e')
        ra_hum_3_e, dec_hum_3_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e', cluster_class='class3')
        cluster_class_hum_12_c = catalog_access.get_hst_cc_class_human(target='ngc0628c')
        cluster_class_hum_3_c = catalog_access.get_hst_cc_class_human(target='ngc0628c', cluster_class='class3')
        cluster_class_hum_12_e = catalog_access.get_hst_cc_class_human(target='ngc0628e')
        cluster_class_hum_3_e = catalog_access.get_hst_cc_class_human(target='ngc0628e', cluster_class='class3')
        age_hum_12_c = catalog_access.get_hst_cc_age(target='ngc0628c')
        age_hum_3_c = catalog_access.get_hst_cc_age(target='ngc0628c', cluster_class='class3')
        age_hum_12_e = catalog_access.get_hst_cc_age(target='ngc0628e')
        age_hum_3_e = catalog_access.get_hst_cc_age(target='ngc0628e', cluster_class='class3')
        ra_hum_12 = np.concatenate([ra_hum_12_c, ra_hum_12_e])
        dec_hum_12 = np.concatenate([dec_hum_12_c, dec_hum_12_e])
        ra_hum_3 = np.concatenate([ra_hum_3_c, ra_hum_3_e])
        dec_hum_3 = np.concatenate([dec_hum_3_c, dec_hum_3_e])
        cluster_class_hum_12 = np.concatenate([cluster_class_hum_12_c, cluster_class_hum_12_e])
        cluster_class_hum_3 = np.concatenate([cluster_class_hum_3_c, cluster_class_hum_3_e])
        age_hum_12 = np.concatenate([age_hum_12_c, age_hum_12_e])
        age_hum_3 = np.concatenate([age_hum_3_c, age_hum_3_e])
    else:
        ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
        ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
        cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
        cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
        age_hum_12 = catalog_access.get_hst_cc_age(target=target)
        age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    pos_mask_dict_hum_12 = catalog_access.get_morph_locations(target=target, ra=ra_hum_12, dec=dec_hum_12)
    pos_mask_dict_hum_3 = catalog_access.get_morph_locations(target=target, ra=ra_hum_3, dec=dec_hum_3)

    class_1_hum = cluster_class_hum_12 == 1
    class_2_hum = cluster_class_hum_12 == 2
    class_3_hum = cluster_class_hum_3 == 3

    class_young_hum_12 = age_hum_12 < 10
    class_inter_hum_12 = (age_hum_12 > 10) & (age_hum_12 < 100)
    class_old_hum_12 = (age_hum_12 > 100)

    class_young_hum_3 = age_hum_3 < 10
    class_inter_hum_3 = (age_hum_3 > 10) & (age_hum_3 < 100)
    class_old_hum_3 = (age_hum_3 > 100)

    mask_ring_hum_12 = pos_mask_dict_hum_12['pos_mask_ring']
    mask_center_hum_12 = pos_mask_dict_hum_12['pos_mask_center'] * ~mask_ring_hum_12
    mask_bar_hum_12 = ((pos_mask_dict_hum_12['pos_mask_bar'] + pos_mask_dict_hum_12['pos_mask_lens']) *
                       ~mask_center_hum_12 * ~mask_ring_hum_12)
    mask_arm_hum_12 = pos_mask_dict_hum_12['pos_mask_arm'] * ~mask_center_hum_12 * ~mask_ring_hum_12 * ~mask_bar_hum_12
    mask_disc_hum_12 = (pos_mask_dict_hum_12['pos_mask_disc'] * ~mask_arm_hum_12 * ~mask_center_hum_12 *
                        ~mask_ring_hum_12 * ~mask_bar_hum_12)
    #
    mask_ring_hum_3 = pos_mask_dict_hum_3['pos_mask_ring']
    mask_center_hum_3 = pos_mask_dict_hum_3['pos_mask_center'] * ~mask_ring_hum_3
    mask_bar_hum_3 = ((pos_mask_dict_hum_3['pos_mask_bar'] + pos_mask_dict_hum_3['pos_mask_lens']) *
                      ~mask_center_hum_3 * ~mask_ring_hum_3)
    mask_arm_hum_3 = pos_mask_dict_hum_3['pos_mask_arm'] * ~mask_center_hum_3 * ~mask_ring_hum_3 * ~mask_bar_hum_3
    mask_disc_hum_3 = (pos_mask_dict_hum_3['pos_mask_disc'] * ~mask_arm_hum_3 * ~mask_center_hum_3 *
                       ~mask_ring_hum_3 * ~mask_bar_hum_3)

    frac_center_hum_1 = np.sum(mask_center_hum_12 * class_1_hum) / np.sum(class_1_hum)
    frac_center_hum_2 = np.sum(mask_center_hum_12 * class_2_hum) / np.sum(class_2_hum)
    frac_center_hum_3 = np.sum(mask_center_hum_3 * class_3_hum) / np.sum(class_3_hum)
    #
    frac_ring_hum_1 = np.sum(mask_ring_hum_12 * class_1_hum) / np.sum(class_1_hum)
    frac_ring_hum_2 = np.sum(mask_ring_hum_12 * class_2_hum) / np.sum(class_2_hum)
    frac_ring_hum_3 = np.sum(mask_ring_hum_3 * class_3_hum) / np.sum(class_3_hum)
    #
    frac_bar_hum_1 = np.sum(mask_bar_hum_12 * class_1_hum) / np.sum(class_1_hum)
    frac_bar_hum_2 = np.sum(mask_bar_hum_12 * class_2_hum) / np.sum(class_2_hum)
    frac_bar_hum_3 = np.sum(mask_bar_hum_3 * class_3_hum) / np.sum(class_3_hum)
    #
    frac_arm_hum_1 = np.sum(mask_arm_hum_12 * class_1_hum) / np.sum(class_1_hum)
    frac_arm_hum_2 = np.sum(mask_arm_hum_12 * class_2_hum) / np.sum(class_2_hum)
    frac_arm_hum_3 = np.sum(mask_arm_hum_3 * class_3_hum) / np.sum(class_3_hum)
    #
    frac_disc_hum_1 = np.sum(mask_disc_hum_12 * class_1_hum) / np.sum(class_1_hum)
    frac_disc_hum_2 = np.sum(mask_disc_hum_12 * class_2_hum) / np.sum(class_2_hum)
    frac_disc_hum_3 = np.sum(mask_disc_hum_3 * class_3_hum) / np.sum(class_3_hum)
    #
    frac_no_class_hum_1 = np.sum(pos_mask_dict_hum_12['pos_not_classified'] * class_1_hum) / np.sum(class_1_hum)
    frac_no_class_hum_2 = np.sum(pos_mask_dict_hum_12['pos_not_classified'] * class_2_hum) / np.sum(class_2_hum)
    frac_no_class_hum_3 = np.sum(pos_mask_dict_hum_3['pos_not_classified'] * class_3_hum) / np.sum(class_3_hum)


    frac_center_young = ((np.sum(class_young_hum_12 * mask_center_hum_12) +
                         np.sum(class_young_hum_3 * mask_center_hum_3)) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))
    frac_ring_young = ((np.sum(class_young_hum_12 * mask_ring_hum_12) +
                         np.sum(class_young_hum_3 * mask_ring_hum_3)) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))
    frac_bar_young = ((np.sum(class_young_hum_12 * mask_bar_hum_12) +
                         np.sum(class_young_hum_3 * mask_bar_hum_3)) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))
    frac_arm_young = ((np.sum(class_young_hum_12 * mask_arm_hum_12) +
                         np.sum(class_young_hum_3 * mask_arm_hum_3)) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))
    frac_disc_young = ((np.sum(class_young_hum_12 * mask_disc_hum_12) +
                         np.sum(class_young_hum_3 * mask_disc_hum_3)) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))
    frac_no_class_young = ((np.sum(class_young_hum_12 * pos_mask_dict_hum_12['pos_not_classified']) +
                         np.sum(class_young_hum_3 * pos_mask_dict_hum_3['pos_not_classified'])) /
                         (np.sum(class_young_hum_12) + np.sum(class_young_hum_3)))

    frac_center_inter = ((np.sum(class_inter_hum_12 * mask_center_hum_12) +
                         np.sum(class_inter_hum_3 * mask_center_hum_3)) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))
    frac_ring_inter = ((np.sum(class_inter_hum_12 * mask_ring_hum_12) +
                         np.sum(class_inter_hum_3 * mask_ring_hum_3)) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))
    frac_bar_inter = ((np.sum(class_inter_hum_12 * mask_bar_hum_12) +
                         np.sum(class_inter_hum_3 * mask_bar_hum_3)) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))
    frac_arm_inter = ((np.sum(class_inter_hum_12 * mask_arm_hum_12) +
                         np.sum(class_inter_hum_3 * mask_arm_hum_3)) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))
    frac_disc_inter = ((np.sum(class_inter_hum_12 * mask_disc_hum_12) +
                         np.sum(class_inter_hum_3 * mask_disc_hum_3)) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))
    frac_no_class_inter = ((np.sum(class_inter_hum_12 * pos_mask_dict_hum_12['pos_not_classified']) +
                         np.sum(class_inter_hum_3 * pos_mask_dict_hum_3['pos_not_classified'])) /
                         (np.sum(class_inter_hum_12) + np.sum(class_inter_hum_3)))

    frac_center_old = ((np.sum(class_old_hum_12 * mask_center_hum_12) +
                         np.sum(class_old_hum_3 * mask_center_hum_3)) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))
    frac_ring_old = ((np.sum(class_old_hum_12 * mask_ring_hum_12) +
                         np.sum(class_old_hum_3 * mask_ring_hum_3)) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))
    frac_bar_old = ((np.sum(class_old_hum_12 * mask_bar_hum_12) +
                         np.sum(class_old_hum_3 * mask_bar_hum_3)) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))
    frac_arm_old = ((np.sum(class_old_hum_12 * mask_arm_hum_12) +
                         np.sum(class_old_hum_3 * mask_arm_hum_3)) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))
    frac_disc_old = ((np.sum(class_old_hum_12 * mask_disc_hum_12) +
                         np.sum(class_old_hum_3 * mask_disc_hum_3)) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))
    frac_no_class_old = ((np.sum(class_old_hum_12 * pos_mask_dict_hum_12['pos_not_classified']) +
                         np.sum(class_old_hum_3 * pos_mask_dict_hum_3['pos_not_classified'])) /
                         (np.sum(class_old_hum_12) + np.sum(class_old_hum_3)))

    morph_map_str = catalog_access.morph_mask_data[target]['morph_map_str']
    classified = morph_map_str != '0000000'
    mask_ring = catalog_access.morph_mask_data[target]['mask_ring']
    mask_center = catalog_access.morph_mask_data[target]['mask_center'] *~mask_ring
    mask_bar = (catalog_access.morph_mask_data[target]['mask_bar'] +
                catalog_access.morph_mask_data[target]['mask_lens']) * ~mask_center * ~mask_ring
    mask_arm = catalog_access.morph_mask_data[target]['mask_arm'] * ~mask_bar * ~mask_center * ~mask_ring
    mask_disc = (catalog_access.morph_mask_data[target]['mask_disc'] + catalog_access.morph_mask_data[target]['mask_in_disc']) * ~mask_arm * ~mask_bar * ~mask_center * ~mask_ring

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


    fig = plt.figure(figsize=(23, 11))
    ax1 = fig.add_axes([-0.16, 0.1, 0.8, 0.8], projection=catalog_access.morph_mask_data[target]['wcs'])
    ax2 = fig.add_axes([0.5, 0.1, 0.49, 0.85])

    ax1.imshow(reproject_dss, cmap=cma_img)

    ax1.contour(mask_disc, levels=0, colors=color_disc, linewidths=3)
    ax1.contour(mask_arm, levels=0, colors=color_arm, linewidths=3)
    ax1.contour(mask_bar, levels=0, colors=color_bar, linewidths=3)
    ax1.contour(mask_ring, levels=0, colors=color_ring, linewidths=3)
    ax1.contour(mask_center, levels=0, colors=color_center, linewidths=3)

    ax1.contour(observed_mask, levels=0, colors=color_hst, linewidths=2)
    pos_12 = SkyCoord(ra=ra_hum_12, dec=dec_hum_12, unit=(u.degree, u.degree), frame='fk5')
    pos_pix_12 = catalog_access.morph_mask_data[target]['wcs'].world_to_pixel(pos_12)
    pos_3 = SkyCoord(ra=ra_hum_3, dec=dec_hum_3, unit=(u.degree, u.degree), frame='fk5')
    pos_pix_3 = catalog_access.morph_mask_data[target]['wcs'].world_to_pixel(pos_3)
    ax1.scatter(pos_pix_12[0][cluster_class_hum_12 == 1], pos_pix_12[1][cluster_class_hum_12 == 1], color=color_c1)
    ax1.scatter(pos_pix_12[0][cluster_class_hum_12 == 2], pos_pix_12[1][cluster_class_hum_12 == 2], color=color_c2)
    ax1.scatter(pos_pix_3[0][cluster_class_hum_3 == 3], pos_pix_3[1][cluster_class_hum_3 == 3], color=color_c3)

    ax1.plot([], [], color=color_disc, label='Disc', linewidth=3)
    ax1.plot([], [], color=color_arm, label='Arm', linewidth=3)
    ax1.plot([], [], color=color_bar, label='Bar', linewidth=3)
    ax1.plot([], [], color=color_ring, label='Ring', linewidth=3)
    ax1.plot([], [], color=color_center, label='Center', linewidth=3)

    ax1.scatter([], [], color=color_c1, label='Class 1')
    ax1.scatter([], [], color=color_c2, label='Class 2')
    ax1.scatter([], [], color=color_c3, label='Class 3')

    photometry_tools.plotting_tools.arr_axis_params(ax=ax1, ra_tick_label=True, dec_tick_label=True,
                        ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                        ra_minpad=0.8, dec_minpad=0.8, tick_color='k',
                        fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
    ax1.legend(frameon=False, fontsize=fontsize)

    ra_length_pix = photometry_tools.analysis_tools.helper_func.transform_world2pix_scale(
        length_in_arcsec=dss_cutout_size[target]/2*60, wcs=catalog_access.morph_mask_data[target]['wcs'], dim=0)
    dec_length_pix = photometry_tools.analysis_tools.helper_func.transform_world2pix_scale(
        length_in_arcsec=dss_cutout_size[target]/2*60, wcs=catalog_access.morph_mask_data[target]['wcs'], dim=1)

    central_pixel_pos = catalog_access.morph_mask_data[target]['wcs'].world_to_pixel(central_coordinates)

    ax1.set_xlim(central_pixel_pos[0] - ra_length_pix/2, central_pixel_pos[0] + ra_length_pix/2)
    ax1.set_ylim(central_pixel_pos[1] - dec_length_pix/2, central_pixel_pos[1] + dec_length_pix/2)

    ax1.set_title(target.upper(), fontsize=fontsize+10)

    ax2.bar(x=1, height=frac_center_all + frac_ring_all + frac_bar_all + frac_arm_all + frac_disc_all, color=color_center)
    ax2.bar(x=1, height=frac_ring_all + frac_bar_all + frac_arm_all + frac_disc_all, color=color_bar)
    ax2.bar(x=1, height=frac_ring_all + frac_arm_all + frac_disc_all, color=color_ring)
    ax2.bar(x=1, height=frac_arm_all + frac_disc_all, color=color_arm)
    ax2.bar(x=1, height=frac_disc_all, color=color_disc)

    ax2.bar(x=2, height=frac_center + frac_ring + frac_bar + frac_arm + frac_disc, color=color_center)
    ax2.bar(x=2, height=frac_ring + frac_bar + frac_arm + frac_disc, color=color_bar)
    ax2.bar(x=2, height=frac_ring + frac_arm + frac_disc, color=color_ring)
    ax2.bar(x=2, height=frac_arm + frac_disc, color=color_arm)
    ax2.bar(x=2, height=frac_disc, color=color_disc)

    ax2.bar(x=3, height=frac_center_hum_1 + frac_ring_hum_1 + frac_bar_hum_1 + frac_arm_hum_1 + frac_disc_hum_1 + frac_no_class_hum_1, color=color_center)
    ax2.bar(x=3, height=frac_ring_hum_1 + frac_bar_hum_1 + frac_arm_hum_1 + frac_disc_hum_1 + frac_no_class_hum_1, color=color_bar)
    ax2.bar(x=3, height=frac_ring_hum_1 + frac_arm_hum_1 + frac_disc_hum_1 + frac_no_class_hum_1, color=color_ring)
    ax2.bar(x=3, height=frac_arm_hum_1 + frac_disc_hum_1 + frac_no_class_hum_1, color=color_arm)
    ax2.bar(x=3, height=frac_disc_hum_1 + frac_no_class_hum_1, color=color_disc)
    ax2.bar(x=3, height=frac_no_class_hum_1, color=color_not_class)

    ax2.bar(x=4, height=frac_center_hum_2 + frac_ring_hum_2 + frac_bar_hum_2 + frac_arm_hum_2 + frac_disc_hum_2 + frac_no_class_hum_2, color=color_center)
    ax2.bar(x=4, height=frac_ring_hum_2 + frac_bar_hum_2 + frac_arm_hum_2 + frac_disc_hum_2 + frac_no_class_hum_2, color=color_bar)
    ax2.bar(x=4, height=frac_ring_hum_2 + frac_arm_hum_2 + frac_disc_hum_2 + frac_no_class_hum_2, color=color_ring)
    ax2.bar(x=4, height=frac_arm_hum_2 + frac_disc_hum_2 + frac_no_class_hum_2, color=color_arm)
    ax2.bar(x=4, height=frac_disc_hum_2 + frac_no_class_hum_2, color=color_disc)
    ax2.bar(x=4, height=frac_no_class_hum_2, color=color_not_class)

    ax2.bar(x=5, height=frac_center_hum_3 + frac_ring_hum_3 + frac_bar_hum_3 + frac_arm_hum_3 + frac_disc_hum_3 + frac_no_class_hum_3, color=color_center)
    ax2.bar(x=5, height=frac_ring_hum_3 + frac_bar_hum_3 + frac_arm_hum_3 + frac_disc_hum_3 + frac_no_class_hum_3, color=color_bar)
    ax2.bar(x=5, height=frac_ring_hum_3 + frac_arm_hum_3 + frac_disc_hum_3 + frac_no_class_hum_3, color=color_ring)
    ax2.bar(x=5, height=frac_arm_hum_3 + frac_disc_hum_3 + frac_no_class_hum_3, color=color_arm)
    ax2.bar(x=5, height=frac_disc_hum_3 + frac_no_class_hum_3, color=color_disc)
    ax2.bar(x=5, height=frac_no_class_hum_3, color=color_not_class)

    ax2.bar(x=6, height=frac_center_young + frac_ring_young + frac_bar_young + frac_arm_young + frac_disc_young + frac_no_class_young, color=color_center)
    ax2.bar(x=6, height=frac_ring_young + frac_bar_young + frac_arm_young + frac_disc_young + frac_no_class_young, color=color_bar)
    ax2.bar(x=6, height=frac_ring_young + frac_arm_young + frac_disc_young + frac_no_class_young, color=color_ring)
    ax2.bar(x=6, height=frac_arm_young + frac_disc_young + frac_no_class_young, color=color_arm)
    ax2.bar(x=6, height=frac_disc_young + frac_no_class_young, color=color_disc)
    ax2.bar(x=6, height=frac_no_class_young, color=color_not_class)

    ax2.bar(x=7, height=frac_center_inter + frac_ring_inter + frac_bar_inter + frac_arm_inter + frac_disc_inter + frac_no_class_inter, color=color_center)
    ax2.bar(x=7, height=frac_ring_inter + frac_bar_inter + frac_arm_inter + frac_disc_inter + frac_no_class_inter, color=color_bar)
    ax2.bar(x=7, height=frac_ring_inter + frac_arm_inter + frac_disc_inter + frac_no_class_inter, color=color_ring)
    ax2.bar(x=7, height=frac_arm_inter + frac_disc_inter + frac_no_class_inter, color=color_arm)
    ax2.bar(x=7, height=frac_disc_inter + frac_no_class_inter, color=color_disc)
    ax2.bar(x=7, height=frac_no_class_inter, color=color_not_class)

    ax2.bar(x=8, height=frac_center_old + frac_ring_old + frac_bar_old + frac_arm_old + frac_disc_old + frac_no_class_old, color=color_center)
    ax2.bar(x=8, height=frac_ring_old + frac_bar_old + frac_arm_old + frac_disc_old + frac_no_class_old, color=color_bar)
    ax2.bar(x=8, height=frac_ring_old + frac_arm_old + frac_disc_old + frac_no_class_old, color=color_ring)
    ax2.bar(x=8, height=frac_arm_old + frac_disc_old + frac_no_class_old, color=color_arm)
    ax2.bar(x=8, height=frac_disc_old + frac_no_class_old, color=color_disc)
    ax2.bar(x=8, height=frac_no_class_old, color=color_not_class)


    ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax2.set_xticklabels(['Galaxy', 'HST footprint', 'Class 1', 'Class 2', 'Class 3', 'young', 'inter', 'old'], rotation=30)
    ax2.set_ylabel('Fraction', fontsize=fontsize)
    ax2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)


    plt.savefig('plot_output/morph_%s.png' % target)


exit()




target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
# catalog_access.load_morph_mask_target_list(target_list=target_list)
# np.save('morph_mask_data.npy', catalog_access.morph_mask_data)
catalog_access.morph_mask_data = np.load('morph_mask_data.npy', allow_pickle=True).item()

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

target = 'ngc0628c'
gal_target = 'ngc0628'

# reproject the mask
reproject_f555w = photometry_tools.plotting_tools.reproject_image(data=phangs_photometry.hst_bands_data['F555W_data_img'],
                                                                  wcs=phangs_photometry.hst_bands_data['F555W_wcs_img'],
                                                                  new_wcs=catalog_access.morph_mask_data[gal_target]['wcs'],
                                                                  new_shape=catalog_access.morph_mask_data[gal_target]['morph_map_float'].shape)
observed_mask = (reproject_f555w != 0) & ~(np.isnan(reproject_f555w))

ra_hum_12_c, dec_hum_12_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c')
ra_hum_3_c, dec_hum_3_c = catalog_access.get_hst_cc_coords_world(target='ngc0628c', cluster_class='class3')
ra_hum_12_e, dec_hum_12_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e')
ra_hum_3_e, dec_hum_3_e = catalog_access.get_hst_cc_coords_world(target='ngc0628e', cluster_class='class3')
cluster_class_hum_12_c = catalog_access.get_hst_cc_class_human(target='ngc0628c')
cluster_class_hum_3_c = catalog_access.get_hst_cc_class_human(target='ngc0628c', cluster_class='class3')
cluster_class_hum_12_e = catalog_access.get_hst_cc_class_human(target='ngc0628e')
cluster_class_hum_3_e = catalog_access.get_hst_cc_class_human(target='ngc0628e', cluster_class='class3')
ra_hum_12 = np.concatenate([ra_hum_12_c, ra_hum_12_e])
dec_hum_12 = np.concatenate([dec_hum_12_c, dec_hum_12_e])
ra_hum_3 = np.concatenate([ra_hum_3_c, ra_hum_3_e])
dec_hum_3 = np.concatenate([dec_hum_3_c, dec_hum_3_e])
cluster_class_hum_12 = np.concatenate([cluster_class_hum_12_c, cluster_class_hum_12_e])
cluster_class_hum_3 = np.concatenate([cluster_class_hum_3_c, cluster_class_hum_3_e])
pos_mask_dict_hum_12 = catalog_access.get_morph_locations(target=gal_target, ra=ra_hum_12, dec=dec_hum_12)
pos_mask_dict_hum_3 = catalog_access.get_morph_locations(target=gal_target, ra=ra_hum_3, dec=dec_hum_3)

mask_arm_hum_1 = (pos_mask_dict_hum_12['pos_mask_arm'] * ~pos_mask_dict_hum_12['pos_mask_bulge'] *
                  (cluster_class_hum_12 == 1))
mask_arm_hum_2 = (pos_mask_dict_hum_12['pos_mask_arm'] * ~pos_mask_dict_hum_12['pos_mask_bulge'] *
                  (cluster_class_hum_12 == 2))
mask_arm_hum_3 = (pos_mask_dict_hum_3['pos_mask_arm'] * ~pos_mask_dict_hum_3['pos_mask_bulge'] *
                  (cluster_class_hum_3 == 3))

mask_disc_hum_1 = (pos_mask_dict_hum_12['pos_mask_disc'] * ~pos_mask_dict_hum_12['pos_mask_bulge'] *
                   ~pos_mask_dict_hum_12['pos_mask_arm'] * (cluster_class_hum_12 == 1))
mask_disc_hum_2 = (pos_mask_dict_hum_12['pos_mask_disc'] * ~pos_mask_dict_hum_12['pos_mask_bulge'] *
                   ~pos_mask_dict_hum_12['pos_mask_arm'] * (cluster_class_hum_12 == 2))
mask_disc_hum_3 = (pos_mask_dict_hum_3['pos_mask_disc'] * ~pos_mask_dict_hum_3['pos_mask_bulge'] *
                   ~pos_mask_dict_hum_3['pos_mask_arm'] * (cluster_class_hum_3 == 3))

mask_bulge_hum_1 = (pos_mask_dict_hum_12['pos_mask_bulge'] * (cluster_class_hum_12 == 1))
mask_bulge_hum_2 = (pos_mask_dict_hum_12['pos_mask_bulge'] * (cluster_class_hum_12 == 2))
mask_bulge_hum_3 = (pos_mask_dict_hum_3['pos_mask_bulge'] * (cluster_class_hum_3 == 3))

frac_arm_hum_1 = np.sum(mask_arm_hum_1) / np.sum(cluster_class_hum_12 == 1)
frac_arm_hum_2 = np.sum(mask_arm_hum_2) / np.sum(cluster_class_hum_12 == 2)
frac_arm_hum_3 = np.sum(mask_arm_hum_3) / np.sum(cluster_class_hum_3 == 3)
frac_disc_hum_1 = np.sum(mask_disc_hum_1) / np.sum(cluster_class_hum_12 == 1)
frac_disc_hum_2 = np.sum(mask_disc_hum_2) / np.sum(cluster_class_hum_12 == 2)
frac_disc_hum_3 = np.sum(mask_disc_hum_3) / np.sum(cluster_class_hum_3 == 3)
frac_bulge_hum_1 = np.sum(mask_bulge_hum_1) / np.sum(cluster_class_hum_12 == 1)
frac_bulge_hum_2 = np.sum(mask_bulge_hum_2) / np.sum(cluster_class_hum_12 == 2)
frac_bulge_hum_3 = np.sum(mask_bulge_hum_3) / np.sum(cluster_class_hum_3 == 3)








morph_map_str = catalog_access.morph_mask_data[gal_target]['morph_map_str']
classified = morph_map_str != '0000000'
mask_arm = catalog_access.morph_mask_data[gal_target]['mask_arm']
mask_bulge = catalog_access.morph_mask_data[gal_target]['mask_bulge']
mask_disc = catalog_access.morph_mask_data[gal_target]['mask_disc']
mask_center = catalog_access.morph_mask_data[gal_target]['mask_center']

total_classified_all = np.sum(classified)
arm_frac_all = np.sum(mask_arm * ~mask_bulge) / total_classified_all
bulge_frac_all = np.sum(mask_bulge) / total_classified_all
disc_frac_all = np.sum(mask_disc * ~mask_bulge * ~mask_arm) / total_classified_all

total_classified = np.sum(classified * observed_mask)
arm_frac = np.sum(mask_arm * ~mask_bulge * observed_mask) / total_classified
bulge_frac = np.sum(mask_bulge * observed_mask) / total_classified
disc_frac = np.sum(mask_disc * ~mask_bulge * ~mask_arm * observed_mask) / total_classified


print('total_classified_all ', total_classified_all)
print('arm_frac_all ', arm_frac_all)
print('bulge_frac_all ', bulge_frac_all)
print('disc_frac_all ', disc_frac_all)
print('total_classified ', total_classified)
print('arm_frac ', arm_frac)
print('bulge_frac ', bulge_frac)
print('disc_frac ', disc_frac)

# load DSS data
dss_hdu = fits.open('dss.01.36.41.7+15.47.01.1.fits')
dss_wcs = WCS(dss_hdu[0].header)
dss_data = dss_hdu[0].data

# reproject the mask
reproject_dss = photometry_tools.plotting_tools.reproject_image(data=dss_data,
                                                                wcs=dss_wcs,
                                                                new_wcs=catalog_access.morph_mask_data[gal_target]['wcs'],
                                                                new_shape=catalog_access.morph_mask_data[gal_target]['morph_map_float'].shape)


fig = plt.figure(figsize=(23, 11))
fontsize = 17
cma_img = 'Greys'
color_disc = 'tab:purple'
color_arm = 'tab:olive'
color_bar = 'tab:cyan'
color_bulge = 'tab:brown'
color_hst = 'black'
color_c1 = 'forestgreen'
color_c2 = 'darkorange'
color_c3 = 'royalblue'

ax1 = fig.add_axes([-0.16, 0.1, 0.8, 0.8], projection=catalog_access.morph_mask_data[gal_target]['wcs'])
ax2 = fig.add_axes([0.5, 0.1, 0.49, 0.85])

ax1.imshow(reproject_dss, cmap=cma_img)
ax1.contour(mask_arm, levels=0, colors=color_arm, linewidths=3)
ax1.contour(mask_bulge, levels=0, colors=color_bulge, linewidths=3)
ax1.contour(mask_disc, levels=0, colors=color_disc, linewidths=3)
ax1.contour(observed_mask, levels=0, colors=color_hst, linewidths=2)
pos_12 = SkyCoord(ra=ra_hum_12, dec=dec_hum_12, unit=(u.degree, u.degree), frame='fk5')
pos_pix_12 = catalog_access.morph_mask_data[gal_target]['wcs'].world_to_pixel(pos_12)
pos_3 = SkyCoord(ra=ra_hum_3, dec=dec_hum_3, unit=(u.degree, u.degree), frame='fk5')
pos_pix_3 = catalog_access.morph_mask_data[gal_target]['wcs'].world_to_pixel(pos_3)
ax1.scatter(pos_pix_12[0][cluster_class_hum_12 == 1], pos_pix_12[1][cluster_class_hum_12 == 1], color=color_c1)
ax1.scatter(pos_pix_12[0][cluster_class_hum_12 == 2], pos_pix_12[1][cluster_class_hum_12 == 2], color=color_c2)
ax1.scatter(pos_pix_3[0][cluster_class_hum_3 == 3], pos_pix_3[1][cluster_class_hum_3 == 3], color=color_c3)
ax1.plot([], [], color=color_disc, label='Disc', linewidth=3)
ax1.plot([], [], color=color_arm, label='Arm', linewidth=3)
ax1.plot([], [], color=color_bulge, label='Bulge', linewidth=3)

ax1.scatter([], [], color=color_c1, label='Class 1')
ax1.scatter([], [], color=color_c2, label='Class 2')
ax1.scatter([], [], color=color_c3, label='Class 3')

photometry_tools.plotting_tools.arr_axis_params(ax=ax1, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='k',
                    fontsize=fontsize, labelsize=fontsize, ra_tick_num=3, dec_tick_num=3)
ax1.legend(frameon=False, fontsize=fontsize)


ra_length_pix = photometry_tools.analysis_tools.helper_func.transform_world2pix_scale(length_in_arcsec=10*60, wcs=catalog_access.morph_mask_data[gal_target]['wcs'], dim=0)
dec_length_pix = photometry_tools.analysis_tools.helper_func.transform_world2pix_scale(length_in_arcsec=10*60, wcs=catalog_access.morph_mask_data[gal_target]['wcs'], dim=1)

central_pixel_pos = catalog_access.morph_mask_data[gal_target]['wcs'].world_to_pixel(SkyCoord(ra=ra_center,
                                                                                              dec=dec_center,
                                                                                              unit=(u.degree, u.degree),
                                                                                              frame='fk5'))

ax1.set_xlim(central_pixel_pos[0] - ra_length_pix/2, central_pixel_pos[0] + ra_length_pix/2)
ax1.set_ylim(central_pixel_pos[1] - dec_length_pix/2, central_pixel_pos[1] + dec_length_pix/2)


ax2.bar(x=1, height=arm_frac_all + bulge_frac_all + disc_frac_all, color=color_bulge)
ax2.bar(x=1, height=arm_frac_all + disc_frac_all, color=color_arm)
ax2.bar(x=1, height=disc_frac_all, color=color_disc)

ax2.bar(x=2, height=arm_frac + bulge_frac + disc_frac, color=color_bulge)
ax2.bar(x=2, height=arm_frac + disc_frac, color=color_arm)
ax2.bar(x=2, height=disc_frac, color=color_disc)

ax2.bar(x=3, height=frac_arm_hum_1 + frac_disc_hum_1 + frac_bulge_hum_1, color=color_bulge)
ax2.bar(x=3, height=frac_arm_hum_1 + frac_disc_hum_1, color=color_arm)
ax2.bar(x=3, height=frac_disc_hum_1, color=color_disc)

ax2.bar(x=4, height=frac_arm_hum_2 + frac_disc_hum_2 + frac_bulge_hum_2, color=color_bulge)
ax2.bar(x=4, height=frac_arm_hum_2 + frac_disc_hum_2, color=color_arm)
ax2.bar(x=4, height=frac_disc_hum_2, color=color_disc)

ax2.bar(x=5, height=frac_arm_hum_3 + frac_disc_hum_3 + frac_bulge_hum_3, color=color_bulge)
ax2.bar(x=5, height=frac_arm_hum_3 + frac_disc_hum_3, color=color_arm)
ax2.bar(x=5, height=frac_disc_hum_3, color=color_disc)

ax2.set_xticks([1, 2, 3, 4, 5])
ax2.set_xticklabels(['Galaxy', 'HST footprint', 'Class 1', 'Class 2', 'Class 3'], rotation=30)
ax2.set_ylabel('Fraction', fontsize=fontsize)
ax2.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)




plt.savefig('plot_output/test_morph.png')





exit()




ax1.imshow((reproject_f555w != 0) & ~(np.isnan(reproject_f555w)))
ax2.imshow(np.log10(catalog_access.morph_mask_data[gal_target]['morph_map_float']))

plt.show()











#
# from reproject import reproject_interp
# array, footprint = reproject_interp(hdu2, hdu1.header)
#
#
# from astropy.wcs import WCS
# import matplotlib.pyplot as plt
#
# ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
# ax1.imshow(array, origin='lower', vmin=-2.e-4, vmax=5.e-4)
# ax1.coords['ra'].set_axislabel('Right Ascension')
# ax1.coords['dec'].set_axislabel('Declination')
# ax1.set_title('Reprojected MSX band E image')
#
# ax2 = plt.subplot(1,2,2, projection=WCS(hdu1.header))
# ax2.imshow(footprint, origin='lower', vmin=0, vmax=1.5)
# ax2.coords['ra'].set_axislabel('Right Ascension')
# ax2.coords['dec'].set_axislabel('Declination')
# ax2.coords['dec'].set_axislabel_position('r')
# ax2.coords['dec'].set_ticklabel_position('r')
# ax2.set_title('MSX band E image footprint')
#




fig = plt.figure()

ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=phangs_photometry.hst_bands_data['F555W_wcs_img'])
ax.imshow(mask_obs)

plt.show()

exit()







total_classified = np.sum(classified)
arm_frac = np.sum(mask_arm * ~mask_bulge) / total_classified
bulge_frac = np.sum(mask_bulge) / total_classified
disc_frac = np.sum(mask_disc * ~mask_bulge * ~mask_arm) / total_classified




print('arm_frac ', arm_frac)
print('bulge_frac ', bulge_frac)
print('disc_frac ', disc_frac)
print('total ', arm_frac + bulge_frac + disc_frac)
exit()

print(np.sum(classified))
print('mask_arm ', np.sum(mask_arm))
print('mask_arm ', np.sum(mask_arm * ~mask_bulge))
print('mask_bulge ', np.sum(mask_bulge))
print('mask_disc ', np.sum(mask_disc))
print('mask_disc ', np.sum(mask_disc * ~mask_bulge))
print('mask_center ', np.sum(mask_center))



fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
ax[0].imshow(mask_disc)
ax[1].imshow(mask_arm)
ax[2].imshow(mask_bulge)
ax[3].imshow(classified)
# ax[0].imshow(mask_disc)

plt.show()

exit()




bar_list = np.zeros(len(dist_list), dtype=bool)
nuc_bar_list = np.zeros(len(dist_list), dtype=bool)
bulge_list = np.zeros(len(dist_list), dtype=bool)
arm_list = np.zeros(len(dist_list), dtype=bool)

for target_index, target in enumerate(target_list):
    if not target[-1].isdigit():
        target = target[:-1]

    bar_list[target_index] = catalog_access.morph_mask_data[target]['presence_bar']
    nuc_bar_list[target_index] = catalog_access.morph_mask_data[target]['presence_nuc_bar']
    bulge_list[target_index] = catalog_access.morph_mask_data[target]['presence_bulge']
    arm_list[target_index] = catalog_access.morph_mask_data[target]['presence_arm']

    print(target_index, target,
          'bar ', catalog_access.morph_mask_data[target]['presence_bar'],
          ' nuc_bar ', catalog_access.morph_mask_data[target]['presence_nuc_bar'],
          ' bulge ', catalog_access.morph_mask_data[target]['presence_bulge'],
          ' arm ', catalog_access.morph_mask_data[target]['presence_arm'])




age_bar_in_hum = np.array([])
age_bar_out_hum = np.array([])
age_arm_in_hum = np.array([])
age_arm_out_hum = np.array([])

age_bar_in_ml = np.array([])
age_bar_out_ml = np.array([])
age_arm_in_ml = np.array([])
age_arm_out_ml = np.array([])


for target_index, target in enumerate(target_list):
    if not target[-1].isdigit():
        gal_target = target[:-1]
    else:
        gal_target = target

    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    mass_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    pos_mask_dict_hum_12 = catalog_access.get_morph_locations(target=gal_target, ra=ra_hum_12, dec=dec_hum_12)
    pos_mask_dict_hum_3 = catalog_access.get_morph_locations(target=gal_target, ra=ra_hum_3, dec=dec_hum_3)

    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    mass_ml_3 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    pos_mask_dict_ml_12 = catalog_access.get_morph_locations(target=gal_target, ra=ra_ml_12, dec=dec_ml_12)
    pos_mask_dict_ml_3 = catalog_access.get_morph_locations(target=gal_target, ra=ra_ml_3, dec=dec_ml_3)

    # bar galaxies
    if bar_list[target_index]:
        mask_bar_in_hum_12 = pos_mask_dict_hum_12['pos_mask_bar'] * ~(pos_mask_dict_hum_12['pos_mask_center'] +
                                                                     pos_mask_dict_hum_12['pos_mask_ring'])
        mask_bar_in_hum_3 = pos_mask_dict_hum_3['pos_mask_bar'] * ~(pos_mask_dict_hum_3['pos_mask_center'] +
                                                                   pos_mask_dict_hum_3['pos_mask_ring'])
        mask_bar_out_hum_12 = pos_mask_dict_hum_12['pos_mask_disc'] * ~pos_mask_dict_hum_12['pos_mask_bar']
        mask_bar_out_hum_3 = pos_mask_dict_hum_3['pos_mask_disc'] * ~pos_mask_dict_hum_3['pos_mask_bar']
        age_bar_in_hum = np.concatenate([age_bar_in_hum, age_hum_12[mask_bar_in_hum_12], age_hum_3[mask_bar_in_hum_3]])
        age_bar_out_hum = np.concatenate([age_bar_out_hum, age_hum_12[mask_bar_out_hum_12], age_hum_3[mask_bar_out_hum_3]])

        mask_bar_in_ml_12 = pos_mask_dict_ml_12['pos_mask_bar'] * ~(pos_mask_dict_ml_12['pos_mask_center'] +
                                                                     pos_mask_dict_ml_12['pos_mask_ring'])
        mask_bar_in_ml_3 = pos_mask_dict_ml_3['pos_mask_bar'] * ~(pos_mask_dict_ml_3['pos_mask_center'] +
                                                                   pos_mask_dict_ml_3['pos_mask_ring'])
        mask_bar_out_ml_12 = pos_mask_dict_ml_12['pos_mask_disc'] * ~pos_mask_dict_ml_12['pos_mask_bar']
        mask_bar_out_ml_3 = pos_mask_dict_ml_3['pos_mask_disc'] * ~pos_mask_dict_ml_3['pos_mask_bar']
        age_bar_in_ml = np.concatenate([age_bar_in_ml, age_ml_12[mask_bar_in_ml_12], age_ml_3[mask_bar_in_ml_3]])
        age_bar_out_ml = np.concatenate([age_bar_out_ml, age_ml_12[mask_bar_out_ml_12], age_ml_3[mask_bar_out_ml_3]])

    # arm galaxies
    if (arm_list[target_index]) * ~(bar_list[target_index]):
        print(target)
        mask_arm_in_hum_12 = pos_mask_dict_hum_12['pos_mask_arm'] * ~(pos_mask_dict_hum_12['pos_mask_center'] +
                                                                      pos_mask_dict_hum_12['pos_mask_ring'] +
                                                                      pos_mask_dict_hum_12['pos_mask_bar'] +
                                                                      pos_mask_dict_hum_12['pos_mask_bulge'])
        mask_arm_in_hum_3 = pos_mask_dict_hum_3['pos_mask_arm'] * ~(pos_mask_dict_hum_3['pos_mask_center'] +
                                                                      pos_mask_dict_hum_3['pos_mask_ring'] +
                                                                      pos_mask_dict_hum_3['pos_mask_bar'] +
                                                                      pos_mask_dict_hum_3['pos_mask_bulge'])
        mask_arm_out_hum_12 = pos_mask_dict_hum_12['pos_mask_disc'] * ~(pos_mask_dict_hum_12['pos_mask_center'] +
                                                                        pos_mask_dict_hum_12['pos_mask_ring'] +
                                                                        pos_mask_dict_hum_12['pos_mask_bar'] +
                                                                        pos_mask_dict_hum_12['pos_mask_bulge'] +
                                                                        pos_mask_dict_hum_12['pos_mask_arm'])
        mask_arm_out_hum_3 = pos_mask_dict_hum_3['pos_mask_disc'] * ~(pos_mask_dict_hum_3['pos_mask_center'] +
                                                                        pos_mask_dict_hum_3['pos_mask_ring'] +
                                                                        pos_mask_dict_hum_3['pos_mask_bar'] +
                                                                        pos_mask_dict_hum_3['pos_mask_bulge'] +
                                                                        pos_mask_dict_hum_3['pos_mask_arm'])
        age_arm_in_hum = np.concatenate([age_arm_in_hum, age_hum_12[mask_arm_in_hum_12], age_hum_3[mask_arm_in_hum_3]])
        age_arm_out_hum = np.concatenate([age_arm_out_hum, age_hum_12[mask_arm_out_hum_12], age_hum_3[mask_arm_out_hum_3]])


        mask_arm_in_ml_12 = pos_mask_dict_ml_12['pos_mask_arm'] * ~(pos_mask_dict_ml_12['pos_mask_center'] +
                                                                      pos_mask_dict_ml_12['pos_mask_ring'] +
                                                                      pos_mask_dict_ml_12['pos_mask_bar'] +
                                                                      pos_mask_dict_ml_12['pos_mask_bulge'])
        mask_arm_in_ml_3 = pos_mask_dict_ml_3['pos_mask_arm'] * ~(pos_mask_dict_ml_3['pos_mask_center'] +
                                                                      pos_mask_dict_ml_3['pos_mask_ring'] +
                                                                      pos_mask_dict_ml_3['pos_mask_bar'] +
                                                                      pos_mask_dict_ml_3['pos_mask_bulge'])
        mask_arm_out_ml_12 = pos_mask_dict_ml_12['pos_mask_disc'] * ~(pos_mask_dict_ml_12['pos_mask_center'] +
                                                                        pos_mask_dict_ml_12['pos_mask_ring'] +
                                                                        pos_mask_dict_ml_12['pos_mask_bar'] +
                                                                        pos_mask_dict_ml_12['pos_mask_bulge'] +
                                                                        pos_mask_dict_ml_12['pos_mask_arm'])
        mask_arm_out_ml_3 = pos_mask_dict_ml_3['pos_mask_disc'] * ~(pos_mask_dict_ml_3['pos_mask_center'] +
                                                                        pos_mask_dict_ml_3['pos_mask_ring'] +
                                                                        pos_mask_dict_ml_3['pos_mask_bar'] +
                                                                        pos_mask_dict_ml_3['pos_mask_bulge'] +
                                                                        pos_mask_dict_ml_3['pos_mask_arm'])
        age_arm_in_ml = np.concatenate([age_arm_in_ml, age_ml_12[mask_arm_in_ml_12], age_ml_3[mask_arm_in_ml_3]])
        age_arm_out_ml = np.concatenate([age_arm_out_ml, age_ml_12[mask_arm_out_ml_12], age_ml_3[mask_arm_out_ml_3]])


df = pd.DataFrame(columns=['x_axis', 'age', 'mass'])

panda_data = pd.DataFrame({'x_axis': ['in_bar_hum'] * len(age_bar_in_hum), 'age': np.log10(age_bar_in_hum) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['in_bar_ml'] * len(age_bar_in_ml), 'age': np.log10(age_bar_in_ml) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['out_bar_hum'] * len(age_bar_out_hum), 'age': np.log10(age_bar_out_hum) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['out_bar_ml'] * len(age_bar_out_ml), 'age': np.log10(age_bar_out_ml) + 6})
df = df.append(panda_data, ignore_index=True)

panda_data = pd.DataFrame({'x_axis': ['in_arm_hum'] * len(age_arm_in_hum), 'age': np.log10(age_arm_in_hum) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['in_arm_ml'] * len(age_arm_in_ml), 'age': np.log10(age_arm_in_ml) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['out_arm_hum'] * len(age_arm_out_hum), 'age': np.log10(age_arm_out_hum) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['out_arm_ml'] * len(age_arm_out_ml), 'age': np.log10(age_arm_out_ml) + 6})
df = df.append(panda_data, ignore_index=True)




fig, ax = plt.subplots(figsize=(10, 10))
sns.violinplot(ax=ax, data=df, x='x_axis', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=15)
plt.xticks(rotation=45)
# ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=13)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/bar_arm_age_1.png', bbox_inches='tight', dpi=300)
# plt.show()







exit()


#
#     ra_12, dec_12 = catalog_access.get_hst_cc_coords_world(target=target)
#     age_12 = catalog_access.get_hst_cc_age(target=target)
#     mass_12 = catalog_access.get_hst_cc_stellar_m(target=target)
#
#     ra_3, dec_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
#     age_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
#     mass_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
#
    # pos_mask_dict_12 = catalog_access.get_morph_locations(target=target, ra=ra_12, dec=dec_12)
#   pos_mask_dict_3 = catalog_access.get_morph_locations(target=target, ra=ra_3, dec=dec_3)
#
#     age_disc_only = np.concatenate([age_disc_only, age_12[pos_mask_dict_12['pos_mask_disc_only']], age_3[pos_mask_dict_3['pos_mask_disc_only']]])
#     age_bar = np.concatenate([age_bar, age_12[pos_mask_dict_12['pos_mask_bar']], age_3[pos_mask_dict_3['pos_mask_bar']]])
#     age_arm = np.concatenate([age_arm, age_12[pos_mask_dict_12['pos_mask_arm']], age_3[pos_mask_dict_3['pos_mask_arm']]])
#     age_ring = np.concatenate([age_ring, age_12[pos_mask_dict_12['pos_mask_ring']], age_3[pos_mask_dict_3['pos_mask_ring']]])
#     age_center = np.concatenate([age_center, age_12[pos_mask_dict_12['pos_mask_center']], age_3[pos_mask_dict_3['pos_mask_center']]])
#     age_lens = np.concatenate([age_lens, age_12[pos_mask_dict_12['pos_mask_lens']], age_3[pos_mask_dict_3['pos_mask_lens']]])
#     age_bulge = np.concatenate([age_bulge, age_12[pos_mask_dict_12['pos_mask_bulge']], age_3[pos_mask_dict_3['pos_mask_bulge']]])
#
#     mass_disc_only = np.concatenate([mass_disc_only, mass_12[pos_mask_dict_12['pos_mask_disc_only']], mass_3[pos_mask_dict_3['pos_mask_disc_only']]])
#     mass_bar = np.concatenate([mass_bar, mass_12[pos_mask_dict_12['pos_mask_bar']], mass_3[pos_mask_dict_3['pos_mask_bar']]])
#     mass_arm = np.concatenate([mass_arm, mass_12[pos_mask_dict_12['pos_mask_arm']], mass_3[pos_mask_dict_3['pos_mask_arm']]])
#     mass_ring = np.concatenate([mass_ring, mass_12[pos_mask_dict_12['pos_mask_ring']], mass_3[pos_mask_dict_3['pos_mask_ring']]])
#     mass_center = np.concatenate([mass_center, mass_12[pos_mask_dict_12['pos_mask_center']], mass_3[pos_mask_dict_3['pos_mask_center']]])
#     mass_lens = np.concatenate([mass_lens, mass_12[pos_mask_dict_12['pos_mask_lens']], mass_3[pos_mask_dict_3['pos_mask_lens']]])
#     mass_bulge = np.concatenate([mass_bulge, mass_12[pos_mask_dict_12['pos_mask_bulge']], mass_3[pos_mask_dict_3['pos_mask_bulge']]])
#
#
#
#
#
# df = pd.DataFrame(columns=['x_axis', 'age', 'mass'])
#
# panda_data = pd.DataFrame({'x_axis': ['disc only'] * len(age_disc_only), 'age': np.log10(age_disc_only) + 6, 'mass': np.log10(mass_disc_only) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['bar'] * len(age_bar), 'age': np.log10(age_bar) + 6, 'mass': np.log10(mass_bar) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['arm'] * len(age_arm), 'age': np.log10(age_arm) + 6, 'mass': np.log10(mass_arm) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['ring'] * len(age_ring), 'age': np.log10(age_ring) + 6, 'mass': np.log10(mass_ring) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['center'] * len(age_center), 'age': np.log10(age_center) + 6, 'mass': np.log10(mass_center) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['lens'] * len(age_lens), 'age': np.log10(age_lens) + 6, 'mass': np.log10(mass_lens) + 6})
# df = df.append(panda_data, ignore_index=True)
# panda_data = pd.DataFrame({'x_axis': ['bulge'] * len(age_bulge), 'age': np.log10(age_bulge) + 6, 'mass': np.log10(mass_bulge) + 6})
# df = df.append(panda_data, ignore_index=True)
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.violinplot(ax=ax, data=df, x='x_axis', y='age', color='skyblue', split=True)
# ax.set_xlabel('')
# ax.set_ylabel('log(Age/yr)', fontsize=15)
# plt.xticks(rotation=45)
# # ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
# ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=13)
# plt.subplots_adjust(wspace=0, hspace=0)
# # plt.savefig('plot_output/violin_age_hum_1.png', bbox_inches='tight', dpi=300)
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.violinplot(ax=ax, data=df, x='x_axis', y='mass', color='skyblue', split=True)
# ax.set_xlabel('')
# ax.set_ylabel('log(Mass)', fontsize=15)
# plt.xticks(rotation=45)
# # ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
# ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=13)
# plt.subplots_adjust(wspace=0, hspace=0)
# # plt.savefig('plot_output/violin_age_hum_1.png', bbox_inches='tight', dpi=300)
# plt.show()
#
# exit()
#
