import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
from astropy.io import fits
import matplotlib.pyplot as plt
from cluster_cat_dr.visualization_tool import PhotVisualize
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle


hst_data_path = '/media/benutzer/Sicherung/data/phangs_hst'
nircam_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
miri_data_path = '/media/benutzer/Sicherung/data/phangs_jwst'
hst_data_ver = 'v1.0'
nircam_data_ver = 'v0p9'
miri_data_ver = 'v0p9'

cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path,
                                                            hst_cc_ver='IR4')

target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


target = 'ngc3627'

# visualization_access = PhotVisualize(
#                                     target_name=target,
#                                     hst_data_path=hst_data_path,
#                                     nircam_data_path=nircam_data_path,
#                                     miri_data_path=miri_data_path,
#                                     hst_data_ver=hst_data_ver,
#                                     nircam_data_ver=nircam_data_ver,
#                                     miri_data_ver=miri_data_ver)
# visualization_access.load_hst_nircam_miri_bands(flux_unit='MJy/sr', load_err=False, band_list=['F555W'])
# img_data = visualization_access.hst_bands_data['F555W_data_img']
# img_wcs = visualization_access.hst_bands_data['F555W_wcs_img']

ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')


number_close_objs = 0
for index in range(len(ra_ml_3)):
    pos_individual = SkyCoord(ra=ra_ml_3[index]*u.deg, dec=dec_ml_3[index]*u.deg)
    pos_all = SkyCoord(ra=ra_ml_3*u.deg, dec=dec_ml_3*u.deg)
    sep = pos_individual.separation(pos_all)
    if sum(sep < 0.2*u.arcsec) > 1:
        number_close_objs += 1

print(number_close_objs)
exit()



# plot figure
figure = plt.figure(figsize=(20, 20))
# parameters for the axis alignment
fontsize = 25
ax = figure.add_axes([0.05, 0.05, 0.9, 0.9], projection=img_wcs)

# get background value as minimum
sigma_clip = SigmaClip()
bkg_estimator = MedianBackground()
bkg = Background2D(img_data, box_size=100, filter_size=21, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
bkg_median = bkg.background_median
img_data -= bkg_median

ax.imshow(img_data, vmin=0, vmax=10, cmap='Greys')
for index in range(len(ra_ml_3)):
    pos = SkyCoord(ra=ra_ml_3[index]*u.deg, dec=dec_ml_3[index]*u.deg)

    visualization_access.plot_circle_on_wcs_img(ax=ax, pos=pos, rad=0.2, color='r')


plt.show()

exit()

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')






exit()
for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        galaxy_name_str = 'ngc0628'
    else:
        galaxy_name_str = target
    inclination = catalog_access.get_target_incl(target=galaxy_name_str)
    print('target ', target, 'dist ', dist, 'inclination ', inclination)
    if 'F438W' in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']:
        b_band = 'F438W'
    else:
        b_band = 'F435W'

    index_hum_12 = catalog_access.get_hst_cc_index(target=target)
    phangs_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target=target)
    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    color_vi_hum_12 = catalog_access.get_hst_color_vi_vega(target=target)
    color_ub_hum_12 = catalog_access.get_hst_color_ub_vega(target=target)
    color_bv_hum_12 = catalog_access.get_hst_color_bv_vega(target=target)
    color_bi_hum_12 = catalog_access.get_hst_color_bi_vega(target=target)
    color_nuvu_hum_12 = catalog_access.get_hst_color_nuvu_vega(target=target)
    color_nuvb_hum_12 = catalog_access.get_hst_color_nuvb_vega(target=target)

    color_vi_err_hum_12 = catalog_access.get_hst_color_vi_err(target=target)
    color_ub_err_hum_12 = catalog_access.get_hst_color_ub_err(target=target)
    color_bv_err_hum_12 = catalog_access.get_hst_color_bv_err(target=target)
    color_bi_err_hum_12 = catalog_access.get_hst_color_bi_err(target=target)
    color_nuvu_err_hum_12 = catalog_access.get_hst_color_nuvu_err(target=target)
    color_nuvb_err_hum_12 = catalog_access.get_hst_color_nuvb_err(target=target)

    detect_nuv_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, band='F275W') >
                          3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F275W')))
    detect_u_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F336W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F336W')))
    detect_b_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band=b_band) >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band=b_band)))
    detect_v_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F555W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F555W')))
    detect_i_hum_12 = ((catalog_access.get_hst_cc_band_flux(target=target, band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, band='F814W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, band='F814W')))

    age_hum_12 = catalog_access.get_hst_cc_age(target=target)
    ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)
    mass_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    ra_hum_12, dec_hum_12 = catalog_access.get_hst_cc_coords_world(target=target)
    x_hum_12, y_hum_12 = catalog_access.get_hst_cc_coords_pix(target=target)


    index_hum_3 = catalog_access.get_hst_cc_index(target=target, cluster_class='class3')
    phangs_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target=target, cluster_class='class3')
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    color_vi_hum_3 = catalog_access.get_hst_color_vi_vega(target=target, cluster_class='class3')
    color_ub_hum_3 = catalog_access.get_hst_color_ub_vega(target=target, cluster_class='class3')
    color_bv_hum_3 = catalog_access.get_hst_color_bv_vega(target=target, cluster_class='class3')
    color_bi_hum_3 = catalog_access.get_hst_color_bi_vega(target=target, cluster_class='class3')
    color_nuvu_hum_3 = catalog_access.get_hst_color_nuvu_vega(target=target, cluster_class='class3')
    color_nuvb_hum_3 = catalog_access.get_hst_color_nuvb_vega(target=target, cluster_class='class3')

    color_vi_err_hum_3 = catalog_access.get_hst_color_vi_err(target=target, cluster_class='class3')
    color_ub_err_hum_3 = catalog_access.get_hst_color_ub_err(target=target, cluster_class='class3')
    color_bv_err_hum_3 = catalog_access.get_hst_color_bv_err(target=target, cluster_class='class3')
    color_bi_err_hum_3 = catalog_access.get_hst_color_bi_err(target=target, cluster_class='class3')
    color_nuvu_err_hum_3 = catalog_access.get_hst_color_nuvu_err(target=target, cluster_class='class3')
    color_nuvb_err_hum_3 = catalog_access.get_hst_color_nuvb_err(target=target, cluster_class='class3')

    detect_nuv_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F275W') >
                          3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F275W')))
    detect_u_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F336W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F336W')))
    detect_b_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band=b_band) >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band=b_band)))
    detect_v_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F555W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F555W')))
    detect_i_hum_3 = ((catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, cluster_class='class3', band='F814W') >
                        3*catalog_access.get_hst_cc_band_flux_err(target=target, cluster_class='class3', band='F814W')))


    age_hum_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    mass_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    ra_hum_3, dec_hum_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    x_hum_3, y_hum_3 = catalog_access.get_hst_cc_coords_pix(target=target, cluster_class='class3')

    color_vi_hum = np.concatenate([color_vi_hum, color_vi_hum_12, color_vi_hum_3])
    color_ub_hum = np.concatenate([color_ub_hum, color_ub_hum_12, color_ub_hum_3])
    color_bv_hum = np.concatenate([color_bv_hum, color_bv_hum_12, color_bv_hum_3])
    color_bi_hum = np.concatenate([color_bi_hum, color_bi_hum_12, color_bi_hum_3])
    color_nuvu_hum = np.concatenate([color_nuvu_hum, color_nuvu_hum_12, color_nuvu_hum_3])
    color_nuvb_hum = np.concatenate([color_nuvb_hum, color_nuvb_hum_12, color_nuvb_hum_3])

    color_vi_err_hum = np.concatenate([color_vi_err_hum, color_vi_err_hum_12, color_vi_err_hum_3])
    color_ub_err_hum = np.concatenate([color_ub_err_hum, color_ub_err_hum_12, color_ub_err_hum_3])
    color_bv_err_hum = np.concatenate([color_bv_err_hum, color_bv_err_hum_12, color_bv_err_hum_3])
    color_bi_err_hum = np.concatenate([color_bi_err_hum, color_bi_err_hum_12, color_bi_err_hum_3])
    color_nuvu_err_hum = np.concatenate([color_nuvu_err_hum, color_nuvu_err_hum_12, color_nuvu_err_hum_3])
    color_nuvb_err_hum = np.concatenate([color_nuvb_err_hum, color_nuvb_err_hum_12, color_nuvb_err_hum_3])

    detect_nuv_hum = np.concatenate([detect_nuv_hum, detect_nuv_hum_12, detect_nuv_hum_3])
    detect_u_hum = np.concatenate([detect_u_hum, detect_u_hum_12, detect_u_hum_3])
    detect_b_hum = np.concatenate([detect_b_hum, detect_b_hum_12, detect_b_hum_3])
    detect_v_hum = np.concatenate([detect_v_hum, detect_v_hum_12, detect_v_hum_3])
    detect_i_hum = np.concatenate([detect_i_hum, detect_i_hum_12, detect_i_hum_3])

    index_hum = np.concatenate([index_hum, index_hum_12, index_hum_3])
    phangs_id_hum = np.concatenate([phangs_id_hum, phangs_id_hum_12, phangs_id_hum_3])
    clcl_color_hum = np.concatenate([clcl_color_hum, cluster_class_hum_12, cluster_class_hum_3])
    target_name_hum_12 = np.array([target]*len(cluster_class_hum_12))
    target_name_hum_3 = np.array([target]*len(cluster_class_hum_3))
    target_name_hum = np.concatenate([target_name_hum, target_name_hum_12, target_name_hum_3])
    incl_hum_12 = np.array([inclination]*len(cluster_class_hum_12))
    incl_hum_3 = np.array([inclination]*len(cluster_class_hum_3))
    incl_hum = np.concatenate([incl_hum, incl_hum_12, incl_hum_3])


    age_hum = np.concatenate([age_hum, age_hum_12, age_hum_3])
    ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])
    mass_hum = np.concatenate([mass_hum, mass_hum_12, mass_hum_3])
    ra_hum = np.concatenate([ra_hum, ra_hum_12, ra_hum_3])
    dec_hum = np.concatenate([dec_hum, dec_hum_12, dec_hum_3])
    x_hum = np.concatenate([x_hum, x_hum_12, x_hum_3])
    y_hum = np.concatenate([y_hum, y_hum_12, y_hum_3])




    index_ml_12= catalog_access.get_hst_cc_index(target=target, classify='ml')
    phangs_id_ml_12= catalog_access.get_hst_cc_phangs_id(target=target, classify='ml')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml')
    color_bv_ml_12 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml')
    color_bi_ml_12 = catalog_access.get_hst_color_bi_vega(target=target, classify='ml')
    color_nuvu_ml_12 = catalog_access.get_hst_color_nuvu_vega(target=target, classify='ml')
    color_nuvb_ml_12 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml')

    color_vi_err_ml_12 = catalog_access.get_hst_color_vi_err(target=target, classify='ml')
    color_ub_err_ml_12 = catalog_access.get_hst_color_ub_err(target=target, classify='ml')
    color_bv_err_ml_12 = catalog_access.get_hst_color_bv_err(target=target, classify='ml')
    color_bi_err_ml_12 = catalog_access.get_hst_color_bi_err(target=target, classify='ml')
    color_nuvu_err_ml_12 = catalog_access.get_hst_color_nuvu_err(target=target, classify='ml')
    color_nuvb_err_ml_12 = catalog_access.get_hst_color_nuvb_err(target=target, classify='ml')

    detect_nuv_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F275W') >
                          catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F275W')))
    detect_u_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F336W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F336W')))
    detect_b_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band=b_band) >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band=b_band)))
    detect_v_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F555W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F555W')))
    detect_i_ml_12 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', band='F814W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', band='F814W')))

    age_ml_12 = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    mass_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    ra_ml_12, dec_ml_12 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml')
    x_ml_12, y_ml_12 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml')

    index_ml_3 = catalog_access.get_hst_cc_index(target=target, classify='ml', cluster_class='class3')
    phangs_id_ml_3 = catalog_access.get_hst_cc_phangs_id(target=target, classify='ml', cluster_class='class3')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi_vega(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub_vega(target=target, classify='ml', cluster_class='class3')
    color_bv_ml_3 = catalog_access.get_hst_color_bv_vega(target=target, classify='ml', cluster_class='class3')
    color_bi_ml_3 = catalog_access.get_hst_color_bi_vega(target=target, classify='ml', cluster_class='class3')
    color_nuvu_ml_3 = catalog_access.get_hst_color_nuvu_vega(target=target, classify='ml', cluster_class='class3')
    color_nuvb_ml_3 = catalog_access.get_hst_color_nuvb_vega(target=target, classify='ml', cluster_class='class3')

    color_vi_err_ml_3 = catalog_access.get_hst_color_vi_err(target=target, classify='ml', cluster_class='class3')
    color_ub_err_ml_3 = catalog_access.get_hst_color_ub_err(target=target, classify='ml', cluster_class='class3')
    color_bv_err_ml_3 = catalog_access.get_hst_color_bv_err(target=target, classify='ml', cluster_class='class3')
    color_bi_err_ml_3 = catalog_access.get_hst_color_bi_err(target=target, classify='ml', cluster_class='class3')
    color_nuvu_err_ml_3 = catalog_access.get_hst_color_nuvu_err(target=target, classify='ml', cluster_class='class3')
    color_nuvb_err_ml_3 = catalog_access.get_hst_color_nuvb_err(target=target, classify='ml', cluster_class='class3')

    detect_nuv_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F275W') > 0) &
                         (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F275W') >
                          catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F275W')))
    detect_u_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F336W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F336W')))
    detect_b_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band=b_band) >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band=b_band)))
    detect_v_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F555W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F555W')))
    detect_i_ml_3 = ((catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') > 0) &
                       (catalog_access.get_hst_cc_band_flux(target=target, classify='ml', cluster_class='class3', band='F814W') >
                        catalog_access.get_hst_cc_band_flux_err(target=target, classify='ml', cluster_class='class3', band='F814W')))

    age_ml_3 = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    mass_ml_3 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
    ra_ml_3, dec_ml_3 = catalog_access.get_hst_cc_coords_world(target=target, classify='ml', cluster_class='class3')
    x_ml_3, y_ml_3 = catalog_access.get_hst_cc_coords_pix(target=target, classify='ml', cluster_class='class3')

    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    color_bv_ml = np.concatenate([color_bv_ml, color_bv_ml_12, color_bv_ml_3])
    color_bi_ml = np.concatenate([color_bi_ml, color_bi_ml_12, color_bi_ml_3])
    color_nuvu_ml = np.concatenate([color_nuvu_ml, color_nuvu_ml_12, color_nuvu_ml_3])
    color_nuvb_ml = np.concatenate([color_nuvb_ml, color_nuvb_ml_12, color_nuvb_ml_3])

    color_vi_err_ml = np.concatenate([color_vi_err_ml, color_vi_err_ml_12, color_vi_err_ml_3])
    color_ub_err_ml = np.concatenate([color_ub_err_ml, color_ub_err_ml_12, color_ub_err_ml_3])
    color_bv_err_ml = np.concatenate([color_bv_err_ml, color_bv_err_ml_12, color_bv_err_ml_3])
    color_bi_err_ml = np.concatenate([color_bi_err_ml, color_bi_err_ml_12, color_bi_err_ml_3])
    color_nuvu_err_ml = np.concatenate([color_nuvu_err_ml, color_nuvu_err_ml_12, color_nuvu_err_ml_3])
    color_nuvb_err_ml = np.concatenate([color_nuvb_err_ml, color_nuvb_err_ml_12, color_nuvb_err_ml_3])

    detect_nuv_ml = np.concatenate([detect_nuv_ml, detect_nuv_ml_12, detect_nuv_ml_3])
    detect_u_ml = np.concatenate([detect_u_ml, detect_u_ml_12, detect_u_ml_3])
    detect_b_ml = np.concatenate([detect_b_ml, detect_b_ml_12, detect_b_ml_3])
    detect_v_ml = np.concatenate([detect_v_ml, detect_v_ml_12, detect_v_ml_3])
    detect_i_ml = np.concatenate([detect_i_ml, detect_i_ml_12, detect_i_ml_3])

    index_ml = np.concatenate([index_ml, index_ml_12, index_ml_3])
    phangs_id_ml = np.concatenate([phangs_id_ml, phangs_id_ml_12, phangs_id_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])
    target_name_ml_12 = np.array([target]*len(cluster_class_ml_12))
    target_name_ml_3 = np.array([target]*len(cluster_class_ml_3))
    target_name_ml = np.concatenate([target_name_ml, target_name_ml_12, target_name_ml_3])
    incl_ml_12 = np.array([inclination]*len(cluster_class_ml_12))
    incl_ml_3 = np.array([inclination]*len(cluster_class_ml_3))
    incl_ml = np.concatenate([incl_ml, incl_ml_12, incl_ml_3])

    age_ml = np.concatenate([age_ml, age_ml_12, age_ml_3])
    ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])
    mass_ml = np.concatenate([mass_ml, mass_ml_12, mass_ml_3])
    ra_ml = np.concatenate([ra_ml, ra_ml_12, ra_ml_3])
    dec_ml = np.concatenate([dec_ml, dec_ml_12, dec_ml_3])
    x_ml = np.concatenate([x_ml, x_ml_12, x_ml_3])
    y_ml = np.concatenate([y_ml, y_ml_12, y_ml_3])


    print('max ebv', np.max(np.concatenate([ebv_ml_12, ebv_ml_3])))

    # now get magnitude cuts
    v_mag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    v_mag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_v_mag_hum_12 = hf.conv_mag2abs_mag(mag=v_mag_hum_12, dist=dist)
    abs_v_mag_hum_3 = hf.conv_mag2abs_mag(mag=v_mag_hum_3, dist=dist)
    v_mag_hum = np.concatenate([v_mag_hum_12, v_mag_hum_3])

    v_mag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml')
    v_mag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', classify='ml', cluster_class='class3')
    abs_v_mag_ml_12 = hf.conv_mag2abs_mag(mag=v_mag_ml_12, dist=dist)
    abs_v_mag_ml_3 = hf.conv_mag2abs_mag(mag=v_mag_ml_3, dist=dist)
    v_mag_ml = np.concatenate([v_mag_ml_12, v_mag_ml_3])

    max_hum_mag = np.nanmax(v_mag_hum)
    selected_mag_ml = v_mag_ml < max_hum_mag

    mag_mask_ml = np.concatenate([mag_mask_ml, selected_mag_ml])
