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


# # get access to HST cluster catalog
# cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
# hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
# morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
# catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
#                                                             hst_obs_hdr_file_path=hst_obs_hdr_file_path,
#                                                             morph_mask_path=morph_mask_path)
# target_list = catalog_access.target_hst_cc
# dist_list = []
# for target in target_list:
#     if (target == 'ngc0628c') | (target == 'ngc0628e'):
#         target = 'ngc0628'
#     dist_list.append(catalog_access.dist_dict[target]['dist'])
# # sort target list after distance
# sort = np.argsort(dist_list)
# target_list = np.array(target_list)[sort]
# dist_list = np.array(dist_list)[sort]
#
# catalog_access.load_hst_cc_list(target_list=target_list)
# catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
# catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


#print(catalog_access.hst_cc_data['ngc0628c_human_class12'].names)

candidates = fits.open('/home/benutzer/data/PHANGS_products/HST_catalog_2/IR4/InteractiveDisplay/ngc628c_phangshst_candidates_bcw_v1p2_IR4.fits')

print(candidates.info())

print(candidates[1].data.names)

exit()


mci_in_hum_12 = catalog_access.get_hst_cc_mci_in(target='ngc7496')
mci_out_hum_12 = catalog_access.get_hst_cc_mci_out(target='ngc0628c')

plt.scatter(mci_in_hum_12, mci_out_hum_12)

plt.show()



