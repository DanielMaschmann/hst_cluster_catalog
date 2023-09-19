import numpy as np
import photometry_tools
"""
This macro will produce Table 1 for the PHANGS HST paper
"""

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
dist_list = []
print(target_list)
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(target_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_sample_table()



# for index in range(2):
for index in range(len(target_list)):
    target = target_list[index]
    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_name_str = target_list[index][0:3] + ' ' +  target_list[index][4:]
    elif target_list[index][0:2] == 'ic':
        target_name_str = target_list[index][0:2] + ' ' +  target_list[index][2:]
    elif target_list[index][0:3] == 'ngc':
        target_name_str = target_list[index][0:3] + ' ' +  target_list[index][3:]
    else:
        target_name_str = target_list[index]
    target_name_str = target_name_str.upper()

    if (target == 'ngc0628e') | (target == 'ngc0628c'):
        target_str = 'ngc0628'
    else:
        target_str = target

    veron = catalog_access.get_target_veron_class(target=target_str)
    milliquas = catalog_access.get_target_milliquas_class(target=target_str)

    print(target_name_str, ' ', veron, ' ', milliquas)


