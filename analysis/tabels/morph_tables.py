import numpy as np
import photometry_tools
from cluster_cat_dr.phot_data_access import PhotAccess
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")




cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'

catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            sample_table_path=sample_table_path
                                                            )


target_list = catalog_access.phangs_galaxy_list

list_delta_ms = np.zeros(len(target_list))
list_m_star = np.zeros(len(target_list))

for index, target in enumerate(target_list):
    list_delta_ms[index] = catalog_access.get_target_delta_ms(target=target)
    list_m_star[index] = catalog_access.get_target_mstar(target=target)

sort = np.argsort(list_delta_ms)[::-1]
target_list = np.array(target_list)[sort]
list_delta_ms = np.array(list_delta_ms)[sort]
list_m_star = np.array(list_m_star)[sort]


# morph_dict = {
#     'ngc1365': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc1672': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc4303': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc7496': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#
#     'ngc1385': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc1559': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc4536': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc4254': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#
#     'ngc4654': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc1087': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc1097': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc1792': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},
#
#     'ngc1566': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
#     'ngc2835': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc5248': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc2903': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
#
#     'ngc4321': {'sf_bar': False, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc3627': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
#     'ngc0628': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
#     'ngc4535': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#
#     'ngc3621': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
#     'ngc6744': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
#     'ngc3351': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
#     'ngc5068': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#
#     'ic5332': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ic1954': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
#     'ngc4298': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc1300': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
#
#     'ngc1512': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
#     'ngc0685': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc4569': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': True},
#     'ngc1433': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#
#     'ngc4689': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc4571': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc1317': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
#     'ngc4548': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},
#
#     'ngc2775': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': True},
#     'ngc4826': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': True},
# }


morph_dict = {
    'ngc1365': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1672': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4303': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc7496': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc1385': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1559': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4536': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4254': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc4654': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1087': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1097': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc1792': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc1566': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc2835': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc5248': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc2903': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc4321': {'sf_bar': False, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc3627': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc0628': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},
    'ngc4535': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},

    'ngc3621': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc6744': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc3351': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc5068': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},

    'ic5332': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ic1954': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': True, 'bulge': False, 'flocc': False, 'quiescent': False},
    'ngc4298': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1300': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': True, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc1512': {'sf_bar': True, 'cent_ring': True, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': False},
    'ngc0685': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4569': {'sf_bar': True, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': False, 'flocc': False, 'quiescent': True},
    'ngc1433': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},

    'ngc4689': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4571': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc1317': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': False, 'flocc': True, 'quiescent': False},
    'ngc4548': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': True, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': False},

    'ngc2775': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': True, 'quiescent': True},
    'ngc4826': {'sf_bar': False, 'cent_ring': False, 'sf_end_bars': False, 'glob_arms': False, 'bulge': True, 'flocc': False, 'quiescent': True},
}



print("")
print('\\multicolumn{1}{c}{$\\Delta$MS} & '
      '\\multicolumn{1}{c}{log M$_*$} & '
      '\\multicolumn{7}{c}{Morphological features} \\\\ ')
print('\\hline')
print('\\multicolumn{1}{c}{[dex]} & '
      '\\multicolumn{1}{c}{[M$_{\\odot}$]} & '
      '\\multicolumn{1}{c}{Long SF Bars} & '
      '\\multicolumn{1}{c}{Central-Ring} & ' 
      '\\multicolumn{1}{c}{SF-end-of-Bar} & ' 
      '\\multicolumn{1}{c}{Global-Arms} & ' 
      '\\multicolumn{1}{c}{Bulges} & ' 
      '\\multicolumn{1}{c}{Flocculant} & ' 
      '\\multicolumn{1}{c}{Quiescent} \\\\ ')
print('\\hline')

for index in range(0, len(target_list)):
    target = target_list[index]
    delta_ms = list_delta_ms[index]
    log_m_star = np.log10(list_m_star[index])
    target_morph_dict = morph_dict[target]

    if target[3] == '0':
        target_name = target[:3] + target[4:]
    else:
        target_name = target

    if target_name[0] == 'n':
        target_name_str = 'N' + target_name[3:]
    if target_name[0] == 'i':
        target_name_str = 'I' + target_name[2:]

    line_string = '%.2f & %.2f & ' % (delta_ms, log_m_star)
    if target_morph_dict['sf_bar']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['cent_ring']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['sf_end_bars']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['glob_arms']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['bulge']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['flocc']:
        line_string += target_name_str + ' & '
    else:
        line_string += ' & '

    if target_morph_dict['quiescent']:
        line_string += target_name_str + '\\\\'
    else:
        line_string += '\\\\'

    print(line_string)

exit()
