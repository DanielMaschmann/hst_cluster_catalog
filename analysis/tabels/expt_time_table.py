import numpy as np
import photometry_tools
from cluster_cat_dr.phot_data_access import PhotAccess
from astropy.io import fits

"""
This macro will produce Table 1 for the PHANGS HST paper
"""


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            #hst_cc_ver='IR4'
                                                            )


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


prop_id_dict = {'ic1954': 11111, 'ic5332': 11111, 'ngc0628e': 11111, 'ngc0628c': 11111, 'ngc0685': 11111,
                'ngc1087': 11111, 'ngc1097': 11111, 'ngc1300': 11111, 'ngc1317': 11111, 'ngc1365': 11111,
                'ngc1385': 11111, 'ngc1433': 11111, 'ngc1512': 11111, 'ngc1559': 11111, 'ngc1566': 11111,
                'ngc1672': 11111, 'ngc1792': 11111, 'ngc2775': 11111, 'ngc2835': 11111, 'ngc2903': 11111,
                'ngc3351': 11111, 'ngc3621': 11111, 'ngc3627': 11111, 'ngc4254': 11111, 'ngc4298': 11111,
                'ngc4303': 11111, 'ngc4321': 11111, 'ngc4535': 11111, 'ngc4536': 11111, 'ngc4548': 11111,
                'ngc4569': 11111, 'ngc4571': 11111, 'ngc4654': 11111, 'ngc4689': 11111, 'ngc4826': 11111,
                'ngc5068': 11111, 'ngc5248': 11111, 'ngc6744': 11111, 'ngc7496': 11111}

n_pointing_dict = {'ic1954': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ic5332': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc0628e': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc0628c': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc0685': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1087': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1097': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1300': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1317': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1365': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1385': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1433': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1512': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1559': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1566': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1672': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc1792': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc2775': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc2835': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc2903': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc3351': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc3621': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc3627': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4254': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4298': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4303': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4321': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4535': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4536': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4548': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4569': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4571': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4654': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4689': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc4826': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc5068': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc5248': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc6744': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2},
                   'ngc7496': {'NUV': 2, 'U': 2, 'B': 2, 'V': 2, 'I': 2}}


print("")
print('\multicolumn{1}{c}{Galaxy} & '
      '\multicolumn{1}{c}{PID} & '
      '\multicolumn{2}{c}{F275W} & '
      '\multicolumn{2}{c}{F336W} & '
      '\multicolumn{3}{c}{F435W} & '
      '\multicolumn{3}{c}{F438W} & '
      '\multicolumn{3}{c}{F555W} & '
      '\multicolumn{3}{c}{F814W} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '      
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} & ' 
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '      
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} & '
      '\multicolumn{1}{c}{Det} & '
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '    
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} & '      
      '\multicolumn{1}{c}{Det} & '
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} & '      
      '\multicolumn{1}{c}{Det} & '
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} & '
      '\multicolumn{1}{c}{Det} & '
      '\multicolumn{1}{c}{${n_{\\rm p}}$} & '
      '\multicolumn{1}{c}{${t_{\\rm exp}}$} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} \\\\ ')
print('\\hline')

for index in range(0, len(target_list)):
    target = target_list[index]

    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_string = target_list[index][0:3] + '\,' +  target_list[index][4:]
    elif target_list[index][0:2] == 'ic':
        target_string = target_list[index][0:2] + '\,' +  target_list[index][2:]
    elif target_list[index][0:3] == 'ngc':
        target_string = target_list[index][0:3] + '\,' +  target_list[index][3:]
    else:
        target_string = target_list[index]
    target_string = target_string.upper()

    line_string = ''
    pid_list = []
    phot_access = PhotAccess(target_name=str(target), hst_data_path='/media/benutzer/Sicherung/data/phangs_hst')

    for band in ['F275W', 'F336W', 'F435W', 'F438W', 'F555W', 'F814W']:
        if ((band in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']) |
                (band in catalog_access.hst_targets[target]['acs_wfc1_observed_bands'])):
            detector = catalog_access.get_hst_band_detector(target=target, band=band)
            exp_time = catalog_access.get_hst_exp_time(target=target, band=band)
            n_pointing = str(int(catalog_access.get_hst_n_pointig(target=target, band=band)))
            if band in ['F275W', 'F336W']:
                line_string += n_pointing + ' & ' + str(int(exp_time[0]))
            else:
                line_string += str(detector[0]) + ' & ' + n_pointing + ' & ' + str(int(exp_time[0]))

            hst_file_name = phot_access.get_hst_img_file_name(band=band)
            hdu = fits.open(hst_file_name)
            pid = hdu[0].header['PROPOSID']
            pid_list.append(pid)
            hdu.close()

        else:
            line_string += '--' + ' & ' + '--' + ' & ' + '--'
        if band != 'F814W':
            line_string += ' & '
        else:
            line_string += '\\\\'

    unique_pid = np.unique(pid_list)
    sort = np.argsort(unique_pid)
    unique_pid = unique_pid[sort]
    string_pid = ''
    for id in unique_pid:
        string_pid += str(id) + ', '
    string_pid = string_pid[:-2]
    target_string += ' & ' + string_pid

    line_string = target_string + ' & ' + line_string

    print(line_string)

exit()

