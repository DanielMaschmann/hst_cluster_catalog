import numpy as np
import photometry_tools
"""
This macro will produce Table 1 for the PHANGS HST paper
"""


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            hst_cc_ver='IR4')


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


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')

catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')

catalog_access.load_hst_cc_list(target_list=target_list, classify='', cluster_class='candidates')


n_candidates = []
n_insp = []
n_hum_1 = []
n_hum_2 = []
n_hum_3 = []
n_ml_1 = []
n_ml_2 = []
n_ml_3 = []

median_abs_vmag_hum = []
median_abs_vmag_ml = []

min_abs_vmag_hum = []
min_abs_vmag_ml = []

max_abs_vmag_hum = []
max_abs_vmag_ml = []

abs_vmag_array_hum = np.array([])
abs_vmag_array_ml = np.array([])







print("")
print('\multicolumn{1}{c}{Galaxy} & '
      '\multicolumn{2}{c}{F275W} & '
      '\multicolumn{2}{c}{F336W} & '
      '\multicolumn{2}{c}{F435W} & '
      '\multicolumn{2}{c}{F438W} & '
      '\multicolumn{2}{c}{F555W} & '
      '\multicolumn{2}{c}{F814W} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} & '      
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} & '
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} & '
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} & '      
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} & '
      '\multicolumn{1}{c}{Detector} & '
      '\multicolumn{1}{c}{${\\rm t_{exp}}$} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{[s]} & '
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

    line_string = target_string + ' & '
    for band in ['F275W', 'F336W', 'F435W', 'F438W', 'F555W', 'F814W']:
        if ((band in catalog_access.hst_targets[target]['wfc3_uvis_observed_bands']) |
                (band in catalog_access.hst_targets[target]['acs_wfc1_observed_bands'])):
            detector = catalog_access.get_hst_band_detector(target=target, band=band)
            exp_time = catalog_access.get_hst_exp_time(target=target, band=band)
            line_string += str(detector[0]) + ' & ' + str(exp_time[0])
        else:
            line_string += '--' + ' & ' + '--'
        if band != 'F814W':
            line_string += ' & '
        else:
            line_string += '\\\\'
    print(line_string)

exit()




for index in range(0, len(target_list)):

    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_string = target_list[index][0:3] + target_list[index][4:]
    else:
        target_string = target_list[index]
    target_string = target_string.upper()

    print(
        '%s & '
        '%i & '
        '%i & '

        '%i & '
        '%i & '
        '%i & '
        '%i & '

        '%i & '
        '%i & '
        '%i & '
        '%i & '

        '%.1f$\\vert$ %.1f$\\vert$ %.1f & '

        '%.1f$\\vert$ %.1f$\\vert$ %.1f \\\\ '
        % (target_string,
           n_candidates[index],
           n_insp[index],
           n_hum_1[index],
           n_hum_2[index],
           n_hum_3[index],
           n_hum_1[index] + n_hum_2[index] + n_hum_3[index],

           n_ml_1[index],
           n_ml_2[index],
           n_ml_3[index],
           n_ml_1[index] + n_ml_2[index] + n_ml_3[index],

           min_abs_vmag_hum[index],
           median_abs_vmag_hum[index],
           max_abs_vmag_hum[index],

           min_abs_vmag_ml[index],
           median_abs_vmag_ml[index],
           max_abs_vmag_ml[index],
        )
    )
print('\\hline')

print(
    'Median & '
    '%i & '
    '%i & '
    
    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '%.1f$\\vert$ %.1f$\\vert$ %.1f & '

    '%.1f$\\vert$ %.1f$\\vert$ %.1f \\\\ '
    % (
        np.median(n_candidates),
        np.median(n_insp),

        np.median(n_hum_1),
        np.median(n_hum_2),
        np.median(n_hum_3),
        np.median(np.array(n_hum_1) + np.array(n_hum_2) + np.array(n_hum_3)),
        np.median(n_ml_1),
        np.median(n_ml_2),
        np.median(n_ml_3),
        np.median(np.array(n_ml_1) + np.array(n_ml_2) + np.array(n_ml_3)),
        np.nanmin(abs_vmag_array_hum),
        np.nanmedian(abs_vmag_array_hum),
        np.nanmax(abs_vmag_array_hum),
        np.nanmin(abs_vmag_array_ml),
        np.nanmedian(abs_vmag_array_ml),
        np.nanmax(abs_vmag_array_ml)

    )
)

print(
    'Mean & '
    '%i & '
    '%i & '
    
    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '- & '

    '- \\\\ '
    % (
        np.mean(n_candidates),
        np.mean(n_insp),

        np.mean(n_hum_1),
        np.mean(n_hum_2),
        np.mean(n_hum_3),
        np.mean(np.array(n_hum_1) + np.array(n_hum_2) + np.array(n_hum_3)),
        np.mean(n_ml_1),
        np.mean(n_ml_2),
        np.mean(n_ml_3),
        np.mean(np.array(n_ml_1) + np.array(n_ml_2) + np.array(n_ml_3)),

    )
)

print(
    'Total & '
    '%i & '
    '%i & '
    
    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '%i & '
    '%i & '
    '%i & '
    '%i & '

    '- & '

    '- \\\\ '
    % (
        np.sum(n_candidates),
        np.sum(n_insp),

        np.sum(n_hum_1),
        np.sum(n_hum_2),
        np.sum(n_hum_3),
        np.sum(n_hum_1) + np.sum(n_hum_2) + np.sum(n_hum_3),
        np.sum(n_ml_1),
        np.sum(n_ml_2),
        np.sum(n_ml_3),
        np.sum(n_ml_1) + np.sum(n_ml_2) + np.sum(n_ml_3),
        # np.nanmin(abs_vmag_array_hum),
        # np.nanmedian(abs_vmag_array_hum),
        # np.nanmax(abs_vmag_array_hum),
        # np.nanmin(abs_vmag_array_ml),
        # np.nanmedian(abs_vmag_array_ml),
        # np.nanmax(abs_vmag_array_ml)
    )
)
print('\\hline')
