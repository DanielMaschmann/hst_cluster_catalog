import numpy as np
import photometry_tools
"""
This macro will produce Table 1 for the PHANGS HST paper
"""


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            # hst_cc_ver='IR4'
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


for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_candidates = catalog_access.get_hst_cc_class_human(target=target, classify='', cluster_class='candidates')
    number_candidates = len(catalog_access.hst_cc_data[target + '_' + '_' + 'candidates'])
    # print(cluster_class_candidates)
    # print(np.isnan(cluster_class_candidates))
    print(np.ma.count(cluster_class_candidates))

    # number_inspected = sum(np.invert(np.isnan(cluster_class_candidates)))
    number_inspected = np.ma.count(cluster_class_candidates)


    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')

    number_hum_1 = np.sum(cluster_class_hum_12 == 1)
    number_hum_2 = np.sum(cluster_class_hum_12 == 2)
    number_hum_3 = np.sum(cluster_class_hum_3 == 3)
    number_ml_1 = np.sum(cluster_class_ml_12 == 1)
    number_ml_2 = np.sum(cluster_class_ml_12 == 2)
    number_ml_3 = np.sum(cluster_class_ml_3 == 3)

    n_candidates.append(number_candidates)
    n_insp.append(number_inspected)
    n_hum_1.append(number_hum_1)
    n_hum_2.append(number_hum_2)
    n_hum_3.append(number_hum_3)
    n_ml_1.append(number_ml_1)
    n_ml_2.append(number_ml_2)
    n_ml_3.append(number_ml_3)

    vmag_hum_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W')
    vmag_hum_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, band='F555W', cluster_class='class3')
    abs_vmag_hum_12 = photometry_tools.analysis_tools.helper_func.conv_mag2abs_mag(mag=vmag_hum_12, dist=dist)
    abs_vmag_hum_3 = photometry_tools.analysis_tools.helper_func.conv_mag2abs_mag(mag=vmag_hum_3, dist=dist)
    abs_vmag_hum = np.concatenate([abs_vmag_hum_12, abs_vmag_hum_3])

    vmag_ml_12 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W')
    vmag_ml_3 = catalog_access.get_hst_cc_band_vega_mag(target=target, classify='ml', band='F555W', cluster_class='class3')
    abs_vmag_ml_12 = photometry_tools.analysis_tools.helper_func.conv_mag2abs_mag(mag=vmag_ml_12, dist=dist)
    abs_vmag_ml_3 = photometry_tools.analysis_tools.helper_func.conv_mag2abs_mag(mag=vmag_ml_3, dist=dist)
    abs_vmag_ml = np.concatenate([abs_vmag_ml_12, abs_vmag_ml_3])

    abs_vmag_array_hum = np.concatenate([abs_vmag_array_hum, abs_vmag_hum])
    abs_vmag_array_ml = np.concatenate([abs_vmag_array_ml, abs_vmag_ml])

    median_abs_vmag_hum.append(np.nanmedian(abs_vmag_hum))
    median_abs_vmag_ml.append(np.nanmedian(abs_vmag_ml))

    min_abs_vmag_hum.append(np.nanmin(abs_vmag_hum))
    min_abs_vmag_ml.append(np.nanmin(abs_vmag_ml))

    max_abs_vmag_hum.append(np.nanmax(abs_vmag_hum))
    max_abs_vmag_ml.append(np.nanmax(abs_vmag_ml))




print("")
print('\multicolumn{1}{c}{Galaxy} & '
      '\multicolumn{2}{c}{Candidates} & '
      '\multicolumn{4}{c}{Human-classified} & '
      '\multicolumn{4}{c}{ML-classified} & '
      
      '\multicolumn{1}{c}{$M_V^{\\rm Hum}$} & '
      '\multicolumn{1}{c}{$M_V^{\\rm ML}$} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{N$_{\\rm Cand}$} & '
      '\multicolumn{1}{c}{N$_{\\rm Insp}$} & '
      
      '\multicolumn{1}{c}{C1} & '
      '\multicolumn{1}{c}{C2} & '
      '\multicolumn{1}{c}{C3} & '
      '\multicolumn{1}{c}{C1+2+3} & '
      
      '\multicolumn{1}{c}{C1} & '
      '\multicolumn{1}{c}{C2} & '
      '\multicolumn{1}{c}{C3} & '
      '\multicolumn{1}{c}{C1+2+3} & '
      
      '\multicolumn{1}{c}{min$\\vert$med$\\vert$max} & '      
      '\multicolumn{1}{c}{min$\\vert$med$\\vert$max} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{4}{c}{} & '
      '\multicolumn{4}{c}{} & '
      
      '\multicolumn{1}{c}{mag} & '
      
      '\multicolumn{1}{c}{mag} \\\\ ')
print('\\hline')

for index in range(0, len(target_list)):

    if (target_list[index][0:3] == 'ngc') & (target_list[index][3] == '0'):
        target_string = target_list[index][0:3] + '\,' +  target_list[index][4:]
    elif target_list[index][0:2] == 'ic':
        target_string = target_list[index][0:2] + '\,' +  target_list[index][2:]
    elif target_list[index][0:3] == 'ngc':
        target_string = target_list[index][0:3] + '\,' +  target_list[index][3:]
    else:
        target_string = target_list[index]
    target_string = target_string.upper()

    # print(target_string,
    #       n_candidates[index],
    #       n_insp[index],
    #       n_hum_1[index],
    #       n_hum_2[index],
    #       n_hum_3[index],
    #       n_hum_1[index] + n_hum_2[index] + n_hum_3[index],
    #
    #       n_ml_1[index],
    #       n_ml_2[index],
    #       n_ml_3[index],
    #       n_ml_1[index] + n_ml_2[index] + n_ml_3[index],
    #
    #       min_abs_vmag_hum[index],
    #       median_abs_vmag_hum[index],
    #       max_abs_vmag_hum[index],
    #
    #        min_abs_vmag_ml[index],
    #        median_abs_vmag_ml[index],
    #        max_abs_vmag_ml[index]
    # )




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

        '%.1f$\\vert$%.1f$\\vert$%.1f & '

        '%.1f$\\vert$%.1f$\\vert$%.1f \\\\ '
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
        np.nanmedian(min_abs_vmag_hum),
        np.nanmedian(abs_vmag_array_hum),
        np.nanmedian(max_abs_vmag_hum),
        np.nanmedian(min_abs_vmag_ml),
        np.nanmedian(abs_vmag_array_ml),
        np.nanmedian(max_abs_vmag_ml)

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
