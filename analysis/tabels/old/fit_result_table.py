import numpy as np
import photometry_tools

"""
This macro will produce Table 1 for the PHANGS HST paper
"""


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)


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
n_hum_1 = []
n_hum_2 = []
n_hum_3 = []
n_ml_1 = []
n_ml_2 = []
n_ml_3 = []

median_ebv_hum = []
median_ebv_ml = []

median_mstar_hum = []
median_mstar_ml = []

min_mstar_hum = []
min_mstar_ml = []

max_mstar_hum = []
max_mstar_ml = []


mstar_hum = np.array([])
mstar_ml = np.array([])

ebv_hum = np.array([])
ebv_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_candidates = catalog_access.get_hst_cc_class_human(target=target, classify='', cluster_class='candidates')
    number_candidates = len(catalog_access.hst_cc_data[target + '_' + '_' + 'candidates'])

    cluster_class_hum_12 = catalog_access.get_hst_cc_class_human(target=target)
    cluster_class_hum_3 = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')

    ebv_hum_12 = catalog_access.get_hst_cc_ebv(target=target)
    ebv_hum_3 = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    ebv_ml_12 = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    ebv_ml_3 = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')

    mstar_hum_12 = catalog_access.get_hst_cc_stellar_m(target=target)
    mstar_hum_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')
    mstar_ml_12 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    mstar_ml_3 = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')

    number_hum_1 = np.sum(cluster_class_hum_12 == 1)
    number_hum_2 = np.sum(cluster_class_hum_12 == 2)
    number_hum_3 = np.sum(cluster_class_hum_3 == 3)
    number_ml_1 = np.sum(cluster_class_ml_12 == 1)
    number_ml_2 = np.sum(cluster_class_ml_12 == 2)
    number_ml_3 = np.sum(cluster_class_ml_3 == 3)

    n_candidates.append(number_candidates)
    n_hum_1.append(number_hum_1)
    n_hum_2.append(number_hum_2)
    n_hum_3.append(number_hum_3)
    n_ml_1.append(number_ml_1)
    n_ml_2.append(number_ml_2)
    n_ml_3.append(number_ml_3)

    median_ebv_hum.append(np.nanmedian(np.concatenate([ebv_hum_12, ebv_hum_3])))
    median_ebv_ml.append(np.nanmedian(np.concatenate([ebv_ml_12, ebv_ml_3])))

    good_mstar_value_hum_12 = mstar_hum_12 > 0
    good_mstar_value_hum_3 = mstar_hum_3 > 0
    good_mstar_value_ml_12 = mstar_ml_12 > 0
    good_mstar_value_ml_3 = mstar_ml_3 > 0

    median_mstar_hum.append(np.nanmedian(np.concatenate([mstar_hum_12[good_mstar_value_hum_12], mstar_hum_3[good_mstar_value_hum_3]])))
    median_mstar_ml.append(np.nanmedian(np.concatenate([mstar_ml_12[good_mstar_value_ml_12], mstar_ml_3[good_mstar_value_ml_3]])))

    min_mstar_hum.append(np.nanmin(np.concatenate([mstar_hum_12[good_mstar_value_hum_12], mstar_hum_3[good_mstar_value_hum_3]])))
    min_mstar_ml.append(np.nanmin(np.concatenate([mstar_ml_12[good_mstar_value_ml_12], mstar_ml_3[good_mstar_value_ml_3]])))

    max_mstar_hum.append(np.nanmax(np.concatenate([mstar_hum_12[good_mstar_value_hum_12], mstar_hum_3[good_mstar_value_hum_3]])))
    max_mstar_ml.append(np.nanmax(np.concatenate([mstar_ml_12[good_mstar_value_ml_12], mstar_ml_3[good_mstar_value_ml_3]])))

    mstar_hum = np.concatenate([mstar_hum, mstar_hum_12[good_mstar_value_hum_12], mstar_hum_3[good_mstar_value_hum_3]])
    mstar_ml = np.concatenate([mstar_ml, mstar_ml_12[good_mstar_value_ml_12], mstar_ml_3[good_mstar_value_ml_3]])

    ebv_hum = np.concatenate([ebv_hum, ebv_hum_12, ebv_hum_3])
    ebv_ml = np.concatenate([ebv_ml, ebv_ml_12, ebv_ml_3])




print("")
print('\multicolumn{1}{c}{Galaxy} & '
      '\multicolumn{1}{c}{N$_{\\rm Cand}$} & '
      '\multicolumn{4}{c}{Human} & '
      '\multicolumn{4}{c}{ML} & '
      
      '\multicolumn{1}{c}{log(M$_{*}^{\\rm Hum})$} & '
      '\multicolumn{1}{c}{E(B-V)$^{\\rm Hum}$} & '
      
      '\multicolumn{1}{c}{log(M$_{*}^{\\rm ML})$} & '
      '\multicolumn{1}{c}{E(B-V)$^{\\rm ML}$} \\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      
      '\multicolumn{1}{c}{C1} & '
      '\multicolumn{1}{c}{C2} & '
      '\multicolumn{1}{c}{C3} & '
      '\multicolumn{1}{c}{total} & '
      
      '\multicolumn{1}{c}{C1} & '
      '\multicolumn{1}{c}{C2} & '
      '\multicolumn{1}{c}{C3} & '
      '\multicolumn{1}{c}{total} & '
      
      '\multicolumn{1}{c}{min$\\vert$ med$\\vert$ max} & '
      '\multicolumn{1}{c}{med} & '
      
      '\multicolumn{1}{c}{min$\\vert$ med$\\vert$ max} & '
      '\multicolumn{1}{c}{med}\\\\ ')
print('\\hline')
print('\multicolumn{1}{c}{} & '
      '\multicolumn{1}{c}{} & '
      '\multicolumn{4}{c}{} & '
      '\multicolumn{4}{c}{} & '
      
      '\multicolumn{1}{c}{M$_{\odot}$} & '
      '\multicolumn{1}{c}{mag} & '
      
      '\multicolumn{1}{c}{M$_{\odot}$} & '
      '\multicolumn{1}{c}{mag} \\\\ ')
print('\\hline')

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

        '%.1f$\\vert$ %.1f$\\vert$ %.1f & '
        '%.1f & '

        '%.1f$\\vert$ %.1f$\\vert$ %.1f & '
        '%.1f \\\\ '
        % (target_string,
           n_candidates[index],
           n_hum_1[index],
           n_hum_2[index],
           n_hum_3[index],
           n_hum_1[index] + n_hum_2[index] + n_hum_3[index],

           n_ml_1[index],
           n_ml_2[index],
           n_ml_3[index],
           n_ml_1[index] + n_ml_2[index] + n_ml_3[index],

           np.log10(min_mstar_hum)[index],
           np.log10(median_mstar_hum)[index],
           np.log10(max_mstar_hum)[index],
           median_ebv_hum[index],

           np.log10(min_mstar_ml)[index],
           np.log10(median_mstar_ml)[index],
           np.log10(max_mstar_ml)[index],
           median_ebv_ml[index]
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

    '%.1f & '
    '%.1f & '

    '%.1f & '
    '%.1f \\\\ '
    % (
        np.median(n_candidates),

        np.median(n_hum_1),
        np.median(n_hum_2),
        np.median(n_hum_3),
        np.median(np.array(n_hum_1) + np.array(n_hum_2) + np.array(n_hum_3)),
        np.median(n_ml_1),
        np.median(n_ml_2),
        np.median(n_ml_3),
        np.median(np.array(n_ml_1) + np.array(n_ml_2) + np.array(n_ml_3)),

        np.log10(np.median(mstar_hum)),
        np.median(ebv_hum),

        np.log10(np.median(mstar_ml)),
        np.median(ebv_ml)
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

    '%.1f & '
    '%.1f & '

    '%.1f & '
    '%.1f \\\\ '
    % (
        np.mean(n_candidates),

        np.mean(n_hum_1),
        np.mean(n_hum_2),
        np.mean(n_hum_3),
        np.mean(np.array(n_hum_1) + np.array(n_hum_2) + np.array(n_hum_3)),
        np.mean(n_ml_1),
        np.mean(n_ml_2),
        np.mean(n_ml_3),
        np.mean(np.array(n_ml_1) + np.array(n_ml_2) + np.array(n_ml_3)),

        np.log10(np.mean(mstar_hum)),
        np.mean(ebv_hum),

        np.log10(np.mean(mstar_ml)),
        np.mean(ebv_ml)
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

    '- & '
    '- & '

    '- & '
    '- \\\\ '
    % (
        np.sum(n_candidates),

        np.sum(n_hum_1),
        np.sum(n_hum_2),
        np.sum(n_hum_3),
        np.sum(n_hum_1) + np.sum(n_hum_2) + np.sum(n_hum_3),
        np.sum(n_ml_1),
        np.sum(n_ml_2),
        np.sum(n_ml_3),
        np.sum(n_ml_1) + np.sum(n_ml_2) + np.sum(n_ml_3),
    )
)
print('\\hline')
