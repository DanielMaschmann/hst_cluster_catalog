import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import seaborn as sns


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path)


target_list = catalog_access.target_hst_cc
dist_list = []
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]


catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


df_hum_1 = pd.DataFrame(columns=['gal', 'age', 'mass'])
df_hum_2 = pd.DataFrame(columns=['gal', 'age', 'mass'])
df_hum_3 = pd.DataFrame(columns=['gal', 'age', 'mass'])

df_ml_1 = pd.DataFrame(columns=['gal', 'age', 'mass'])
df_ml_2 = pd.DataFrame(columns=['gal', 'age', 'mass'])
df_ml_3 = pd.DataFrame(columns=['gal', 'age', 'mass'])


for index in range(0, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')

    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')

    ages_hum = np.log10(np.concatenate([age_12_hum, age_3_hum])) + 6
    masses_hum = np.log10(np.concatenate([m_star_12_hum, m_star_3_hum]))

    cluster_class_qual = np.concatenate([cluster_class_qual_12_ml, cluster_class_qual_3_ml])
    ages_ml = np.log10(np.concatenate([age_12_ml, age_3_ml])) + 6
    masses_ml = np.log10(np.concatenate([m_star_12_ml, m_star_3_ml]))

    patient_df_hum = pd.DataFrame({'gal': [target.upper()]*len(ages_hum), 'age': ages_hum, 'mass': masses_hum})
    patient_df_ml = pd.DataFrame({'gal': [target.upper()]*len(ages_ml[cluster_class_qual >= 0.9]),
                                  'age': ages_ml[cluster_class_qual >= 0.9],
                                  'mass': masses_ml[cluster_class_qual >= 0.9]})

    if index < 13:
        df_hum_1 = df_hum_1.append(patient_df_hum, ignore_index=True)
        df_ml_1 = df_ml_1.append(patient_df_ml, ignore_index=True)
    elif (index > 13) & (index < 26):
        df_hum_2 = df_hum_2.append(patient_df_hum, ignore_index=True)
        df_ml_2 = df_ml_2.append(patient_df_ml, ignore_index=True)
    else:
        df_hum_3 = df_hum_3.append(patient_df_hum, ignore_index=True)
        df_ml_3 = df_ml_3.append(patient_df_ml, ignore_index=True)

figsize=(20, 13)
fontsize = 20


fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_1, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_hum_1.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_2, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_hum_2.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_3, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_hum_3.png', bbox_inches='tight', dpi=300)



fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_1, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_ml_1.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_2, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_ml_2.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_3, x='gal', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_age_ml_3.png', bbox_inches='tight', dpi=300)


# mass
fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_1, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_hum_1.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_2, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_hum_2.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_hum_3, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_hum_3.png', bbox_inches='tight', dpi=300)



fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_1, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_ml_1.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_2, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_ml_2.png', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(ax=ax, data=df_ml_3, x='gal', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(M$_{*}$/M$_{\odot}$)', fontsize=fontsize)
plt.xticks(rotation=45)
ax.set_title('ML Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot_output/violin_m_star_ml_3.png', bbox_inches='tight', dpi=300)






