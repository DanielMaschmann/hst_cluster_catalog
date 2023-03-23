""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import photometry_tools
import pandas as pd
import seaborn as sns

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
# catalog_access.load_morph_mask_target_list(target_list=target_list)
# np.save('morph_mask_data.npy', catalog_access.morph_mask_data)
catalog_access.morph_mask_data = np.load('morph_mask_data.npy', allow_pickle=True).item()

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')

catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


age_disc_only = np.array([])
age_bar = np.array([])
age_arm = np.array([])
age_ring = np.array([])
age_center = np.array([])
age_lens = np.array([])
age_bulge = np.array([])

mass_disc_only = np.array([])
mass_bar = np.array([])
mass_arm = np.array([])
mass_ring = np.array([])
mass_center = np.array([])
mass_lens = np.array([])
mass_bulge = np.array([])

for target in target_list:
    ra_12, dec_12 = catalog_access.get_hst_cc_coords_world(target=target)
    age_12 = catalog_access.get_hst_cc_age(target=target)
    mass_12 = catalog_access.get_hst_cc_stellar_m(target=target)

    ra_3, dec_3 = catalog_access.get_hst_cc_coords_world(target=target, cluster_class='class3')
    age_3 = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    mass_3 = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')

    pos_mask_dict_12 = catalog_access.get_morph_locations(target=target, ra=ra_12, dec=dec_12)
    pos_mask_dict_3 = catalog_access.get_morph_locations(target=target, ra=ra_3, dec=dec_3)

    age_disc_only = np.concatenate([age_disc_only, age_12[pos_mask_dict_12['pos_mask_disc_only']], age_3[pos_mask_dict_3['pos_mask_disc_only']]])
    age_bar = np.concatenate([age_bar, age_12[pos_mask_dict_12['pos_mask_bar']], age_3[pos_mask_dict_3['pos_mask_bar']]])
    age_arm = np.concatenate([age_arm, age_12[pos_mask_dict_12['pos_mask_arm']], age_3[pos_mask_dict_3['pos_mask_arm']]])
    age_ring = np.concatenate([age_ring, age_12[pos_mask_dict_12['pos_mask_ring']], age_3[pos_mask_dict_3['pos_mask_ring']]])
    age_center = np.concatenate([age_center, age_12[pos_mask_dict_12['pos_mask_center']], age_3[pos_mask_dict_3['pos_mask_center']]])
    age_lens = np.concatenate([age_lens, age_12[pos_mask_dict_12['pos_mask_lens']], age_3[pos_mask_dict_3['pos_mask_lens']]])
    age_bulge = np.concatenate([age_bulge, age_12[pos_mask_dict_12['pos_mask_bulge']], age_3[pos_mask_dict_3['pos_mask_bulge']]])

    mass_disc_only = np.concatenate([mass_disc_only, mass_12[pos_mask_dict_12['pos_mask_disc_only']], mass_3[pos_mask_dict_3['pos_mask_disc_only']]])
    mass_bar = np.concatenate([mass_bar, mass_12[pos_mask_dict_12['pos_mask_bar']], mass_3[pos_mask_dict_3['pos_mask_bar']]])
    mass_arm = np.concatenate([mass_arm, mass_12[pos_mask_dict_12['pos_mask_arm']], mass_3[pos_mask_dict_3['pos_mask_arm']]])
    mass_ring = np.concatenate([mass_ring, mass_12[pos_mask_dict_12['pos_mask_ring']], mass_3[pos_mask_dict_3['pos_mask_ring']]])
    mass_center = np.concatenate([mass_center, mass_12[pos_mask_dict_12['pos_mask_center']], mass_3[pos_mask_dict_3['pos_mask_center']]])
    mass_lens = np.concatenate([mass_lens, mass_12[pos_mask_dict_12['pos_mask_lens']], mass_3[pos_mask_dict_3['pos_mask_lens']]])
    mass_bulge = np.concatenate([mass_bulge, mass_12[pos_mask_dict_12['pos_mask_bulge']], mass_3[pos_mask_dict_3['pos_mask_bulge']]])





df = pd.DataFrame(columns=['x_axis', 'age', 'mass'])

panda_data = pd.DataFrame({'x_axis': ['disc only'] * len(age_disc_only), 'age': np.log10(age_disc_only) + 6, 'mass': np.log10(mass_disc_only) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['bar'] * len(age_bar), 'age': np.log10(age_bar) + 6, 'mass': np.log10(mass_bar) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['arm'] * len(age_arm), 'age': np.log10(age_arm) + 6, 'mass': np.log10(mass_arm) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['ring'] * len(age_ring), 'age': np.log10(age_ring) + 6, 'mass': np.log10(mass_ring) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['center'] * len(age_center), 'age': np.log10(age_center) + 6, 'mass': np.log10(mass_center) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['lens'] * len(age_lens), 'age': np.log10(age_lens) + 6, 'mass': np.log10(mass_lens) + 6})
df = df.append(panda_data, ignore_index=True)
panda_data = pd.DataFrame({'x_axis': ['bulge'] * len(age_bulge), 'age': np.log10(age_bulge) + 6, 'mass': np.log10(mass_bulge) + 6})
df = df.append(panda_data, ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 10))
sns.violinplot(ax=ax, data=df, x='x_axis', y='age', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Age/yr)', fontsize=15)
plt.xticks(rotation=45)
# ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=13)
plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('plot_output/violin_age_hum_1.png', bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))
sns.violinplot(ax=ax, data=df, x='x_axis', y='mass', color='skyblue', split=True)
ax.set_xlabel('')
ax.set_ylabel('log(Mass)', fontsize=15)
plt.xticks(rotation=45)
# ax.set_title('Human Classified 1/2/3 | Increasing Distance', fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=13)
plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('plot_output/violin_age_hum_1.png', bbox_inches='tight', dpi=300)
plt.show()

exit()

