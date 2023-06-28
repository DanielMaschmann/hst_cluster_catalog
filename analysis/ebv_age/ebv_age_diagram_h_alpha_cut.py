import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import os
from astropy.io import fits
from matplotlib.colorbar import ColorbarBase
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=1, vmax=20)

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


color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'darkorange'

fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
fontsize = 18

row_index = 0
col_index = 0
for index in range(0, 20):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
    class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
    class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

    h_alpha_intensity_hum_12 = np.zeros(len(age_12_hum))
    hii_reg_hum_12 = np.zeros(len(age_12_hum))
    if target == 'ngc0685':
        target_ext = 'ngc685'
    elif target == 'ngc0628c':
        target_ext = 'ngc628c'
    elif target == 'ngc0628e':
        target_ext = 'ngc628e'
    else:
        target_ext = target
    extra_file = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
    if os.path.isfile(extra_file):
        extra_tab_hdu = fits.open(extra_file)
        extra_data_table = extra_tab_hdu[1].data
        extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
        extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
        extra_hii_reg_id = extra_data_table['NBHa_HIIreg']

        for running_index, cluster_id in enumerate(cluster_id_hum_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_12[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_hum_12[running_index] = extra_hii_reg_id[index_extra_table]
    else:
        print(extra_file, ' does not exist')

    h_alpha_coverage = (h_alpha_intensity_hum_12 > 1) & (hii_reg_hum_12 > 0)

    ax[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         ebv_12_hum[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         c=h_alpha_intensity_hum_12[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         s=20, cmap=cmap, norm=norm, alpha=0.7)
    ax[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         ebv_12_hum[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         c=h_alpha_intensity_hum_12[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         s=20, cmap=cmap, norm=norm, alpha=0.7)

    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax[row_index, col_index].add_artist(anchored_left)
    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0

ax[0, 0].set_xlim(5.7, 10.3)
ax[0, 0].set_ylim(-0.1, 2.1)
fig.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)

# plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/age_ebv_hum_h_alpha_1.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/age_ebv_hum_h_alpha_1.pdf', bbox_inches='tight', dpi=300)



fig, ax = plt.subplots(5, 4, sharex=True, sharey=True)
fig.set_size_inches(16, 18)

fontsize = 18

row_index = 0
col_index = 0
for index in range(20, 39):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)
    cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
    age_12_hum = catalog_access.get_hst_cc_age(target=target)
    ebv_12_hum = catalog_access.get_hst_cc_ebv(target=target)
    cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
    age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
    ebv_3_hum = catalog_access.get_hst_cc_ebv(target=target, cluster_class='class3')
    cluster_id_hum_12 = catalog_access.get_hst_cc_phangs_id(target)
    cluster_id_hum_3 = catalog_access.get_hst_cc_phangs_id(target, cluster_class='class3')

    cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
    ebv_12_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml')
    cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
    ebv_3_ml = catalog_access.get_hst_cc_ebv(target=target, classify='ml', cluster_class='class3')
    class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
    class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
    class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

    h_alpha_intensity_hum_12 = np.zeros(len(age_12_hum))
    hii_reg_hum_12 = np.zeros(len(age_12_hum))
    if target == 'ngc0685':
        target_ext = 'ngc685'
    elif target == 'ngc0628c':
        target_ext = 'ngc628c'
    elif target == 'ngc0628e':
        target_ext = 'ngc628e'
    else:
        target_ext = target
    extra_file = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
    if os.path.isfile(extra_file):
        extra_tab_hdu = fits.open(extra_file)
        extra_data_table = extra_tab_hdu[1].data
        extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
        extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
        extra_hii_reg_id = extra_data_table['NBHa_HIIreg']

        for running_index, cluster_id in enumerate(cluster_id_hum_12):
            index_extra_table = np.where(extra_cluster_id == cluster_id)
            h_alpha_intensity_hum_12[running_index] = extra_h_alpha_intensity[index_extra_table]
            hii_reg_hum_12[running_index] = extra_hii_reg_id[index_extra_table]
    else:
        print(extra_file, ' does not exist')


    h_alpha_coverage = (h_alpha_intensity_hum_12 > 1) & (hii_reg_hum_12 > 0)

    ax[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         ebv_12_hum[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         c=h_alpha_intensity_hum_12[(cluster_class_12_hum == 1) & h_alpha_coverage],
                                         s=20, cmap=cmap, norm=norm, alpha=0.7)
    ax[row_index, col_index].scatter((np.log10(age_12_hum) + 6)[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         ebv_12_hum[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         c=h_alpha_intensity_hum_12[(cluster_class_12_hum == 2) & h_alpha_coverage],
                                         s=20, cmap=cmap, norm=norm, alpha=0.7)

    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    ax[row_index, col_index].add_artist(anchored_left)
    ax[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                             direction='in', labelsize=fontsize)
    anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                 frameon=False, prop=dict(size=fontsize-4))
    col_index += 1
    if col_index == 4:
        row_index += 1
        col_index = 0


ax[0, 0].set_xlim(5.7, 10.3)
ax[0, 0].set_ylim(-0.1, 2.1)
fig.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
fig.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)
ax[4, 3].scatter([], [], c=color_c1, s=30, label='Class 1')
ax[4, 3].scatter([], [], c=color_c2, s=30, label='Class 2')
ax[4, 3].legend(frameon=False, fontsize=fontsize)
ax[4, 3].axis('off')

ax_cbar = fig.add_axes([0.73, 0.1, 0.11, 0.015])
ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
ax_cbar.set_xlabel(r'$\Sigma$ H$\alpha$ 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=0, fontsize=fontsize)
ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)


# plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/age_ebv_hum_h_alpha_2.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/age_ebv_hum_h_alpha_2.pdf', bbox_inches='tight', dpi=300)



