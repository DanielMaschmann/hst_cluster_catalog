import os.path

import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from matplotlib.colorbar import ColorbarBase
import matplotlib
from shapely.geometry import polygon, point




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
for target in target_list:
    if (target == 'ngc0628c') | (target == 'ngc0628e'):
        target = 'ngc0628'
    dist_list.append(catalog_access.dist_dict[target]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list, classify='human')
catalog_access.load_hst_cc_list(target_list=target_list, classify='human', cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


color_c1 = 'darkorange'
color_c2 = 'tab:green'
color_c3 = 'tab:gray'

cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.LogNorm(vmin=5, vmax=100)


extra_file_path = '/home/benutzer/data/PHANGS_products/HST_catalogs/Extrainfotables/'

hull_mask_folder = '/home/benutzer/Documents/projects/hst_cluster_catalog/analysis/identify_gc/data_output/'

def plot_age_ebv_panels(classify='human', cluster_class='class12', h_alpha_cut=3, save_pdf=False):
    fig_1, ax_1 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fig_2, ax_2 = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(16, 18))
    fontsize = 18

    row_index = 0
    col_index = 0
    for index in range(0, 20):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)

        age = catalog_access.get_hst_cc_age(target=target, classify=classify, cluster_class=cluster_class)
        ebv = catalog_access.get_hst_cc_ebv(target=target, classify=classify, cluster_class=cluster_class)

        color_vi = catalog_access.get_hst_color_vi_vega(target=target, classify=classify, cluster_class=cluster_class)
        color_ub = catalog_access.get_hst_color_ub_vega(target=target, classify=classify, cluster_class=cluster_class)

        cluster_id = catalog_access.get_hst_cc_phangs_id(target=target, classify=classify, cluster_class=cluster_class)

        h_alpha_intensity = np.zeros(len(cluster_id))
        hii_mask = np.zeros(len(cluster_id))
        if target == 'ngc0685':
            target_ext = 'ngc685'
        elif target == 'ngc0628c':
            target_ext = 'ngc628c'
        elif target == 'ngc0628e':
            target_ext = 'ngc628e'
        else:
            target_ext = target
        extra_file = extra_file_path + '%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
        if os.path.isfile(extra_file):
            extra_tab_hdu = fits.open(extra_file)
            extra_data_table = extra_tab_hdu[1].data
            extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
            extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
            extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev%i' % h_alpha_cut]

            for running_index, cluster_id in enumerate(cluster_id):
                index_extra_table = np.where(extra_cluster_id == cluster_id)
                h_alpha_intensity[running_index] = extra_h_alpha_intensity[index_extra_table]
                hii_mask[running_index] = extra_hii_mask[index_extra_table]
        else:
            print(extra_file, ' does not exist')

        h_alpha_mask = (h_alpha_intensity > h_alpha_cut) & (hii_mask > 0)

        # load hull
        if classify == 'human':
            hull_data = fits.open(hull_mask_folder + 'hull_table_hum.fits')[1].data

        else:
            hull_data = fits.open(hull_mask_folder + 'hull_table_ml.fits')[1].data
        vi_hull = hull_data['vi_hull']
        ub_hull = hull_data['ub_hull']

        zipped = zip(vi_hull, ub_hull)
        verticis = list(zipped)
        ubvi_poly = polygon.Polygon(verticis)

        gc_mask = np.zeros(len(color_ub), dtype=bool)
        for point_index in range(len(gc_mask)):
            color_point = point.Point(color_vi[point_index], color_ub[point_index])
            gc_mask[point_index] = ubvi_poly.contains(color_point)

        # data points
        ax_1[row_index, col_index].scatter((np.log10(age) + 6)[~h_alpha_mask], ebv[~h_alpha_mask],
                                           color='silver', s=10)
        ax_1[row_index, col_index].scatter((np.log10(age) + 6)[gc_mask], ebv[gc_mask],
                                           edgecolor='red', s=40, linewidth=2, facecolor='None')
        ax_1[row_index, col_index].scatter((np.log10(age) + 6)[h_alpha_mask], ebv[h_alpha_mask],
                                           c=h_alpha_intensity[h_alpha_mask], norm=norm, cmap=cmap, s=30)

        ax_1[row_index, col_index].plot([7, 7], [-0.1, 2.5], color='k', linestyle='--', linewidth=1.5)

        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_1[row_index, col_index].add_artist(anchored_left)
        ax_1[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                 direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0


    ax_1[0, 0].set_xlim(5.7, 10.3)
    ax_1[0, 0].set_ylim(-0.1, 2.1)
    fig_1.text(0.5, 0.89, r'%s %s H$\alpha$ > %.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify.upper(), cluster_class.upper(), h_alpha_cut),
               ha='center', fontsize=fontsize)
    fig_1.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
    fig_1.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)

    # plt.tight_layout()
    fig_1.subplots_adjust(wspace=0, hspace=0)

    fig_name_str = 'age_ebv_panel_1_%s_c%s_h%s' % (classify, cluster_class[5:], h_alpha_cut)
    fig_1.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_1.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)

    row_index = 0
    col_index = 0
    for index in range(20, 39):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)

        age = catalog_access.get_hst_cc_age(target=target, classify=classify, cluster_class=cluster_class)
        ebv = catalog_access.get_hst_cc_ebv(target=target, classify=classify, cluster_class=cluster_class)

        color_vi = catalog_access.get_hst_color_vi_vega(target=target, classify=classify, cluster_class=cluster_class)
        color_ub = catalog_access.get_hst_color_ub_vega(target=target, classify=classify, cluster_class=cluster_class)

        cluster_id = catalog_access.get_hst_cc_phangs_id(target=target, classify=classify, cluster_class=cluster_class)

        h_alpha_intensity = np.zeros(len(cluster_id))
        hii_mask = np.zeros(len(cluster_id))
        if target == 'ngc0685':
            target_ext = 'ngc685'
        elif target == 'ngc0628c':
            target_ext = 'ngc628c'
        elif target == 'ngc0628e':
            target_ext = 'ngc628e'
        else:
            target_ext = target
        extra_file = extra_file_path + '%s_phangshst_candidates_bcw_v1p2_IR4_Extrainfo.fits' % target_ext
        if os.path.isfile(extra_file):
            extra_tab_hdu = fits.open(extra_file)
            extra_data_table = extra_tab_hdu[1].data
            extra_cluster_id = extra_data_table['ID_PHANGS_CLUSTERS_v1p2']
            extra_h_alpha_intensity = extra_data_table['NBHa_intensity_medsub']
            extra_hii_mask = extra_data_table['NBHa_mask_medsub_lev%i' % h_alpha_cut]

            for running_index, cluster_id in enumerate(cluster_id):
                index_extra_table = np.where(extra_cluster_id == cluster_id)
                h_alpha_intensity[running_index] = extra_h_alpha_intensity[index_extra_table]
                hii_mask[running_index] = extra_hii_mask[index_extra_table]
        else:
            print(extra_file, ' does not exist')

        h_alpha_mask = (h_alpha_intensity > h_alpha_cut) & (hii_mask > 0)

        # load hull
        if classify == 'human':
            hull_data = fits.open(hull_mask_folder + 'hull_table_hum.fits')[1].data

        else:
            hull_data = fits.open(hull_mask_folder + 'hull_table_ml.fits')[1].data
        vi_hull = hull_data['vi_hull']
        ub_hull = hull_data['ub_hull']

        zipped = zip(vi_hull, ub_hull)
        verticis = list(zipped)
        ubvi_poly = polygon.Polygon(verticis)

        gc_mask = np.zeros(len(color_ub), dtype=bool)
        for point_index in range(len(gc_mask)):
            color_point = point.Point(color_vi[point_index], color_ub[point_index])
            gc_mask[point_index] = ubvi_poly.contains(color_point)

        # data points
        ax_2[row_index, col_index].scatter((np.log10(age) + 6)[~h_alpha_mask], ebv[~h_alpha_mask],
                                           color='silver', s=10)
        ax_2[row_index, col_index].scatter((np.log10(age) + 6)[gc_mask], ebv[gc_mask],
                                           edgecolor='red', s=40, linewidth=2, facecolor='None')
        ax_2[row_index, col_index].scatter((np.log10(age) + 6)[h_alpha_mask], ebv[h_alpha_mask],
                                           c=h_alpha_intensity[h_alpha_mask], norm=norm, cmap=cmap, s=30)
        ax_2[row_index, col_index].plot([7, 7], [-0.1, 2.5], color='k', linestyle='--', linewidth=1.5)

        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_2[row_index, col_index].add_artist(anchored_left)
        ax_2[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                 direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0


    ax_2[0, 0].set_xlim(5.7, 10.3)
    ax_2[0, 0].set_ylim(-0.1, 2.1)
    fig_2.text(0.5, 0.89, r'%s %s H$\alpha$ > %.1f 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$' %
               (classify.upper(), cluster_class.upper(), h_alpha_cut),
               ha='center', fontsize=fontsize)
    fig_2.text(0.5, 0.08, 'log(Age/yr)', ha='center', fontsize=fontsize)
    fig_2.text(0.08, 0.5, 'E(B-V)', va='center', rotation='vertical', fontsize=fontsize)

    ax_2[4, 3].plot([], [], color='k', label='ML Class 1|2')
    ax_2[4, 3].legend(frameon=False, fontsize=fontsize)
    ax_2[4, 3].axis('off')

    ax_cbar = fig_2.add_axes([0.73, 0.16, 0.15, 0.015])
    ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap, norm=norm, extend='neither', ticks=None)
    ax_cbar.set_xlabel(r'$\Sigma$ H$\alpha$ 10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$', labelpad=0, fontsize=fontsize-3)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    fig_2.subplots_adjust(wspace=0, hspace=0)

    fig_name_str = 'age_ebv_panel_2_%s_c%s_h%s' % (classify, cluster_class[5:], h_alpha_cut)
    fig_2.savefig('plot_output/' + fig_name_str + '.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig_2.savefig('plot_output/' + fig_name_str + '.pdf', bbox_inches='tight', dpi=300)


plot_age_ebv_panels(classify='human', cluster_class='class12', h_alpha_cut=2, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class12', h_alpha_cut=3, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class12', h_alpha_cut=5, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class12', h_alpha_cut=10, save_pdf=False)

plot_age_ebv_panels(classify='ml', cluster_class='class12', h_alpha_cut=2, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class12', h_alpha_cut=3, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class12', h_alpha_cut=5, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class12', h_alpha_cut=10, save_pdf=False)



plot_age_ebv_panels(classify='human', cluster_class='class3', h_alpha_cut=2, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class3', h_alpha_cut=3, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class3', h_alpha_cut=5, save_pdf=False)
plot_age_ebv_panels(classify='human', cluster_class='class3', h_alpha_cut=10, save_pdf=False)

plot_age_ebv_panels(classify='ml', cluster_class='class3', h_alpha_cut=2, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class3', h_alpha_cut=3, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class3', h_alpha_cut=5, save_pdf=False)
plot_age_ebv_panels(classify='ml', cluster_class='class3', h_alpha_cut=10, save_pdf=False)






