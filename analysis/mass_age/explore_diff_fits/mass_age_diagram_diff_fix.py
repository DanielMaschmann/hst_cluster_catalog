import numpy as np
import photometry_tools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits


# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod = hdu_a[1].data
age_mod = data_mod['sfh.age']
m_star_mod = data_mod['stellar.m_star']
flux_f555w_mod = data_mod['F555W_UVIS_CHIP2']

ABmag_F555W = - 6
f_mJy_f555w = 1.e3 * 1.e23 * 10.**((ABmag_F555W + 48.6) / -2.5)
lower_lim_m_star = np.log10(f_mJy_f555w * m_star_mod / flux_f555w_mod)


cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'

for catalog_version in ['SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha1_inclusiveGCcc_restrictiveGCclass',
                        'SEDfix_Ha2_inclusiveGCcc_restrictiveGCclass',
                        'SEDfix_Ha2_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha3_inclusiveGCcc_inclusiveGCclass',
                        'SEDfix_Ha3_inclusiveGCcc_restrictiveGCclass']:

    catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                                hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                                hst_cc_ver=catalog_version)


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


    fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
    fig_hum.set_size_inches(16, 18)
    fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
    fig_ml.set_size_inches(16, 18)
    fontsize = 18

    row_index = 0
    col_index = 0
    for index in range(0, 20):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)
        cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
        age_12_hum = catalog_access.get_hst_cc_age(target=target)
        m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
        cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
        age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
        m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')

        cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
        cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
        age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
        m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
        cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
        cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
        age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
        m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
        class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
        class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
        class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

        # random dots
        random_x_12_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_12_hum))
        random_x_3_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_3_hum))

        ax_hum[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
        ax_hum[row_index, col_index].scatter((np.log10(age_3_hum) + random_x_3_hum + 6)[cluster_class_3_hum == 3],
                                             np.log10(m_star_3_hum[cluster_class_3_hum == 3]), c='royalblue', s=1)
        ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + random_x_12_hum + 6)[cluster_class_12_hum == 1],
                                             np.log10(m_star_12_hum)[cluster_class_12_hum == 1], c='darkorange', s=1)
        ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + random_x_12_hum + 6)[cluster_class_12_hum == 2],
                                             np.log10(m_star_12_hum)[cluster_class_12_hum == 2], c='forestgreen', s=1)


        # random dots
        random_x_12_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_12_ml))
        random_x_3_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_3_ml))

        ax_ml[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
        ax_ml[row_index, col_index].scatter((np.log10(age_3_ml) + random_x_3_ml + 6)[cluster_class_3_ml == 3],
                                             np.log10(m_star_3_ml[cluster_class_3_ml == 3]), c='royalblue', s=1)
        ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + random_x_12_ml + 6)[cluster_class_12_ml == 1],
                                             np.log10(m_star_12_ml)[cluster_class_12_ml == 1], c='darkorange', s=1)
        ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + random_x_12_ml + 6)[cluster_class_12_ml == 2],
                                             np.log10(m_star_12_ml)[cluster_class_12_ml == 2], c='forestgreen', s=1)


        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                 direction='in', labelsize=fontsize)

        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)
        ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0

    ax_hum[0, 0].set_xlim(5.7, 10.3)
    ax_hum[0, 0].set_ylim(1.9, 8.7)
    fig_hum.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
    fig_hum.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

    ax_ml[0, 0].set_xlim(5.7, 10.3)
    ax_ml[0, 0].set_ylim(1.9, 8.7)
    fig_ml.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
    fig_ml.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)

    # plt.tight_layout()
    fig_hum.subplots_adjust(wspace=0, hspace=0)
    fig_hum.savefig('plot_output/age_m_star_hum_1_%s.png' % catalog_version, bbox_inches='tight', dpi=300)
    # fig_hum.savefig('plot_output/age_m_star_hum_1.pdf', bbox_inches='tight', dpi=300)

    fig_ml.subplots_adjust(wspace=0, hspace=0)
    fig_ml.savefig('plot_output/age_m_star_ml_1_%s.png' % catalog_version, bbox_inches='tight', dpi=300)
    # fig_ml.savefig('plot_output/age_m_star_ml_1.pdf', bbox_inches='tight', dpi=300)



    fig_hum, ax_hum = plt.subplots(5, 4, sharex=True, sharey=True)
    fig_hum.set_size_inches(16, 18)
    fig_ml, ax_ml = plt.subplots(5, 4, sharex=True, sharey=True)
    fig_ml.set_size_inches(16, 18)
    fontsize = 18

    row_index = 0
    col_index = 0
    for index in range(20, 39):
        target = target_list[index]
        dist = dist_list[index]
        print('target ', target, 'dist ', dist)
        cluster_class_12_hum = catalog_access.get_hst_cc_class_human(target=target)
        age_12_hum = catalog_access.get_hst_cc_age(target=target)
        m_star_12_hum = catalog_access.get_hst_cc_stellar_m(target=target)
        cluster_class_3_hum = catalog_access.get_hst_cc_class_human(target=target, cluster_class='class3')
        age_3_hum = catalog_access.get_hst_cc_age(target=target, cluster_class='class3')
        m_star_3_hum = catalog_access.get_hst_cc_stellar_m(target=target, cluster_class='class3')

        cluster_class_12_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
        cluster_class_qual_12_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
        age_12_ml = catalog_access.get_hst_cc_age(target=target, classify='ml')
        m_star_12_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml')
        cluster_class_3_ml = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
        cluster_class_qual_3_ml = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
        age_3_ml = catalog_access.get_hst_cc_age(target=target, classify='ml', cluster_class='class3')
        m_star_3_ml = catalog_access.get_hst_cc_stellar_m(target=target, classify='ml', cluster_class='class3')
        class_1_ml = (cluster_class_12_ml == 1) & (cluster_class_qual_12_ml >= 0.9)
        class_2_ml = (cluster_class_12_ml == 2) & (cluster_class_qual_12_ml >= 0.9)
        class_3_ml = (cluster_class_3_ml == 3) & (cluster_class_qual_3_ml >= 0.9)

        # random dots
        random_x_12_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_12_hum))
        random_x_3_hum = np.random.uniform(low=-0.1, high=0.1, size=len(age_3_hum))

        ax_hum[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
        ax_hum[row_index, col_index].scatter((np.log10(age_3_hum) + random_x_3_hum + 6)[cluster_class_3_hum == 3],
                                             np.log10(m_star_3_hum[cluster_class_3_hum == 3]), c='royalblue', s=1)
        ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + random_x_12_hum + 6)[cluster_class_12_hum == 1],
                                             np.log10(m_star_12_hum)[cluster_class_12_hum == 1], c='darkorange', s=1)
        ax_hum[row_index, col_index].scatter((np.log10(age_12_hum) + random_x_12_hum + 6)[cluster_class_12_hum == 2],
                                             np.log10(m_star_12_hum)[cluster_class_12_hum == 2], c='forestgreen', s=1)


        # random dots
        random_x_12_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_12_ml))
        random_x_3_ml = np.random.uniform(low=-0.1, high=0.1, size=len(age_3_ml))

        ax_ml[row_index, col_index].plot(np.log10(age_mod) + 6, lower_lim_m_star, color='r', linewidth=1.2)
        ax_ml[row_index, col_index].scatter((np.log10(age_3_ml) + random_x_3_ml + 6)[cluster_class_3_ml == 3],
                                             np.log10(m_star_3_ml[cluster_class_3_ml == 3]), c='royalblue', s=1)
        ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + random_x_12_ml + 6)[cluster_class_12_ml == 1],
                                             np.log10(m_star_12_ml)[cluster_class_12_ml == 1], c='darkorange', s=1)
        ax_ml[row_index, col_index].scatter((np.log10(age_12_ml) + random_x_12_ml + 6)[cluster_class_12_ml == 2],
                                             np.log10(m_star_12_ml)[cluster_class_12_ml == 2], c='forestgreen', s=1)


        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_hum[row_index, col_index].add_artist(anchored_left)
        ax_hum[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                 direction='in', labelsize=fontsize)

        anchored_left = AnchoredText(target.upper()+'\nd='+str(dist)+' Mpc',  loc='upper left', borderpad=0.1,
                                     frameon=False, prop=dict(size=fontsize-4))
        ax_ml[row_index, col_index].add_artist(anchored_left)
        ax_ml[row_index, col_index].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True,
                                                direction='in', labelsize=fontsize)

        col_index += 1
        if col_index == 4:
            row_index += 1
            col_index = 0

    ax_hum[0, 0].set_xlim(5.7, 10.3)
    ax_hum[0, 0].set_ylim(1.9, 8.7)
    fig_hum.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
    fig_hum.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)
    ax_hum[4, 3].scatter([], [], c='darkorange', s=30, label='Class 1')
    ax_hum[4, 3].scatter([], [], c='forestgreen', s=30, label='Class 2')
    ax_hum[4, 3].scatter([], [], c='royalblue', s=30, label='Compact Associations')
    ax_hum[4, 3].legend(frameon=False, fontsize=fontsize-3)
    ax_hum[4, 3].axis('off')

    ax_hum[4, 3].axis('off')

    ax_ml[0, 0].set_xlim(5.7, 10.3)
    ax_ml[0, 0].set_ylim(1.9, 8.7)
    fig_ml.text(0.5, 0.08, 'log(Age/yr) + random.uniform(-0.1, 0.1)', ha='center', fontsize=fontsize)
    fig_ml.text(0.08, 0.5, 'log(M$_{*}$/M$_{\odot}$)', va='center', rotation='vertical', fontsize=fontsize)
    ax_ml[4, 3].scatter([], [], c='darkorange', s=30, label='Class 1')
    ax_ml[4, 3].scatter([], [], c='forestgreen', s=30, label='Class 2')
    ax_ml[4, 3].scatter([], [], c='royalblue', s=30, label='Compact Associations')
    ax_ml[4, 3].legend(frameon=False, fontsize=fontsize-3)
    ax_ml[4, 3].axis('off')


    fig_hum.subplots_adjust(wspace=0, hspace=0)
    fig_hum.savefig('plot_output/age_m_star_hum_2_%s.png' % catalog_version, bbox_inches='tight', dpi=300)
    # fig_hum.savefig('plot_output/age_m_star_hum_2.pdf', bbox_inches='tight', dpi=300)

    fig_ml.subplots_adjust(wspace=0, hspace=0)
    fig_ml.savefig('plot_output/age_m_star_ml_2_%s.png' % catalog_version, bbox_inches='tight', dpi=300)
    # fig_ml.savefig('plot_output/age_m_star_ml_2.pdf', bbox_inches='tight', dpi=300)

