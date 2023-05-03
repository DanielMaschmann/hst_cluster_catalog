import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve




def contours_with_points(ax, x, y, binx=None, biny=None, threshold=1, levels=None):

    if levels is None:
        levels = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    if binx is None:
        binx = np.linspace(-1.5, 2.5, 200)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 200)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])

    kernel = Gaussian2DKernel(x_stddev=3.)
    hist = convolve(hist, kernel)


    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask


    cs = ax.contour(xmesh, ymesh, hist.T, levels=levels, linewidths=2, origin='lower')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.')
    ax.set_ylim(ax.get_ylim()[::-1])



def density_with_points(ax, x, y, binx=None, biny=None, threshold=1):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    kernel = Gaussian2DKernel(x_stddev=2.0)
    hist = convolve(hist, kernel)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask

    hist[hist <= threshold] = np.nan
    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap='inferno', interpolation='nearest')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.')
    ax.set_ylim(ax.get_ylim()[::-1])

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
        galaxy_name = 'ngc0628'
    else:
        galaxy_name = target
    dist_list.append(catalog_access.dist_dict[galaxy_name]['dist'])
sort = np.argsort(dist_list)
target_list = np.array(target_list)[sort]
dist_list = np.array(dist_list)[sort]

catalog_access.load_hst_cc_list(target_list=target_list)
catalog_access.load_hst_cc_list(target_list=target_list, cluster_class='class3')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml')
catalog_access.load_hst_cc_list(target_list=target_list, classify='ml', cluster_class='class3')


color_vi_ml = np.array([])
color_ub_ml = np.array([])
clcl_color_ml = np.array([])
clcl_qual_color_ml = np.array([])

for index in range(0, len(target_list)):
    target = target_list[index]
    dist = dist_list[index]
    print('target ', target, 'dist ', dist)

    cluster_class_ml_12 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml')
    cluster_class_qual_ml_12 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml')
    color_ub_ml_12 = catalog_access.get_hst_color_ub(target=target, classify='ml')
    color_vi_ml_12 = catalog_access.get_hst_color_vi(target=target, classify='ml')
    cluster_class_ml_3 = catalog_access.get_hst_cc_class_ml_vgg(target=target, classify='ml', cluster_class='class3')
    cluster_class_qual_ml_3 = catalog_access.get_hst_cc_class_ml_vgg_qual(target=target, classify='ml', cluster_class='class3')
    color_ub_ml_3 = catalog_access.get_hst_color_ub(target=target, classify='ml', cluster_class='class3')
    color_vi_ml_3 = catalog_access.get_hst_color_vi(target=target, classify='ml', cluster_class='class3')

    color_vi_ml = np.concatenate([color_vi_ml, color_vi_ml_12, color_vi_ml_3])
    color_ub_ml = np.concatenate([color_ub_ml, color_ub_ml_12, color_ub_ml_3])
    clcl_color_ml = np.concatenate([clcl_color_ml, cluster_class_ml_12, cluster_class_ml_3])
    clcl_qual_color_ml = np.concatenate([clcl_qual_color_ml, cluster_class_qual_ml_12, cluster_class_qual_ml_3])


mask_good_colors = (color_vi_ml > -2) & (color_vi_ml < 3) & (color_ub_ml > -3) & (color_ub_ml < 2)
mask_class_1 = (clcl_color_ml == 1) #& (clcl_qual_color_ml >= 0.9)
mask_class_2 = (clcl_color_ml == 2) #& (clcl_qual_color_ml >= 0.9)
mask_class_3 = (clcl_color_ml == 3) #& (clcl_qual_color_ml >= 0.9)

print('class_1 ', sum(clcl_color_ml == 1))
print('class_2 ', sum(clcl_color_ml == 2))
print('class_3 ', sum(clcl_color_ml == 3))

print('class_1 ', sum(mask_class_1))
print('class_2 ', sum(mask_class_2))
print('class_3 ', sum(mask_class_3))

print('class_1 ', sum(mask_class_1 * mask_good_colors))
print('class_2 ', sum(mask_class_2 * mask_good_colors))
print('class_3 ', sum(mask_class_3 * mask_good_colors))




# plotting very standard
fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(25, 9))
fontsize = 17


ax[0].scatter(color_vi_ml[mask_class_1], color_ub_ml[mask_class_1], marker='.', color='k', alpha=0.3)
ax[1].scatter(color_vi_ml[mask_class_2], color_ub_ml[mask_class_2], marker='.', color='k', alpha=0.3)
ax[2].scatter(color_vi_ml[mask_class_1 + mask_class_2], color_ub_ml[mask_class_1 + mask_class_2], marker='.', color='k', alpha=0.3)
ax[3].scatter(color_vi_ml[mask_class_3], color_ub_ml[mask_class_3], marker='.', color='k', alpha=0.3)


ax[0].set_title('Class 1 ML', fontsize=fontsize)
ax[1].set_title('Class 2 ML', fontsize=fontsize)
ax[2].set_title('Class 1|2 ML', fontsize=fontsize)
ax[3].set_title('Class 3 ML', fontsize=fontsize)

ax[0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1)), fontsize=fontsize)
ax[1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2)), fontsize=fontsize)
ax[2].text(-1, 1.7, 'N=%i' % (sum(mask_class_1 + mask_class_2)), fontsize=fontsize)
ax[3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3)), fontsize=fontsize)


ax[0].set_ylim(1.9, -2.2)
ax[0].set_xlim(-1.2, 2.4)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_ml_first_view_standard.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_ml_first_view_standard.pdf', bbox_inches='tight', dpi=300)
fig.clf()



# plotting contours
fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(25, 9))
fontsize = 17

contours_with_points(ax=ax[0], x=color_vi_ml[mask_class_1 * mask_good_colors],
                     y=color_ub_ml[mask_class_1 * mask_good_colors],
                     levels=[1, 1.5, 2, 3, 4, 5])
contours_with_points(ax=ax[1], x=color_vi_ml[mask_class_2 * mask_good_colors],
                     y=color_ub_ml[mask_class_2 * mask_good_colors],
                     levels=[1, 2, 3, 5, 7])
contours_with_points(ax=ax[2], x=color_vi_ml[(mask_class_1 + mask_class_2) * mask_good_colors],
                     y=color_ub_ml[(mask_class_1 + mask_class_2) * mask_good_colors],
                     levels=[1, 3, 5, 8, 10])
contours_with_points(ax=ax[3], x=color_vi_ml[mask_class_3 * mask_good_colors],
                     y=color_ub_ml[mask_class_3 * mask_good_colors],
                     levels=[1, 3, 5, 10, 15, 20])



ax[0].set_title('Class 1 ML', fontsize=fontsize)
ax[1].set_title('Class 2 ML', fontsize=fontsize)
ax[2].set_title('Class 1|2 ML', fontsize=fontsize)
ax[3].set_title('Class 3 ML', fontsize=fontsize)

ax[0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1)), fontsize=fontsize)
ax[1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2)), fontsize=fontsize)
ax[2].text(-1, 1.7, 'N=%i' % (sum(mask_class_1 + mask_class_2)), fontsize=fontsize)
ax[3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3)), fontsize=fontsize)


ax[0].set_ylim(1.9, -2.2)
ax[0].set_xlim(-1.2, 2.4)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_ml_first_view_contours.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_ml_first_view_contours.pdf', bbox_inches='tight', dpi=300)
fig.clf()



# plotting density
fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(25, 9))
fontsize = 17

density_with_points(ax=ax[0], x=color_vi_ml[mask_class_1 * mask_good_colors],
                    y=color_ub_ml[mask_class_1 * mask_good_colors])
density_with_points(ax=ax[1], x=color_vi_ml[mask_class_2 * mask_good_colors],
                    y=color_ub_ml[mask_class_2 * mask_good_colors])
density_with_points(ax=ax[2], x=color_vi_ml[(mask_class_1 + mask_class_2) * mask_good_colors],
                    y=color_ub_ml[(mask_class_1 + mask_class_2) * mask_good_colors])
density_with_points(ax=ax[3], x=color_vi_ml[mask_class_3 * mask_good_colors],
                    y=color_ub_ml[mask_class_3 * mask_good_colors])

ax[0].set_title('Class 1 ML', fontsize=fontsize)
ax[1].set_title('Class 2 ML', fontsize=fontsize)
ax[2].set_title('Class 1|2 ML', fontsize=fontsize)
ax[3].set_title('Class 3 ML', fontsize=fontsize)

ax[0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1)), fontsize=fontsize)
ax[1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2)), fontsize=fontsize)
ax[2].text(-1, 1.7, 'N=%i' % (sum(mask_class_1 + mask_class_2)), fontsize=fontsize)
ax[3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3)), fontsize=fontsize)

ax[0].set_ylim(1.9, -2.2)
ax[0].set_xlim(-1.2, 2.4)
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)

ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_ml_first_view_density.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_ml_first_view_density.pdf', bbox_inches='tight', dpi=300)
fig.clf()




