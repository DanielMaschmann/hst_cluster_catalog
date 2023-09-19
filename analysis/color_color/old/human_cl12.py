import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse


def density_with_points(ax, x, y, binx=None, biny=None, threshold=1, kernel_std=2.0, save=False, save_name=''):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    if save:
        np.save('data_output/binx.npy', binx)
        np.save('data_output/biny.npy', biny)
        np.save('data_output/hist_%s_un_smoothed.npy' % save_name, hist)

    kernel = Gaussian2DKernel(x_stddev=kernel_std)
    hist = convolve(hist, kernel)

    if save:
        np.save('data_output/hist_%s_smoothed.npy' % save_name, hist)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask

    hist[hist <= threshold] = np.nan
    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap='inferno', interpolation='nearest')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color='k', marker='.', s=100)
    ax.set_ylim(ax.get_ylim()[::-1])



age_mod_sol = np.load('data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('data_output/model_ub_sol.npy')
model_bv_sol = np.load('data_output/model_bv_sol.npy')
model_vi_sol = np.load('data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('data_output/model_bv_sol50.npy')
model_vi_sol50 = np.load('data_output/model_vi_sol50.npy')

color_vi_hum = np.load('data_output/color_vi_hum.npy')
color_ub_hum = np.load('data_output/color_ub_hum.npy')
color_bv_hum = np.load('data_output/color_bv_hum.npy')
color_nuvu_hum = np.load('data_output/color_nuvu_hum.npy')
color_nuvb_hum = np.load('data_output/color_nuvb_hum.npy')
color_vi_err_hum = np.load('data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('data_output/color_ub_err_hum.npy')
color_bv_err_hum = np.load('data_output/color_bv_err_hum.npy')
color_nuvu_err_hum = np.load('data_output/color_nuvu_err_hum.npy')
color_nuvb_err_hum = np.load('data_output/color_nuvb_err_hum.npy')
detect_vi_hum = np.load('data_output/detect_vi_hum.npy')
detect_ub_hum = np.load('data_output/detect_ub_hum.npy')
detect_bv_hum = np.load('data_output/detect_bv_hum.npy')
detect_nuvu_hum = np.load('data_output/detect_nuvu_hum.npy')
detect_nuvb_hum = np.load('data_output/detect_nuvb_hum.npy')
clcl_color_hum = np.load('data_output/clcl_color_hum.npy')
age_hum = np.load('data_output/age_hum.npy')
ebv_hum = np.load('data_output/ebv_hum.npy')
color_vi_ml = np.load('data_output/color_vi_ml.npy')
color_ub_ml = np.load('data_output/color_ub_ml.npy')
color_bv_ml = np.load('data_output/color_bv_ml.npy')
color_nuvu_ml = np.load('data_output/color_nuvu_ml.npy')
color_nuvb_ml = np.load('data_output/color_nuvb_ml.npy')
color_vi_err_ml = np.load('data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('data_output/color_bv_err_ml.npy')
color_nuvu_err_ml = np.load('data_output/color_nuvu_err_ml.npy')
color_nuvb_err_ml = np.load('data_output/color_nuvb_err_ml.npy')
detect_vi_ml = np.load('data_output/detect_vi_ml.npy')
detect_ub_ml = np.load('data_output/detect_ub_ml.npy')
detect_bv_ml = np.load('data_output/detect_bv_ml.npy')
detect_nuvu_ml = np.load('data_output/detect_nuvu_ml.npy')
detect_nuvb_ml = np.load('data_output/detect_nuvb_ml.npy')
clcl_color_ml = np.load('data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('data_output/clcl_qual_color_ml.npy')
age_ml = np.load('data_output/age_ml.npy')
ebv_ml = np.load('data_output/ebv_ml.npy')
mag_mask_ml = np.load('data_output/mag_mask_ml.npy')

mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_hum = detect_vi_hum * detect_ub_hum
mask_detect_ubvi_ml = detect_vi_ml * detect_ub_ml

mask_good_colors_ubvi_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                        (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
mask_good_colors_ubvi_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                        (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml


fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(13, 13))
fontsize = 25

ax.plot(model_vi_sol, model_ub_sol, color='cyan', linewidth=4, linestyle='-', alpha=1, label=r'BC03, Z$_{\odot}$')

kernal_std = 4.0
xedges = np.linspace(-1.5, 2.5, 190)
kernal_rad = (xedges[1] - xedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.97, 1.7), width=kernal_rad, height=kernal_rad, angle=0, edgecolor='r', fc='None', lw=2)
ax.add_patch(ellipse)
ax.text(-0.75, 1.7, 'Smoothing kernel', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


density_with_points(ax=ax, x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                    y=color_ub_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_ubvi_hum],
                    threshold=10000,
                    kernel_std=kernal_std)
ax.plot(np.nan, np.nan, color='white', label='N=%i' % (sum((mask_class_1_hum + mask_class_2_hum))))

ax.set_title('Human class 1|2', fontsize=fontsize)
ax.legend(frameon=False, loc=3, bbox_to_anchor=[0.0, 0.05], fontsize=fontsize)

ax.set_ylim(1.9, -2.2)
ax.set_xlim(-1.2, 2.4)

ax.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig('plot_output/ub_vi_human_cl12_no_heat_map.png')
fig.savefig('plot_output/ub_vi_human_cl12_no_heat_map.pdf')
fig.clf()
plt.close()




exit()


fig, ax = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True, figsize=(22.2, 19))
fontsize = 17

ax[0, 0].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol, model_bv_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[0, 0].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol50, model_bv_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

kernal_std = 4.0
xedges = np.linspace(-1.5, 2.5, 190)
kernal_rad = (xedges[1] - xedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.95, 1.7), width=kernal_rad, height=kernal_rad, angle=0, edgecolor='r', fc='None', lw=2)
ax[0, 0].add_patch(ellipse)
ax[0, 0].text(-0.8, 1.7, 'Smoothing kernel', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

biny = np.linspace(-1.0, 2.0, 190)

density_with_points(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_hum],
                     y=color_bv_hum[mask_class_1_hum * mask_good_colors_hum], biny=biny, kernel_std=1.0, save=False, save_name='c1_hum')
density_with_points(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_hum],
                     y=color_bv_hum[mask_class_2_hum * mask_good_colors_hum], biny=biny, kernel_std=1.0, save=False, save_name='c2_hum')
density_with_points(ax=ax[0, 2], x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum],
                     y=color_bv_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum], biny=biny, kernel_std=1.0)
density_with_points(ax=ax[0, 3], x=color_vi_hum[mask_class_3_hum * mask_good_colors_hum],
                     y=color_bv_hum[mask_class_3_hum * mask_good_colors_hum], biny=biny, kernel_std=1.0, save=False, save_name='c3_hum')

density_with_points(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml],
                     y=color_bv_ml[mask_class_1_ml * mask_good_colors_ml], biny=biny, kernel_std=1.0, save=False, save_name='c1_ml')
density_with_points(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml],
                     y=color_bv_ml[mask_class_2_ml * mask_good_colors_ml], biny=biny, kernel_std=1.0, save=False, save_name='c2_ml')
density_with_points(ax=ax[1, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                     y=color_bv_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml], biny=biny, kernel_std=1.0)
density_with_points(ax=ax[1, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml],
                     y=color_bv_ml[mask_class_3_ml * mask_good_colors_ml], biny=biny, kernel_std=1.0, save=False, save_name='c3_ml')

density_with_points(ax=ax[2, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml * mag_mask_ml],
                     y=color_bv_ml[mask_class_1_ml * mask_good_colors_ml * mag_mask_ml], biny=biny, kernel_std=1.0, save=False, save_name='c1_ml_mag_cut')
density_with_points(ax=ax[2, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml * mag_mask_ml],
                     y=color_bv_ml[mask_class_2_ml * mask_good_colors_ml * mag_mask_ml], biny=biny, kernel_std=1.0, save=False, save_name='c2_ml_mag_cut')
density_with_points(ax=ax[2, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml * mag_mask_ml],
                     y=color_bv_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml * mag_mask_ml], biny=biny, kernel_std=1.0)
density_with_points(ax=ax[2, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml * mag_mask_ml],
                     y=color_bv_ml[mask_class_3_ml * mask_good_colors_ml * mag_mask_ml], biny=biny, kernel_std=1.0, save=False, save_name='c3_ml_mag_cut')



ax[0, 0].set_title('Class 1', fontsize=fontsize)
ax[0, 1].set_title('Class 2', fontsize=fontsize)
ax[0, 2].set_title('Class 1|2', fontsize=fontsize)
ax[0, 3].set_title('Compact associations', fontsize=fontsize)

ax[0, 0].text(-1, 1.25, 'Human', fontsize=fontsize)
ax[0, 1].text(-1, 1.25, 'Human', fontsize=fontsize)
ax[0, 2].text(-1, 1.25, 'Human', fontsize=fontsize)
ax[0, 3].text(-1, 1.25, 'Human', fontsize=fontsize)

ax[0, 0].text(-1, 1.5, 'N=%i' % (sum(mask_class_1_hum)), fontsize=fontsize)
ax[0, 1].text(-1, 1.5, 'N=%i' % (sum(mask_class_2_hum)), fontsize=fontsize)
ax[0, 2].text(-1, 1.5, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum))), fontsize=fontsize)
ax[0, 3].text(-1, 1.5, 'N=%i' % (sum(mask_class_3_hum)), fontsize=fontsize)

ax[1, 0].text(-1, 1.25, 'ML', fontsize=fontsize)
ax[1, 1].text(-1, 1.25, 'ML', fontsize=fontsize)
ax[1, 2].text(-1, 1.25, 'ML', fontsize=fontsize)
ax[1, 3].text(-1, 1.25, 'ML', fontsize=fontsize)

ax[1, 0].text(-1, 1.5, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 1].text(-1, 1.5, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 2].text(-1, 1.5, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 3].text(-1, 1.5, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)


ax[2, 0].text(-1, 1.25, 'ML, V-band cut', fontsize=fontsize)
ax[2, 1].text(-1, 1.25, 'ML, V-band cut', fontsize=fontsize)
ax[2, 2].text(-1, 1.25, 'ML, V-band cut', fontsize=fontsize)
ax[2, 3].text(-1, 1.25, 'ML, V-band cut', fontsize=fontsize)

ax[2, 0].text(-1, 1.5, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml * mag_mask_ml)), fontsize=fontsize)
ax[2, 1].text(-1, 1.5, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml * mag_mask_ml)), fontsize=fontsize)
ax[2, 2].text(-1, 1.5, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml * mag_mask_ml)), fontsize=fontsize)
ax[2, 3].text(-1, 1.5, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml * mag_mask_ml)), fontsize=fontsize)


ax[0, 0].set_ylim(1.9, -1.2)
ax[0, 0].set_xlim(-1.2, 2.4)

ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
ax[1, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)
ax[2, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig('plot_output/bv_vi_compare_density.png')
fig.savefig('plot_output/bv_vi_compare_density.pdf')
fig.clf()
plt.close()




exit()


fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(25, 13))
fontsize = 17

ax[0, 0].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol, model_nuvb_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[0, 0].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol50, model_nuvb_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)


density_with_points(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_hum],
                     y=color_nuvb_hum[mask_class_1_hum * mask_good_colors_hum],
                     biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_hum],
                     y=color_nuvb_hum[mask_class_2_hum * mask_good_colors_hum], biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[0, 2], x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum],
                     y=color_nuvb_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum], biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[0, 3], x=color_vi_hum[mask_class_3_hum * mask_good_colors_hum],
                     y=color_nuvb_hum[mask_class_3_hum * mask_good_colors_hum], biny=np.linspace(-2.5, 2.3, 190))

density_with_points(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml],
                     y=color_nuvb_ml[mask_class_1_ml * mask_good_colors_ml], biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml],
                     y=color_nuvb_ml[mask_class_2_ml * mask_good_colors_ml], biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[1, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                     y=color_nuvb_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml], biny=np.linspace(-2.5, 2.3, 190))
density_with_points(ax=ax[1, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml],
                     y=color_nuvb_ml[mask_class_3_ml * mask_good_colors_ml], biny=np.linspace(-2.5, 2.3, 190))

ax[0, 0].text(-1, 1.45, 'Class 1 Human', fontsize=fontsize)
ax[0, 1].text(-1, 1.45, 'Class 2 Human', fontsize=fontsize)
ax[0, 2].text(-1, 1.45, 'Class 1|2 Human', fontsize=fontsize)
ax[0, 3].text(-1, 1.45, 'Class 3 Human', fontsize=fontsize)

ax[0, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_hum)), fontsize=fontsize)
ax[0, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_hum)), fontsize=fontsize)
ax[0, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum))), fontsize=fontsize)
ax[0, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_hum)), fontsize=fontsize)

ax[1, 0].text(-1, 1.45, 'Class 1 ML', fontsize=fontsize)
ax[1, 1].text(-1, 1.45, 'Class 2 ML', fontsize=fontsize)
ax[1, 2].text(-1, 1.45, 'Class 1|2 ML', fontsize=fontsize)
ax[1, 3].text(-1, 1.45, 'Class 3 ML', fontsize=fontsize)

ax[1, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)


ax[0, 0].set_ylim(1.9, -2.7)
ax[0, 0].set_xlim(-1.2, 2.4)

ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/nuvb_vi_compare_density.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/nuvb_vi_compare_density.pdf', bbox_inches='tight', dpi=300)
fig.clf()
plt.close()



fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(25, 13))
fontsize = 17

ax[0, 0].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol, model_nuvu_sol, color='red', linewidth=2, linestyle='--', alpha=0.9)

ax[0, 0].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 1].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 2].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[0, 3].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)

ax[1, 0].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 1].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 2].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)
ax[1, 3].plot(model_vi_sol50, model_nuvu_sol50, color='gray', linewidth=2, linestyle='--', alpha=0.9)


density_with_points(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_hum],
                     y=color_nuvu_hum[mask_class_1_hum * mask_good_colors_hum])
density_with_points(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_hum],
                     y=color_nuvu_hum[mask_class_2_hum * mask_good_colors_hum])
density_with_points(ax=ax[0, 2], x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum],
                     y=color_nuvu_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum])
density_with_points(ax=ax[0, 3], x=color_vi_hum[mask_class_3_hum * mask_good_colors_hum],
                     y=color_nuvu_hum[mask_class_3_hum * mask_good_colors_hum])

density_with_points(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml],
                     y=color_nuvu_ml[mask_class_1_ml * mask_good_colors_ml])
density_with_points(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml],
                     y=color_nuvu_ml[mask_class_2_ml * mask_good_colors_ml])
density_with_points(ax=ax[1, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                     y=color_nuvu_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml])
density_with_points(ax=ax[1, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml],
                     y=color_nuvu_ml[mask_class_3_ml * mask_good_colors_ml])

ax[0, 0].text(-1, 1.45, 'Class 1 Human', fontsize=fontsize)
ax[0, 1].text(-1, 1.45, 'Class 2 Human', fontsize=fontsize)
ax[0, 2].text(-1, 1.45, 'Class 1|2 Human', fontsize=fontsize)
ax[0, 3].text(-1, 1.45, 'Class 3 Human', fontsize=fontsize)

ax[0, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_hum)), fontsize=fontsize)
ax[0, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_hum)), fontsize=fontsize)
ax[0, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum))), fontsize=fontsize)
ax[0, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_hum)), fontsize=fontsize)

ax[1, 0].text(-1, 1.45, 'Class 1 ML', fontsize=fontsize)
ax[1, 1].text(-1, 1.45, 'Class 2 ML', fontsize=fontsize)
ax[1, 2].text(-1, 1.45, 'Class 1|2 ML', fontsize=fontsize)
ax[1, 3].text(-1, 1.45, 'Class 3 ML', fontsize=fontsize)

ax[1, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)


ax[0, 0].set_ylim(1.9, -1.4)
ax[0, 0].set_xlim(-1.2, 2.4)

ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('NUV (F275W) - U (F336W)', fontsize=fontsize)
ax[1, 0].set_ylabel('NUV (F275W) - U (F336W)', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/nuvu_vi_compare_density.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/nuvu_vi_compare_density.pdf', bbox_inches='tight', dpi=300)
fig.clf()

