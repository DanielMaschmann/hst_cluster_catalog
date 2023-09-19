import numpy as np
import photometry_tools
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from xgaltool import analysis_tools
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde

def contours(ax, x, y, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    origin = 'lower'
    cs = ax.contour(xi, yi, zi, levels=levels,
                    linewidths=(2,),
                    origin=origin)

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)



age_mod_sol = np.load('data_output/age_mod_sol.npy')
model_vi_sol = np.load('data_output/model_vi_sol.npy')
model_ub_sol = np.load('data_output/model_ub_sol.npy')
age_mod_sol50 = np.load('data_output/age_mod_sol50.npy')
model_vi_sol50 = np.load('data_output/model_vi_sol50.npy')
model_ub_sol50 = np.load('data_output/model_ub_sol50.npy')

color_vi_hum = np.load('data_output/color_vi_hum.npy')
color_ub_hum = np.load('data_output/color_ub_hum.npy')
color_vi_err_hum = np.load('data_output/color_vi_err_hum.npy')
color_ub_err_hum = np.load('data_output/color_ub_err_hum.npy')
detect_vi_hum = np.load('data_output/detect_vi_hum.npy')
detect_ub_hum = np.load('data_output/detect_ub_hum.npy')
clcl_color_hum = np.load('data_output/clcl_color_hum.npy')

color_vi_ml = np.load('data_output/color_vi_ml.npy')
color_ub_ml = np.load('data_output/color_ub_ml.npy')
color_bv_ml = np.load('data_output/color_bv_ml.npy')
color_vi_err_ml = np.load('data_output/color_vi_err_ml.npy')
color_ub_err_ml = np.load('data_output/color_ub_err_ml.npy')
color_bv_err_ml = np.load('data_output/color_bv_err_ml.npy')
detect_vi_ml = np.load('data_output/detect_vi_ml.npy')
detect_ub_ml = np.load('data_output/detect_ub_ml.npy')
clcl_color_ml = np.load('data_output/clcl_color_ml.npy')
clcl_qual_color_ml = np.load('data_output/clcl_qual_color_ml.npy')


mask_class_1_hum = clcl_color_hum == 1
mask_class_2_hum = clcl_color_hum == 2
mask_class_3_hum = clcl_color_hum == 3

mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_ubvi_hum = detect_vi_hum * detect_ub_hum
mask_detect_ubvi_ml = detect_vi_ml * detect_ub_ml

mask_good_colors_hum = ((color_vi_hum > -1.5) & (color_vi_hum < 2.5) &
                        (color_ub_hum > -3) & (color_ub_hum < 1.5)) * mask_detect_ubvi_hum
mask_good_colors_ml = ((color_vi_ml > -1.5) & (color_vi_ml < 2.5) &
                        (color_ub_ml > -3) & (color_ub_ml < 1.5)) * mask_detect_ubvi_ml


fig, ax = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(25, 13))
fontsize = 17


ax_shadow = ax[0, 0].twinx()
ax_shadow.plot(np.NaN, np.NaN, color='red', linewidth=3, linestyle='--', zorder=10, label='BC03, Z=Z$_{\odot}$')
ax_shadow.plot(np.NaN, np.NaN, color='gray', linewidth=3, linestyle='--', zorder=10, label='BC03, Z=Z$_{\odot}$/50')
ax_shadow.get_yaxis().set_visible(False)

ax_shadow.legend(loc=4, frameon=False, fontsize=fontsize - 2)


contours(ax=ax[0, 0], x=color_vi_hum[mask_class_1_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_1_hum * mask_good_colors_hum], legend=True, fontsize=fontsize - 4)
contours(ax=ax[0, 1], x=color_vi_hum[mask_class_2_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_2_hum * mask_good_colors_hum])
contours(ax=ax[0, 2], x=color_vi_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum],
                     y=color_ub_hum[(mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum])
contours(ax=ax[0, 3], x=color_vi_hum[mask_class_3_hum * mask_good_colors_hum],
                     y=color_ub_hum[mask_class_3_hum * mask_good_colors_hum])

contours(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_1_ml * mask_good_colors_ml])
contours(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_2_ml * mask_good_colors_ml])
contours(ax=ax[1, 2], x=color_vi_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml],
                     y=color_ub_ml[(mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml])
contours(ax=ax[1, 3], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ml],
                     y=color_ub_ml[mask_class_3_ml * mask_good_colors_ml])


ax[0, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[0, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)

ax[1, 0].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 1].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 2].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)
ax[1, 3].plot(model_vi_sol, model_ub_sol, color='red', linewidth=3, linestyle='--', zorder=10)

ax[0, 0].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 1].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 2].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[0, 3].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)

ax[1, 0].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 1].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 2].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)
ax[1, 3].plot(model_vi_sol50, model_ub_sol50, color='gray', linewidth=3, linestyle='--', zorder=10)

ax[0, 0].text(-1, 1.45, 'Class 1 Human', fontsize=fontsize)
ax[0, 1].text(-1, 1.45, 'Class 2 Human', fontsize=fontsize)
ax[0, 2].text(-1, 1.45, 'Class 1|2 Human', fontsize=fontsize)
ax[0, 3].text(-1, 1.45, 'Class 3 Human', fontsize=fontsize)

ax[0, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_hum * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_hum * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_hum + mask_class_2_hum) * mask_good_colors_hum)), fontsize=fontsize)
ax[0, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_hum * mask_good_colors_hum)), fontsize=fontsize)

ax[1, 0].text(-1, 1.45, 'Class 1 ML', fontsize=fontsize)
ax[1, 1].text(-1, 1.45, 'Class 2 ML', fontsize=fontsize)
ax[1, 2].text(-1, 1.45, 'Class 1|2 ML', fontsize=fontsize)
ax[1, 3].text(-1, 1.45, 'Class 3 ML', fontsize=fontsize)

ax[1, 0].text(-1, 1.7, 'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 1].text(-1, 1.7, 'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 2].text(-1, 1.7, 'N=%i' % (sum((mask_class_1_ml + mask_class_2_ml) * mask_good_colors_ml)), fontsize=fontsize)
ax[1, 3].text(-1, 1.7, 'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ml)), fontsize=fontsize)


ax[0, 0].set_ylim(1.9, -2.2)
ax[0, 0].set_xlim(-1.2, 2.4)

ax[1, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1, 3].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 3].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('plot_output/ub_vi_overview.png', bbox_inches='tight', dpi=300)
fig.savefig('plot_output/ub_vi_overview.pdf', bbox_inches='tight', dpi=300)
fig.clf()

