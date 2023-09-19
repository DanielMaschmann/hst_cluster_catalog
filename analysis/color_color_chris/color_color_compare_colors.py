import numpy as np
from photometry_tools import helper_func as hf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
from dust_extinction.parameter_averages import CCM89
import astropy.units as u

from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde


hst_wfc3_uvis1_bands_mean_wave = {
            'F218W': 2231.14,
            'FQ232N': 2327.12,
            'F225W': 2377.24,
            'FQ243N': 2420.59,
            'F275W': 2718.36,
            'F280N': 2796.98,
            'F300X': 2867.82,
            'F336W': 3365.86,
            'F343N': 3438.50,
            'F373N': 3730.19,
            'FQ378N': 3792.78,
            'FQ387N': 3873.61,
            'F390M': 3898.62,
            'F390W': 3952.50,
            'F395N': 3955.38,
            'F410M': 4109.81,
            'FQ422M': 4219.70,
            'F438W': 4338.57,
            'FQ436N': 4367.41,
            'FQ437N': 4371.30,
            'G280': 4628.43,
            'F467M': 4683.55,
            'F469N': 4688.29,
            'F475W': 4827.71,
            'F487N': 4871.54,
            'FQ492N': 4933.83,
            'F502N': 5009.93,
            'F475X': 5076.23,
            'FQ508N': 5091.59,
            'F555W': 5388.55,
            'F547M': 5459.04,
            'FQ575N': 5756.92,
            'F606W': 5999.27,
            'F200LP': 6043.00,
            'FQ619N': 6198.49,
            'F621M': 6227.39,
            'F625W': 6291.29,
            'F631N': 6304.27,
            'FQ634N': 6349.37,
            'F645N': 6453.59,
            'F350LP': 6508.00,
            'F656N': 6561.54,
            'F657N': 6566.93,
            'F658N': 6585.64,
            'F665N': 6656.23,
            'FQ672N': 6717.13,
            'FQ674N': 6730.58,
            'F673N': 6766.27,
            'F680N': 6880.13,
            'F689M': 6885.92,
            'FQ727N': 7275.84,
            'FQ750N': 7502.54,
            'F763M': 7623.09,
            'F600LP': 7656.67,
            'F775W': 7683.41,
            'F814W': 8117.36,
            'F845M': 8449.34,
            'FQ889N': 8892.56,
            'FQ906N': 9058.19,
            'F850LP': 9207.49,
            'FQ924N': 9247.91,
            'FQ937N': 9372.90,
            'F953N': 9531.11,
        }

def plot_contours(ax, x, y, levels=None, legend=False, fontsize=13):

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


def density_with_points(ax, x, y, binx=None, biny=None, threshold=1, kernel_std=2.0, cmap='inferno', scatter_size=10):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    kernel = Gaussian2DKernel(x_stddev=kernel_std)
    hist = convolve(hist, kernel)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask
    hist[hist <= threshold] = np.nan

    cmap = cm.get_cmap(cmap)

    scatter_color = cmap(0)

    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap=cmap,
              interpolation='nearest', aspect='auto')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color=scatter_color, marker='.', s=scatter_size)
    ax.set_ylim(ax.get_ylim()[::-1])


def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

    model_ccm89 = CCM89(Rv=r_v)
    reddening1 = model_ccm89(wave1*u.micron) * r_v
    reddening2 = model_ccm89(wave2*u.micron) * r_v

    wave_v = 5388.55 * 1e-4
    reddening_v = model_ccm89(wave_v*u.micron) * r_v

    return (reddening1 - reddening2)*av/reddening_v


def plot_hst_reddening_vect(ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                        x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=False, fontsize=20, text_color='k', x_text_offset=0.1, y_text_offset=-0.3):

    nuv_wave = hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
    u_wave = hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
    b_wave = hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
    v_wave = hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
    i_wave = hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4

    x_wave_1 = locals()[x_color_1 + '_wave']
    x_wave_2 = locals()[x_color_2 + '_wave']
    y_wave_1 = locals()[y_color_1 + '_wave']
    y_wave_2 = locals()[y_color_2 + '_wave']

    color_ext_x = color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_val)
    color_ext_y = color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_val)

    slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

    angle_av_vector =  np.arctan(color_ext_y/color_ext_x) * 180/np.pi

    ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                xytext=(x_color_int, y_color_int), fontsize=fontsize,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

    if text:
        if isinstance(av_val, int):
            arrow_text = r'A$_{\rm V}$=%i mag' % av_val
        else:
            arrow_text = r'A$_{\rm V}$=%.1f mag' % av_val
        ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
                horizontalalignment='left', verticalalignment='bottom',
                transform_rotates_text=True, rotation_mode='anchor',
                rotation=angle_av_vector, fontsize=fontsize, color=text_color)


age_mod_sol = np.load('data/age_mod_sol.npy')
model_nuvb_sol = np.load('data/model_nuvb_sol.npy')
model_ub_sol = np.load('data/model_ub_sol.npy')
model_bv_sol = np.load('data/model_bv_sol.npy')
model_vi_sol = np.load('data/model_vi_sol.npy')

color_vi_ml = np.load('data/color_vi_ml.npy')
color_ub_ml = np.load('data/color_ub_ml.npy')
color_bv_ml = np.load('data/color_bv_ml.npy')
color_nuvb_ml = np.load('data/color_nuvb_ml.npy')
color_vi_err_ml = np.load('data/color_vi_err_ml.npy')
color_ub_err_ml = np.load('data/color_ub_err_ml.npy')
color_bv_err_ml = np.load('data/color_bv_err_ml.npy')
color_nuvb_err_ml = np.load('data/color_nuvb_err_ml.npy')
detect_nuv_ml = np.load('data/detect_nuv_ml.npy')
detect_u_ml = np.load('data/detect_u_ml.npy')
detect_b_ml = np.load('data/detect_b_ml.npy')
detect_v_ml = np.load('data/detect_v_ml.npy')
detect_i_ml = np.load('data/detect_i_ml.npy')
clcl_color_ml = np.load('data/clcl_color_ml.npy')


mask_class_1_ml = clcl_color_ml == 1
mask_class_2_ml = clcl_color_ml == 2
mask_class_3_ml = clcl_color_ml == 3

mask_detect_nuvbvi_ml = detect_nuv_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_ubvi_ml = detect_u_ml * detect_b_ml * detect_v_ml * detect_i_ml
mask_detect_bvvi_ml = detect_b_ml * detect_v_ml * detect_i_ml

x_lim_vi = (-0.7, 2.4)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)
n_bins = 190
kernal_std = 3.0

mask_good_colors_nuvbvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_nuvb_ml > (y_lim_nuvb[1] - 1)) & (color_nuvb_ml < (y_lim_nuvb[0] + 1)) &
                               mask_detect_nuvbvi_ml)
mask_good_colors_ubvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_ub_ml > (y_lim_ub[1] - 1)) & (color_ub_ml < (y_lim_ub[0] + 1)) &
                               mask_detect_ubvi_ml)
mask_good_colors_bvvi_ml = ((color_vi_ml > (x_lim_vi[0] - 1)) & (color_vi_ml < (x_lim_vi[1] + 1)) &
                               (color_bv_ml > (y_lim_bv[1] - 1)) & (color_bv_ml < (y_lim_bv[0] + 1)) &
                               mask_detect_bvvi_ml)


fig, ax = plt.subplots(ncols=3, nrows=3, sharex='all', sharey='row', figsize=(19, 19))
fontsize = 23


density_with_points(ax=ax[0, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                    y=color_nuvb_ml[mask_class_1_ml * mask_good_colors_nuvbvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[0, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                    y=color_nuvb_ml[mask_class_2_ml * mask_good_colors_nuvbvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[0, 2], x=color_vi_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                    y=color_nuvb_ml[mask_class_3_ml * mask_good_colors_nuvbvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)

density_with_points(ax=ax[1, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_1_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[1, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_2_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[1, 2], x=color_vi_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                    y=color_ub_ml[mask_class_3_ml * mask_good_colors_ubvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)

density_with_points(ax=ax[2, 0], x=color_vi_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                    y=color_bv_ml[mask_class_1_ml * mask_good_colors_bvvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[2, 1], x=color_vi_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                    y=color_bv_ml[mask_class_2_ml * mask_good_colors_bvvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)
density_with_points(ax=ax[2, 2], x=color_vi_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                    y=color_bv_ml[mask_class_3_ml * mask_good_colors_bvvi_ml],
                    binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins),
                    biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins),
                    kernel_std=kernal_std, cmap='viridis', scatter_size=20)

# plot model tracks
ax[0, 0].plot(model_vi_sol, model_nuvb_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10, label=r'BC03, Z$_{\odot}$')
ax[0, 1].plot(model_vi_sol, model_nuvb_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)
ax[0, 2].plot(model_vi_sol, model_nuvb_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)

ax[1, 0].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)
ax[1, 1].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)
ax[1, 2].plot(model_vi_sol, model_ub_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)

ax[2, 0].plot(model_vi_sol, model_bv_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)
ax[2, 1].plot(model_vi_sol, model_bv_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)
ax[2, 2].plot(model_vi_sol, model_bv_sol, color='tab:red', linewidth=4, linestyle='-', zorder=10)

# pot reddening vector
vi_int = 1.75
nuvb_int = -2.2
ub_int = -1.8
bv_int = -0.5
av_value = 1
plot_hst_reddening_vect(ax=ax[0, 0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
plot_hst_reddening_vect(ax=ax[0, 1], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
plot_hst_reddening_vect(ax=ax[0, 2], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

plot_hst_reddening_vect(ax=ax[1, 0], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
plot_hst_reddening_vect(ax=ax[1, 1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
plot_hst_reddening_vect(ax=ax[1, 2], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

plot_hst_reddening_vect(ax=ax[2, 0], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
plot_hst_reddening_vect(ax=ax[2, 1], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
plot_hst_reddening_vect(ax=ax[2, 2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

# add some ages
ax[0, 0].text(model_vi_sol[age_mod_sol == 1], model_nuvb_sol[age_mod_sol == 1]-0.1, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[0, 0].text(model_vi_sol[age_mod_sol == 5]-0.0, model_nuvb_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='top', color='darkred', fontsize=fontsize+3)
ax[0, 0].text(model_vi_sol[age_mod_sol == 10]+0.1, model_nuvb_sol[age_mod_sol == 10]-0.1, r'10 Myr',
                   horizontalalignment='left', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[0, 0].text(model_vi_sol[age_mod_sol == 100]-0.1, model_nuvb_sol[age_mod_sol == 100]+0.0, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', color='darkred', fontsize=fontsize+3)
ax[0, 0].text(model_vi_sol[age_mod_sol == 1000]-0.15, model_nuvb_sol[age_mod_sol == 1000]+0.1, r'1 Gyr',
                   horizontalalignment='right', verticalalignment='center', color='darkred', fontsize=fontsize+3)
ax[0, 0].text(model_vi_sol[age_mod_sol == 13750], model_nuvb_sol[age_mod_sol == 13750]+0.1, r'13.8 Gyr',
                   horizontalalignment='center', verticalalignment='top', color='darkred', fontsize=fontsize+3)


ax[1, 0].text(model_vi_sol[age_mod_sol == 1], model_ub_sol[age_mod_sol == 1]-0.1, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[1, 0].text(model_vi_sol[age_mod_sol == 5]+0.05, model_ub_sol[age_mod_sol == 5]+0.1, r'5 Myr',
                   horizontalalignment='right', verticalalignment='top', color='darkred', fontsize=fontsize+3)
ax[1, 0].text(model_vi_sol[age_mod_sol == 10]+0.1, model_ub_sol[age_mod_sol == 10]-0.1, r'10 Myr',
                   horizontalalignment='left', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[1, 0].text(model_vi_sol[age_mod_sol == 100]-0.1, model_ub_sol[age_mod_sol == 100]+0.0, r'100 Myr',
                   horizontalalignment='right', verticalalignment='center', color='darkred', fontsize=fontsize+3)
ax[1, 0].text(model_vi_sol[age_mod_sol == 1000]-0.15, model_ub_sol[age_mod_sol == 1000]+0.1, r'1 Gyr',
                   horizontalalignment='right', verticalalignment='top', color='darkred', fontsize=fontsize+3)
ax[1, 0].text(model_vi_sol[age_mod_sol == 13750], model_ub_sol[age_mod_sol == 13750]+0.1, r'13.8 Gyr',
                   horizontalalignment='center', verticalalignment='top', color='darkred', fontsize=fontsize+3)


ax[2, 0].text(model_vi_sol[age_mod_sol == 1], model_bv_sol[age_mod_sol == 1]-0.1, r'1 Myr',
                       horizontalalignment='center', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[2, 0].text(model_vi_sol[age_mod_sol == 5]-0.0, model_bv_sol[age_mod_sol == 5]+0.05, r'5 Myr',
                   horizontalalignment='right', verticalalignment='top', color='darkred', fontsize=fontsize+3)
ax[2, 0].text(model_vi_sol[age_mod_sol == 10]+0.1, model_bv_sol[age_mod_sol == 10]-0.1, r'10 Myr',
                   horizontalalignment='left', verticalalignment='bottom', color='darkred', fontsize=fontsize+3)
ax[2, 0].text(model_vi_sol[age_mod_sol == 100]-0.1, model_bv_sol[age_mod_sol == 100]+0.0, r'100 Myr',
                   horizontalalignment='right', verticalalignment='top', color='darkred', fontsize=fontsize+3)
ax[2, 0].text(model_vi_sol[age_mod_sol == 1000]-0.15, model_bv_sol[age_mod_sol == 1000]+0.1, r'1 Gyr',
                   horizontalalignment='right', verticalalignment='center', color='darkred', fontsize=fontsize+3)
ax[2, 0].text(model_vi_sol[age_mod_sol == 13750], model_bv_sol[age_mod_sol == 13750]+0.1, r'13.8 Gyr',
                   horizontalalignment='center', verticalalignment='top', color='darkred', fontsize=fontsize+3)




# add smoothing kernel size
xedges = np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins)
yedges = np.linspace(y_lim_nuvb[0], y_lim_nuvb[1], n_bins)
kernal_rad_width = (xedges[1] - xedges[0]) * kernal_std
kernal_rad_hight = (yedges[1] - yedges[0]) * kernal_std
# plot_kernel_std
ellipse = Ellipse(xy=(-0.5, 2.4), width=kernal_rad_width, height=kernal_rad_hight, angle=0, edgecolor='r', fc='None', lw=2)
ax[0, 1].add_patch(ellipse)
ax[0, 1].text(-0.4, 2.4, 'Smoothing', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)


ax[0, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'N=%i' % (sum(mask_class_1_ml * mask_good_colors_nuvbvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'N=%i' % (sum(mask_class_2_ml * mask_good_colors_nuvbvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_nuvb[0] + (y_lim_nuvb[1]-y_lim_nuvb[0])*0.05,
              'N=%i' % (sum(mask_class_3_ml * mask_good_colors_nuvbvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[1, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_1_ml * mask_good_colors_ubvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_2_ml * mask_good_colors_ubvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[1, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_ub[0] + (y_lim_ub[1]-y_lim_ub[0])*0.05,
              'N=%i' % (sum(mask_class_3_ml * mask_good_colors_ubvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[2, 0].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_1_ml * mask_good_colors_bvvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 1].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_2_ml * mask_good_colors_bvvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[2, 2].text(x_lim_vi[0] + (x_lim_vi[1]-x_lim_vi[0])*0.05, y_lim_bv[0] + (y_lim_bv[1]-y_lim_bv[0])*0.05,
              'N=%i' % (sum(mask_class_3_ml * mask_good_colors_bvvi_ml)),
              horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[0, 0].legend(frameon=False, loc=3, bbox_to_anchor=(0, 0.05), fontsize=fontsize)

ax[0, 0].set_title('ML Class 1', fontsize=fontsize)
ax[0, 1].set_title('ML Class 2', fontsize=fontsize)
ax[0, 2].set_title('ML Compact Association', fontsize=fontsize)

ax[2, 0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2, 2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)

ax[0, 0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')',labelpad=25, fontsize=fontsize)
ax[1, 0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')',labelpad=25, fontsize=fontsize)
ax[2, 0].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)



ax[0, 0].set_xlim(x_lim_vi)
ax[0, 1].set_xlim(x_lim_vi)
ax[0, 2].set_xlim(x_lim_vi)
ax[0, 0].set_ylim(y_lim_nuvb)
ax[0, 1].set_ylim(y_lim_nuvb)
ax[0, 2].set_ylim(y_lim_nuvb)

ax[1, 0].set_xlim(x_lim_vi)
ax[1, 1].set_xlim(x_lim_vi)
ax[1, 2].set_xlim(x_lim_vi)
ax[1, 0].set_ylim(y_lim_ub)
ax[1, 1].set_ylim(y_lim_ub)
ax[1, 2].set_ylim(y_lim_ub)

ax[2, 0].set_xlim(x_lim_vi)
ax[2, 1].set_xlim(x_lim_vi)
ax[2, 2].set_xlim(x_lim_vi)
ax[2, 0].set_ylim(y_lim_bv)
ax[2, 1].set_ylim(y_lim_bv)
ax[2, 2].set_ylim(y_lim_bv)


ax[0, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[0, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2, 2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)





fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig('plot_output/color_color_compare.png')
fig.savefig('plot_output/color_color_compare.pdf')
fig.clf()
plt.close()

exit()

