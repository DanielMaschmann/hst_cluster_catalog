import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from astropy.convolution import Gaussian2DKernel, convolve
from lmfit import Model, Parameters
from scipy.interpolate import interp1d


def gauss(x, amp, mu, sig):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def two_gauss(x, amp_1, mu_1, sig_1, amp_2, mu_2, sig_2):
    return amp_1 * np.exp(-(x - mu_1) ** 2 / (2 * sig_1 ** 2)) + amp_2 * np.exp(-(x - mu_2) ** 2 / (2 * sig_2 ** 2))


def three_gauss(x, amp_1, mu_1, sig_1, amp_2, mu_2, sig_2, amp_3, mu_3, sig_3):
    return (amp_1 * np.exp(-(x - mu_1) ** 2 / (2 * sig_1 ** 2)) +
            amp_2 * np.exp(-(x - mu_2) ** 2 / (2 * sig_2 ** 2)) +
            amp_3 * np.exp(-(x - mu_3) ** 2 / (2 * sig_3 ** 2)))


def solve_lin(x1, y1, phi):
    slop = - 1/np.tan(phi * np.pi/180)
    inter = y1 + x1/np.tan(phi * np.pi/180)
    return slop, inter

def get_projected_dist(x_val, y_val, x_cent, y_cent, slit_width, slit_angle):
    slop, inter = solve_lin(x1=x_cent, y1=y_cent, phi=slit_angle)
    # get angle
    orient_angle = np.arctan((y_val - y_val_gc_pos) / (x_val - x_val_gc_pos)) + np.pi/2
    orient_angle[x_val < x_val_gc_pos] += np.pi
    orient_angle += np.pi
    orient_angle[orient_angle > np.pi] -= np.pi*2
    # get projected distance
    projected_dist = dist2cent * np.cos(slit_angle*np.pi/180 - orient_angle)

    # get the mask inside the regions
    if slit_angle != 0:
        mask_in_region = ((y_val > (slop * x_val + inter - (slit_width/2)/np.sin(slit_angle*np.pi/180))) &
                          (y_val < (slop * x_val + inter + (slit_width/2)/np.sin(slit_angle*np.pi/180))))
    else:
        mask_in_region = ((x_val > (x_val_gc_pos - slit_width/2)) &
                          (x_val < (x_val_gc_pos + slit_width/2)))

    return projected_dist, mask_in_region, slop, inter


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def identify_gc(color_vi, color_ub, slit_width_ref, slit_angle_ref, slit_bin_borders, n_slit_bins, use_saddle=True):

    projected_dist_ref, mask_in_region_ref, slop_ref, inter_ref = get_projected_dist(x_val=color_vi,
                                                                                     y_val=color_ub,
                                                                                     x_cent=x_val_gc_pos,
                                                                                     y_cent=y_val_gc_pos,
                                                                                     slit_width=slit_width_ref,
                                                                                     slit_angle=slit_angle_ref)

    # get bin histogram
    slit_bins = np.linspace(slit_bin_borders[0], slit_bin_borders[1], n_slit_bins)
    center_of_slit_bins = (slit_bins[:-1] + slit_bins[1:]) / 2
    slit_hist, slit_bins = np.histogram(projected_dist_ref[mask_in_region_ref], bins=slit_bins)

    result_dict = {'center_of_slit_bins': center_of_slit_bins, 'slit_bins': slit_bins,
                   'slit_hist': slit_hist, 'slit_hist_err': np.sqrt(slit_hist),
                   'slop_ref': slop_ref, 'inter_ref': inter_ref, 'slit_angle_ref': slit_angle_ref}

    max_value = np.nanmax(slit_hist)

    if use_saddle:
        # fit gaussian model
        gmodel = Model(two_gauss)
        params = Parameters()
        params.add('amp_1', value=max_value*0.8, vary=True, min=0, max=max_value*1.5)
        params.add('amp_2', value=max_value*0.5, vary=True, min=0, max=max_value*1.2)
        params.add('mu_1', value=0, vary=True, min=-0.1, max=0.1)
        params.add('mu_2', value=0.7, vary=True, min=0.25, max=1.5)
        params.add('sig_1', value=0.2, vary=True, min=0.01, max=0.5)
        params.add('sig_2', value=0.2, vary=True, min=0.01, max=1.5)
        result = gmodel.fit(slit_hist, x=center_of_slit_bins, params=params)
        smoothed_hist = smooth(result_dict['slit_hist'], 3)
        interp_func = interp1d(result_dict['center_of_slit_bins'], smoothed_hist, kind='quadratic')
        dummy_x_data = np.linspace(min(result_dict['center_of_slit_bins']), max(result_dict['center_of_slit_bins']),
                                   n_slit_bins*10)
        dummy_y_data = interp_func(dummy_x_data)

        saddle_val = np.min(dummy_y_data[(dummy_x_data > result.params['mu_1'].value) &
                                         (dummy_x_data < result.params['mu_2'].value)])
        right_lim = dummy_x_data[np.where(dummy_y_data == saddle_val)]

        # right value
        left_lim = np.percentile(projected_dist_ref[mask_in_region_ref & (projected_dist_ref < right_lim)], 16)

    else:

        right_lim = np.percentile(projected_dist_ref[mask_in_region_ref], 84)
        left_lim = np.percentile(projected_dist_ref[mask_in_region_ref], 16)

    result_dict.update({'right_lim': right_lim, 'left_lim': left_lim})

    # plt.step(result_dict['center_of_slit_bins'], result_dict['slit_hist'], where='mid', color='k')
    # # plt.plot(result_dict['center_of_slit_bins'], smoothed_hist, color='g')
    # # plt.plot(dummy_x_data, dummy_y_data, color='r')
    # plt.plot([right_lim, right_lim], [0, max_value])
    # plt.plot([left_lim, left_lim], [0, max_value])
    # plt.show()
    # exit()

    return result_dict


def plot_hist(ax, result_dict, fontsize):
    ax.step(result_dict['center_of_slit_bins'], result_dict['slit_hist'], where='mid', color='k')
    # plot_fit
    # dummy_x = np.linspace(slit_bin_borders[0], slit_bin_borders[1], 1000)
    # gauss_1 = gauss(x=dummy_x, amp=result_dict['amp_1'], mu=result_dict['mu_1'], sig=result_dict['sig_1'])
    # gauss_2 = gauss(x=dummy_x, amp=result_dict['amp_2'], mu=result_dict['mu_2'], sig=result_dict['sig_2'])
    # gauss_3 = gauss(x=dummy_x, amp=result_dict['amp_3'], mu=result_dict['mu_3'], sig=result_dict['sig_3'])
    # ax.plot(dummy_x, gauss_1 + gauss_2 + gauss_3, color='g')
    # ax.plot(dummy_x, gauss_3, color='y')
    # ax.plot(dummy_x, gauss_2, color='b')
    # ax.plot(dummy_x, gauss_1, color='r', linewidth=3)

    ax.plot([result_dict['right_lim'], result_dict['right_lim']], [0, max(result_dict['slit_hist'])])
    ax.plot([result_dict['left_lim'], result_dict['left_lim']], [0, max(result_dict['slit_hist'])])
    ax.set_ylabel('# Counts', fontsize=fontsize)

    ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                      labelsize=fontsize)


def plot_offset_sigma(ax, result_dict):
    angle = result_dict['slit_angle_ref']
    x_offset_neg_1 = x_val_gc_pos - result_dict['left_lim']*np.sin(angle*np.pi/180)
    y_offset_neg_1 = y_val_gc_pos - result_dict['left_lim']*np.cos(angle*np.pi/180)
    x_offset_pos_1 = x_val_gc_pos - result_dict['right_lim']*np.sin(angle*np.pi/180)
    y_offset_pos_1 = y_val_gc_pos - result_dict['right_lim']*np.cos(angle*np.pi/180)
    ax.scatter(x_offset_neg_1, y_offset_neg_1, color='b', marker='o', s=80)
    ax.scatter(x_offset_pos_1, y_offset_pos_1, color='g', marker='o', s=80)



# load the raw color-color-points
color_vi_hum_class_1 = np.load('data/color_vi_hum_class_1.npy')
color_ub_hum_class_1 = np.load('data/color_ub_hum_class_1.npy')
color_vi_ml_class_1 = np.load('data/color_vi_ml_class_1.npy')
color_ub_ml_class_1 = np.load('data/color_ub_ml_class_1.npy')
color_vi_ml_mag_cut_class_1 = np.load('data/color_vi_ml_mag_cut_class_1.npy')
color_ub_ml_mag_cut_class_1 = np.load('data/color_ub_ml_mag_cut_class_1.npy')


# get also histograms and convolve them with a gaussian
x_bins = np.linspace(-1.2, 2.4, 200)
y_bins = np.linspace(-2.2, 1.9, 200)
# get center of bins if needed
center_of_xbins = (x_bins[:-1] + x_bins[1:]) / 2
center_of_ybins = (y_bins[:-1] + y_bins[1:]) / 2
# get a meshgrid
x_mesh, y_mesh = np.meshgrid(center_of_xbins, center_of_ybins)

kernel = Gaussian2DKernel(x_stddev=2.0)
hist_hum, xedges, yedges = np.histogram2d(color_vi_hum_class_1, color_ub_hum_class_1, bins=(x_bins, y_bins))
hist_ml, xedges, yedges = np.histogram2d(color_vi_ml_class_1, color_ub_ml_class_1, bins=(x_bins, y_bins))
hist_ml_mag_cut, xedges, yedges = np.histogram2d(color_vi_ml_mag_cut_class_1, color_ub_ml_mag_cut_class_1, bins=(x_bins, y_bins))
hist_hum = convolve(hist_hum, kernel)
hist_ml = convolve(hist_ml, kernel)
hist_ml_mag_cut = convolve(hist_ml_mag_cut, kernel)
hist_hum = hist_hum.T
hist_ml = hist_ml.T
hist_ml_mag_cut = hist_ml_mag_cut.T


# get center of gravity in GC distribution
mask_gc_reg = (x_mesh > 0.9) & (x_mesh < 1.4) & (y_mesh > -0.3) & (y_mesh < 0.4)
max_value_in_gc_reg = np.max(hist_ml_mag_cut[mask_gc_reg])
array_pos_gc = np.where(hist_ml_mag_cut == max_value_in_gc_reg)
x_val_gc_pos = x_mesh[array_pos_gc]
y_val_gc_pos = y_mesh[array_pos_gc]
# global distance to central points
dist2cent = np.sqrt((color_vi_ml_class_1 - x_val_gc_pos) ** 2 +
                    (color_ub_ml_class_1 - y_val_gc_pos) ** 2)


# get linear function
slit_width_ref = 0.1
# slit_angle_ref = 30
slit_bin_borders = (-1.2, 2.5)
n_slit_bins = 40




# plot
figure = plt.figure(figsize=(10, 40))
fontsize = 23
ax_cc = figure.add_axes([0.05, 0.78, 0.95, 0.20])
# ax_dist_cb = figure.add_axes([0.86, 0.44, 0.02, 0.4])
ax_hist_1 = figure.add_axes([0.09, 0.65, 0.9, 0.1])
ax_hist_2 = figure.add_axes([0.09, 0.55, 0.9, 0.1])
ax_hist_3 = figure.add_axes([0.09, 0.45, 0.9, 0.1])
ax_hist_4 = figure.add_axes([0.09, 0.35, 0.9, 0.1])
ax_hist_5 = figure.add_axes([0.09, 0.25, 0.9, 0.1])
ax_hist_6 = figure.add_axes([0.09, 0.15, 0.9, 0.1])
ax_hist_7 = figure.add_axes([0.09, 0.05, 0.9, 0.1])

ax_cc.imshow(hist_ml, origin='lower', extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()))
ax_cc.scatter(x_val_gc_pos, y_val_gc_pos, color='r', marker='*', s=80)
# plot_3 sigma_cuts


for angle in np.linspace(0, 170, 18):
    print(angle)
    if (angle < 15) | (angle > 150):
        use_saddle = False
    else:
        use_saddle = True
    gc_results = identify_gc(color_vi=color_vi_ml_class_1, color_ub=color_ub_ml_class_1,
                               slit_width_ref=slit_width_ref, slit_angle_ref=angle,
                             slit_bin_borders=slit_bin_borders, n_slit_bins=n_slit_bins, use_saddle=use_saddle)
    # plot_hist(ax=ax_hist_1, result_dict=gc_results, fontsize=fontsize)

    plot_offset_sigma(ax=ax_cc, result_dict=gc_results)



cmap_dist = matplotlib.cm.get_cmap('seismic')
norm_dist = matplotlib.colors.Normalize(vmin=-1, vmax=1)

ax_cc.set_ylim(1.9, -2)
ax_cc.set_xlim(-1.2, 2.4)

# add labels and other cosmetics for the plots
# title
ax_cc.set_title('ML mag-cut', fontsize=fontsize)
# axis labels
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_hist_7.set_xlabel('Dist. to GC', fontsize=fontsize)
# tick parameters
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)


plt.savefig('plot_output/gc_percentile.png')









