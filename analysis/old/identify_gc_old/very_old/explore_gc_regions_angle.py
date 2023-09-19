import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
import astropy.units as u
from matplotlib.colors import Normalize, LogNorm

from matplotlib.colorbar import ColorbarBase
import matplotlib


def solve_lin(x1, y1, phi):
    slop = - 1/np.tan(phi * np.pi/180)
    inter = y1 + x1/np.tan(phi * np.pi/180)
    return slop, inter



def get_projected_dist(x_av, y_val, x_cent, y_cent, slit_width, slit_angle):
    slop, inter = solve_lin(x1=x_cent, y1=y_cent, phi=slit_angle)
    # get angle
    orient_angle = np.arctan((y_val - y_val_gc_pos) / (x_av - x_val_gc_pos)) + np.pi/2
    orient_angle[color_vi_ml_mag_cut_class_1 < x_val_gc_pos] += np.pi
    orient_angle += np.pi
    orient_angle[orient_angle > np.pi] -= np.pi*2
    # get projected distance
    projected_dist = dist2cent * np.cos(slit_angle*np.pi/180 - orient_angle)

    # get the mask inside the regions
    if slit_angle != 0:
        mask_in_region = ((color_ub_ml_mag_cut_class_1 > (slop * color_vi_ml_mag_cut_class_1 + inter - (slit_width/2)/np.sin(slit_angle*np.pi/180))) &
                          (color_ub_ml_mag_cut_class_1 < (slop * color_vi_ml_mag_cut_class_1 + inter + (slit_width/2)/np.sin(slit_angle*np.pi/180))))
    else:
        mask_in_region = ((color_vi_ml_mag_cut_class_1 > (x_val_gc_pos - slit_width/2)) &
                          (color_vi_ml_mag_cut_class_1 < (x_val_gc_pos + slit_width/2)))

    return projected_dist, mask_in_region, slop, inter






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
dist2cent = np.sqrt((color_vi_ml_mag_cut_class_1 - x_val_gc_pos) ** 2 +
                    (color_ub_ml_mag_cut_class_1 - y_val_gc_pos) ** 2)


# get linear function
slit_width_ref = 0.1
slit_angle_ref = 160

projected_dist_ref, mask_in_region_ref, slop_ref, inter_ref = get_projected_dist(x_av=color_vi_ml_mag_cut_class_1,
                                                                 y_val=color_ub_ml_mag_cut_class_1,
                                                                 x_cent=x_val_gc_pos, y_cent=y_val_gc_pos,
                                                                 slit_width=slit_width_ref, slit_angle=slit_angle_ref)

# plot
figure = plt.figure(figsize=(13, 16))
fontsize = 23
ax_cc = figure.add_axes([-0.005, 0.32, 0.95, 0.65])
ax_dist_cb = figure.add_axes([0.86, 0.44, 0.02, 0.4])
ax_hist = figure.add_axes([0.08, 0.06, 0.9, 0.2])



dummy_x_data = np.linspace(- 2, 3, 100)
dummy_y_data_low = slop_ref * dummy_x_data + inter_ref - (slit_width_ref/2)/np.sin(slit_angle_ref*np.pi/180)
dummy_y_data_high = slop_ref * dummy_x_data + inter_ref + (slit_width_ref/2)/np.sin(slit_angle_ref*np.pi/180)
ax_cc.plot(dummy_x_data, dummy_y_data_low)
ax_cc.plot(dummy_x_data, dummy_y_data_high)

ax_cc.imshow(hist_ml_mag_cut, origin='lower', extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()))
ax_cc.scatter(x_val_gc_pos, y_val_gc_pos)



cmap_dist = matplotlib.cm.get_cmap('seismic')
norm_dist = matplotlib.colors.Normalize(vmin=-1, vmax=1)

ColorbarBase(ax_dist_cb, orientation='vertical', cmap=cmap_dist, norm=norm_dist, extend='neither', ticks=None)
ax_dist_cb.set_ylabel(r'Dist. to GC', labelpad=0, fontsize=fontsize)
ax_dist_cb.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                    labeltop=True, labelsize=fontsize)

ax_cc.scatter(color_vi_ml_mag_cut_class_1[mask_in_region_ref], color_ub_ml_mag_cut_class_1[mask_in_region_ref],
              c=projected_dist_ref[mask_in_region_ref], s=40, cmap=cmap_dist, norm=norm_dist)

ax_cc.set_ylim(1.9, -2)
ax_cc.set_xlim(-1.2, 2.4)

bins = np.linspace(-1, 1, 50)
center_of_bins = (bins[:-1] + bins[1:]) / 2
hist, xbins = np.histogram(projected_dist_ref[mask_in_region_ref], bins=bins)
ax_hist.hist(projected_dist_ref[mask_in_region_ref], bins=bins,
             histtype='step', align='mid', color='k')
ax_hist.errorbar(center_of_bins, hist, yerr=np.sqrt(hist), fmt='o')

from scipy.optimize import curve_fit
from lmfit import Model

def gauss(x, amp, mu, sig):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))

def two_gauss(x, amp_1, mu_1, sig_1, amp_2, mu_2, sig_2):
    return amp_1 * np.exp(-(x - mu_1) ** 2 / (2 * sig_1 ** 2)) + amp_2 * np.exp(-(x - mu_2) ** 2 / (2 * sig_2 ** 2))

gmodel = Model(two_gauss)

from lmfit import Parameters
params = Parameters()
params.add('amp_1', value=40, vary=True, min=0, max=100)
params.add('amp_2', value=40, vary=True, min=0, max=100)
params.add('mu_1', value=0, vary=True, min=-0.1, max=0.1)
params.add('mu_2', value=-0.5, vary=True, min=0.25, max=0.7)
params.add('sig_1', value=0.2, vary=True, min=0.01, max=0.5)
params.add('sig_2', value=0.2, vary=True, min=0.01, max=0.5)



result = gmodel.fit(hist, x=center_of_bins,
                    params=params
                    )# , weights=weights)




print('result ', result.fit_report())

print(result.__dict__)
print(result.params['amp_1'].value)
print(result.params['amp_1'].stderr)

ax_hist.plot(center_of_bins, result.best_fit, 'b--')





# add labels and other cosmetics for the plots
# title
ax_cc.set_title('ML mag-cut', fontsize=fontsize)
# axis labels
ax_cc.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax_cc.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
ax_hist.set_xlabel('Dist. to GC', fontsize=fontsize)
ax_hist.set_ylabel('# Counts', fontsize=fontsize)
# tick parameters
ax_cc.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
ax_hist.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)

plt.savefig('plot_output/gc_angle.png')









