import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from dust_extinction.parameter_averages import CCM89
import astropy.units as u
from astropy.table import QTable
import math

def sort_counterclockwise(points, centre = None):
  if centre:
    centre_x, centre_y = centre
  else:
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  return counterclockwise_points


def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

    model_ccm89 = CCM89(Rv=r_v)
    reddening1 = model_ccm89(wave1*u.micron) * r_v
    reddening2 = model_ccm89(wave2*u.micron) * r_v

    wave_v = 5388.55 * 1e-4
    reddening_v = model_ccm89(wave_v*u.micron) * r_v

    return (reddening1 - reddening2)*av/reddening_v


def convolve_2d_data(x, y):
    # exclude all bad values
    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))
    x = x[good_values]
    y = y[good_values]
    # create a representation of a kernel-density estimate using Gaussian kernels
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    return xi, yi, zi


def get_contour_of_interest(cs, contour_index):
    #extract contour
    p = cs.collections[contour_index].get_paths()[1]
    v = p.vertices
    # get all points from contour
    vi_cont = []
    ub_cont = []
    for point in v:
        vi_cont.append(point[0])
        ub_cont.append(point[1])
    # convert from list to array
    vi_cont = np.array(vi_cont)
    ub_cont = np.array(ub_cont)
    # reddened contour points
    # calculate an Av shift of 0.5
    # band wavelength taken from
    # http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=HST&gname2=ACS_WFC&asttype=
    v_wave = 5388.55*1e-4
    i_wave = 8117.36*1e-4
    u_wave = 3365.86*1e-4
    b_wave = 4338.57*1e-4
    color_shift_vi = color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=0.5)
    color_shift_ub = color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=0.5)
    # add reddening vector
    vi_cont_red = vi_cont + color_shift_vi
    ub_cont_red = ub_cont + color_shift_ub

    # get convex hull of all points
    vi_cont_all_points = np.concatenate([vi_cont_red, vi_cont])
    ub_cont_all_points = np.concatenate([ub_cont_red, ub_cont])

    all_points = np.array([vi_cont_all_points, ub_cont_all_points]).T

    hull = ConvexHull(all_points, incremental=True, qhull_options='Qx')

    vi_convex_hull = []
    ub_convex_hull = []
    for simplex in hull.simplices:
        vi_convex_hull.append(all_points[simplex[0], 0])
        ub_convex_hull.append(all_points[simplex[0], 1])
    vi_convex_hull = np.array(vi_convex_hull)
    ub_convex_hull = np.array(ub_convex_hull)

    # sort the points by angle
    counterclockwise_points = sort_counterclockwise(points=np.array([vi_convex_hull, ub_convex_hull]).T)
    counterclockwise_points = np.array(counterclockwise_points)
    vi_convex_hull = counterclockwise_points[:,0]
    ub_convex_hull = counterclockwise_points[:,1]

    return_dict = {'vi_cont': vi_cont, 'ub_cont': ub_cont, 'vi_cont_red': vi_cont_red, 'ub_cont_red': ub_cont_red,
                   'vi_convex_hull': vi_convex_hull,'ub_convex_hull': ub_convex_hull}
    return return_dict


# load the raw color-color-points
color_vi_hum_class_1 = np.load('data/color_vi_hum_class_1.npy')
color_ub_hum_class_1 = np.load('data/color_ub_hum_class_1.npy')
color_vi_ml_class_1 = np.load('data/color_vi_ml_class_1.npy')
color_ub_ml_class_1 = np.load('data/color_ub_ml_class_1.npy')
color_vi_ml_mag_cut_class_1 = np.load('data/color_vi_ml_mag_cut_class_1.npy')
color_ub_ml_mag_cut_class_1 = np.load('data/color_ub_ml_mag_cut_class_1.npy')

# now convolve the observed data in order to produce contours
xi_hum, yi_hum, zi_hum = convolve_2d_data(x=color_vi_hum_class_1, y=color_ub_hum_class_1)
xi_ml, yi_ml, zi_ml = convolve_2d_data(x=color_vi_ml_class_1, y=color_ub_ml_class_1)
xi_ml_mag_cut, yi_ml_mag_cut, zi_ml_mag_cut = convolve_2d_data(x=color_vi_ml_mag_cut_class_1,
                                                               y=color_ub_ml_mag_cut_class_1)

# get also histograms and convolve them with a gaussian
x_bins = np.linspace(-1.2, 2.4, 200)
y_bins = np.linspace(-2.2, 1.9, 200)
kernel = Gaussian2DKernel(x_stddev=2.0)
hist_hum, xedges, yedges = np.histogram2d(color_vi_hum_class_1, color_ub_hum_class_1, bins=(x_bins, y_bins))
hist_ml, xedges, yedges = np.histogram2d(color_vi_ml_class_1, color_ub_ml_class_1, bins=(x_bins, y_bins))
hist_ml_mag_cut, xedges, yedges = np.histogram2d(color_vi_ml_mag_cut_class_1, color_ub_ml_mag_cut_class_1, bins=(x_bins, y_bins))
hist_hum = convolve(hist_hum, kernel)
hist_ml = convolve(hist_ml, kernel)
hist_ml_mag_cut = convolve(hist_ml_mag_cut, kernel)

# the percentual contour levels for each contours
levels_hum = [0.0, 0.1, 0.25, 0.4, 0.48, 0.55, 0.68, 0.95, 0.975]
levels_ml = [0.0, 0.1, 0.25, 0.4, 0.48, 0.56, 0.58, 0.68, 0.95, 0.975]
levels_ml_mag_cut = [0.0, 0.1, 0.25, 0.4, 0.48, 0.52, 0.68, 0.95, 0.975]

contour_index_hum = -5
contour_index_ml = -4
contour_index_ml_mag_cut = -4


# plot
fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(20, 9))
fontsize = 17

# plot densities
ax[0].imshow(hist_hum.T, origin='lower', extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()))
ax[1].imshow(hist_ml.T, origin='lower', extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()))
ax[2].imshow(hist_ml_mag_cut.T, origin='lower', extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()))

# plot histograms
cs_hum = ax[0].contour(xi_hum, yi_hum, zi_hum, levels=levels_hum, linewidths=(2,), colors='k', origin='lower')
cs_ml = ax[1].contour(xi_ml, yi_ml, zi_ml, levels=levels_ml, linewidths=(2,), colors='k', origin='lower')
cs_ml_mag_cut = ax[2].contour(xi_ml_mag_cut, yi_ml_mag_cut, zi_ml_mag_cut, levels=levels_ml_mag_cut, linewidths=(2,),
                              colors='k', origin='lower')

# get regions of interest
contour_dict_hum = get_contour_of_interest(cs=cs_hum, contour_index=contour_index_hum)
contour_dict_ml = get_contour_of_interest(cs=cs_ml, contour_index=contour_index_ml)
contour_dict_ml_mag_cut = get_contour_of_interest(cs=cs_ml_mag_cut, contour_index=contour_index_ml_mag_cut)

# plot contours
ax[0].scatter(contour_dict_hum['vi_cont'], contour_dict_hum['ub_cont'], color='r')
ax[0].scatter(contour_dict_hum['vi_cont_red'], contour_dict_hum['ub_cont_red'], color='orange')
ax[0].scatter(contour_dict_hum['vi_convex_hull'], contour_dict_hum['ub_convex_hull'], color='c')

ax[1].scatter(contour_dict_ml['vi_cont'], contour_dict_ml['ub_cont'], color='r')
ax[1].scatter(contour_dict_ml['vi_cont_red'], contour_dict_ml['ub_cont_red'], color='orange')
ax[1].scatter(contour_dict_ml['vi_convex_hull'], contour_dict_ml['ub_convex_hull'], color='c')

ax[2].scatter(contour_dict_ml_mag_cut['vi_cont'], contour_dict_ml_mag_cut['ub_cont'], color='r')
ax[2].scatter(contour_dict_ml_mag_cut['vi_cont_red'], contour_dict_ml_mag_cut['ub_cont_red'], color='orange')
ax[2].scatter(contour_dict_ml_mag_cut['vi_convex_hull'], contour_dict_ml_mag_cut['ub_convex_hull'], color='c')


ax[0].set_ylim(1.9, -2)
ax[0].set_xlim(-1.2, 2.4)
# add labels and other cosmetics for the plots
# title
ax[0].set_title('HUM', fontsize=fontsize)
ax[1].set_title('ML', fontsize=fontsize)
ax[2].set_title('ML mag-cut', fontsize=fontsize)
# axis labels
ax[0].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[1].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize)
# tick parameters
ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                  labelsize=fontsize)

# final adjustment and save figure
fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('plot_output/contour_hull.png')
plt.clf()


# create tables and save them
contour_data_hum = np.array([contour_dict_hum['vi_cont'], contour_dict_hum['ub_cont'],
                             contour_dict_hum['vi_cont_red'], contour_dict_hum['ub_cont_red']]).T
hull_data_hum = np.array([contour_dict_hum['vi_convex_hull'], contour_dict_hum['ub_convex_hull']]).T

contour_data_ml = np.array([contour_dict_ml['vi_cont'], contour_dict_ml['ub_cont'],
                            contour_dict_ml['vi_cont_red'], contour_dict_ml['ub_cont_red']]).T
hull_data_ml = np.array([contour_dict_ml['vi_convex_hull'], contour_dict_ml['ub_convex_hull']]).T

contour_data_ml_mag_cut = np.array([contour_dict_ml_mag_cut['vi_cont'], contour_dict_ml_mag_cut['ub_cont'],
                                    contour_dict_ml_mag_cut['vi_cont_red'], contour_dict_ml_mag_cut['ub_cont_red']]).T
hull_data_ml_mag_cut = np.array([contour_dict_ml_mag_cut['vi_convex_hull'],
                                 contour_dict_ml_mag_cut['ub_convex_hull']]).T

names_contour = ['vi_cont', 'ub_cont', 'vi_cont_red', 'ub_cont_red']
names_hull = ['vi_hull', 'ub_hull']

contour_table_hum = QTable(data=contour_data_hum, names=names_contour)
contour_table_ml = QTable(data=contour_data_ml, names=names_contour)
contour_table_ml_mag_cut = QTable(data=contour_data_ml_mag_cut, names=names_contour)

hull_table_hum = QTable(data=hull_data_hum, names=names_hull)
hull_table_ml = QTable(data=hull_data_ml, names=names_hull)
hull_table_ml_mag_cut = QTable(data=hull_data_ml_mag_cut, names=names_hull)

contour_table_hum.write('data_output/contour_table_hum.fits', overwrite=True)
contour_table_ml.write('data_output/contour_table_ml.fits', overwrite=True)
contour_table_ml_mag_cut.write('data_output/contour_table_ml_mag_cut.fits', overwrite=True)
hull_table_hum.write('data_output/hull_table_hum.fits', overwrite=True)
hull_table_ml.write('data_output/hull_table_ml.fits', overwrite=True)
hull_table_ml_mag_cut.write('data_output/hull_table_ml_mag_cut.fits', overwrite=True)


