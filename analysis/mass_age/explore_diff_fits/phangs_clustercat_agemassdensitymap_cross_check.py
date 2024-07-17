import numpy as np
from astropy.table import Table
from astropy.table import QTable
from photutils.datasets import (make_gaussian_sources_image,
                                make_random_gaussians_table)
from matplotlib import colors
import matplotlib.pyplot as plt
from astropy.io import fits
from photometry_tools import helper_func

normtype='flux'
if normtype=='amplitude': normfact=61.
if normtype=='flux': normfact=1.
minuncdex=0.125
# t=Table.read('/Users/dthilker/SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass/SEDfix_PHANGS_IR4_allgal_Ha1_inclusiveGCcc_inclusiveGCclass_phangs_hst_v1p2_ml_class12.fits')
t=Table.read('/home/benutzer/data/PHANGS_products/HST_catalogs/SEDfix_final_test_catalogs/'
             'SEDfix_PHANGS_IR4_ngc1512_Ha1_inclusiveGCcc_inclusiveGCclass_phangs_hst_v1p2_ml_class12.fits')
#t=Table.read('/Users/dthilker/SEDfix_Ha1_inclusiveGCcc_inclusiveGCclass/SEDfix_PHANGS_IR4_ngc1512_Ha1_inclusiveGCcc_inclusiveGCclass_phangs_hst_v1p2_ml_class12.fits')
# tlim=Table.read('/Users/dthilker/Downloads/models-block-Z0p02_fcov0_SSP_masslimMVminus6.fits')
# sfhage=(tlim['sfh.age']*1.e6)
# masslim=(tlim['masslim_MVminus6']+0.7)

# get model
hdu_a = fits.open('../cigale_model/sfh2exp/no_dust/sol_met/out/models-block-0.fits')
data_mod = hdu_a[1].data
sfhage = data_mod['sfh.age']
m_star_mod = data_mod['stellar.m_star']
flux_f555w_mod = data_mod['F555W_UVIS_CHIP2']

ABmag_F555W = - 6
f_mJy_f555w = 1.e3 * 1.e23 * 10.**((ABmag_F555W + 48.6) / -2.5)
masslim = np.log10(f_mJy_f555w * m_star_mod / flux_f555w_mod)



log_age_ml = np.log10(t['SEDfix_age']*1.e6)
log_m_star_ml = np.log10(t['SEDfix_mass'])
age_log_err_ml = np.log10((t['SEDfix_age']+t['PHANGS_AGE_MINCHISQ_ERR'])/t['SEDfix_age'])  #log10((60+10)/60.)
m_star_log_err_ml = np.log10((t['SEDfix_mass']+t['PHANGS_MASS_MINCHISQ_ERR'])/t['SEDfix_mass'])

age_log_err_ml[age_log_err_ml<1.]=minuncdex #impose min of minuncdex dex uncertainty
m_star_log_err_ml[m_star_log_err_ml<1.]=minuncdex

log_age_ml_old = np.log10(t['PHANGS_AGE_MINCHISQ']*1.e6)
log_m_star_ml_old = np.log10(t['PHANGS_MASS_MINCHISQ'])
age_log_err_ml_old = np.log10((t['PHANGS_AGE_MINCHISQ']+t['PHANGS_AGE_MINCHISQ_ERR'])/t['PHANGS_AGE_MINCHISQ'])  #log10((60+10)/60.)
m_star_log_err_ml_old = np.log10((t['PHANGS_MASS_MINCHISQ']+t['PHANGS_MASS_MINCHISQ_ERR'])/t['PHANGS_MASS_MINCHISQ'])

age_log_err_ml_old[age_log_err_ml_old<1.]=minuncdex #impose min of minuncdex dex uncertainty
m_star_log_err_ml_old[m_star_log_err_ml_old<1.]=minuncdex


# shape = (100, 200)
# data = make_gaussian_sources_image(shape, table) / normfact
# data[np.where(data==0.)]=0.
# print('SEDfix_sum ',np.sum(data),' max ',np.max(data))

# age_lim = (0, 4.3)
age_lim = (5.7, 10.3)
mass_lim = (2, 7.5)
n_bins = 100

# bins
x_bins_gauss = np.linspace(age_lim[0], age_lim[1], n_bins)
y_bins_gauss = np.linspace(mass_lim[0], mass_lim[1], n_bins)
# get a mesh
x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
gauss_map_ml = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
gauss_map_ml_old = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

mask_bad_values_ml = (np.isinf(age_log_err_ml)) | (np.isinf(m_star_log_err_ml)) | (np.isnan(age_log_err_ml)) | (np.isnan(m_star_log_err_ml))
mask_bad_values_ml_old = (np.isinf(age_log_err_ml_old)) | (np.isinf(m_star_log_err_ml_old)) | (np.isnan(age_log_err_ml_old)) | (np.isnan(m_star_log_err_ml_old))

for color_index in range(len(log_age_ml)):
    if mask_bad_values_ml[color_index]:
        continue
    x_err = np.sqrt(age_log_err_ml[color_index]**2)
    y_err = np.sqrt(m_star_log_err_ml[color_index]**2)
    gauss = helper_func.gauss2d(x=x_mesh, y=y_mesh, x0=log_age_ml[color_index], y0=log_m_star_ml[color_index],
                    sig_x=x_err, sig_y=y_err)
    gauss_map_ml += gauss

for color_index in range(len(log_age_ml_old)):
    if mask_bad_values_ml_old[color_index]:
        continue
    x_err = np.sqrt(age_log_err_ml_old[color_index]**2)
    y_err = np.sqrt(m_star_log_err_ml_old[color_index]**2)
    gauss = helper_func.gauss2d(x=x_mesh, y=y_mesh, x0=log_age_ml_old[color_index], y0=log_m_star_ml_old[color_index],
                    sig_x=x_err, sig_y=y_err)
    gauss_map_ml_old += gauss



fig, ax = plt.subplots()
plot1=ax.imshow(gauss_map_ml, origin='lower', interpolation='nearest',cmap='plasma',extent=(age_lim[0], age_lim[1], mass_lim[0], mass_lim[1]),norm=colors.LogNorm(vmin=0.001, vmax=500.0))
ax.set_aspect("auto")
ax.set_xlabel("log age [yr]")
ax.set_ylabel("log mass [Msun]")
plt.plot(np.log10(sfhage) + 6, masslim,linestyle='dashed')
plt.plot([5.8,7.0,7.0,5.8],[4.7,4.7,7.5,7.5],color='black')
plt.plot([10.2,10.0,10.0,10.2],[5.0,5.0,7.5,7.5],color='black')
plt.plot([7.5,7.5],[2.5,7.5],linestyle='dotted',color='black')
plt.plot([9.0,9.0],[2.5,7.5],linestyle='dotted',color='black')
ageKDESEDfix=np.sum(gauss_map_ml,axis=0)
massKDESEDfix=np.sum(gauss_map_ml,axis=1)
cb=fig.colorbar(plot1, ax=ax)
cb.ax.set_ylabel('age-mass number density')
plt.savefig('./agemassdensityplot_SEDfix_check.png')
plt.show()
dataSEDfix=gauss_map_ml.copy()


fig, ax = plt.subplots()
plot2=ax.imshow(gauss_map_ml_old, origin='lower', interpolation='nearest',cmap='plasma',extent=(age_lim[0], age_lim[1], mass_lim[0], mass_lim[1]),norm=colors.LogNorm(vmin=0.1, vmax=150.0))
ax.set_aspect("auto")
ax.set_xlabel("log age [yr]")
ax.set_ylabel("log mass [Msun]")
plt.plot(np.log10(sfhage) + 6,masslim,linestyle='dashed')
plt.plot([5.8,7.0,7.0,5.8],[4.7,4.7,7.5,7.5],color='black')
plt.plot([10.2,10.0,10.0,10.2],[5.0,5.0,7.5,7.5],color='black')
plt.plot([7.5,7.5],[2.5,7.5],linestyle='dotted',color='black')
plt.plot([9.0,9.0],[2.5,7.5],linestyle='dotted',color='black')
cb=fig.colorbar(plot2, ax=ax)
cb.ax.set_ylabel('age-mass number density')
ageKDEConv=np.sum(gauss_map_ml_old,axis=0)
massKDEConv=np.sum(gauss_map_ml_old,axis=1)
plt.savefig('./agemassdensityplot_conventional_check.png')
plt.show()
dataConv=gauss_map_ml_old.copy()


dataSEDmConv=dataSEDfix-dataConv
print(dataSEDmConv)

print('Difference ',np.sum(dataSEDmConv),' min ',np.min(dataSEDmConv),' max ',np.max(dataSEDmConv),'median ', np.median(dataSEDmConv))
fig, ax = plt.subplots()
plot3=ax.imshow(dataSEDmConv, origin='lower', interpolation='nearest',cmap='bwr_r',extent=(age_lim[0], age_lim[1], mass_lim[0], mass_lim[1]), )
                # norm=colors.SymLogNorm(linthresh=3., linscale=2., vmin=-30.0, vmax=30.0, base=10))
ax.set_aspect("auto")
ax.set_xlabel("log age [yr]")
ax.set_ylabel("log mass [Msun]")
plt.plot(np.log10(sfhage) + 6,masslim,linestyle='dashed')
plt.plot([5.8,7.0,7.0,5.8],[4.7,4.7,7.5,7.5],color='black')
plt.plot([10.2,10.0,10.0,10.2],[5.0,5.0,7.5,7.5],color='black')
plt.plot([7.5,7.5],[2.5,7.5],linestyle='dotted',color='black')
plt.plot([9.0,9.0],[2.5,7.5],linestyle='dotted',color='black')
cb=fig.colorbar(plot3, ax=ax)
cb.ax.set_ylabel('age-mass number density diff')
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional_check.png')
plt.show()

plt.plot(ageKDESEDfix-ageKDEConv)
plt.xlabel("log age [yr]")
plt.ylabel("mass-marginalized num. density diff")
# plt.plot([58,102],[0,0],linestyle='dashed')
# plt.xlim([58,102])
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional_massmarginalized_check.png')
plt.show()

plt.plot(massKDESEDfix-massKDEConv)
plt.xlabel("log mass [Msun]")
plt.ylabel("age-marginalized num. density diff")
# plt.plot([25,75],[0,0],linestyle='dashed')
# plt.xlim([25,75])
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional_agemarginalized_check.png')
plt.show()
