import numpy as np
from astropy.table import Table
from astropy.table import QTable
from photutils.datasets import (make_gaussian_sources_image,
                                make_random_gaussians_table)
from matplotlib import colors
from astropy.io import fits
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



table = QTable()
table[normtype] = np.full(len(t),1.)
table['x_mean'] = np.log10(t['SEDfix_age']*1.e6)*10.
table['y_mean'] = np.log10(t['SEDfix_mass'])*10.
table['x_stddev'] = np.log10((t['SEDfix_age']+t['PHANGS_AGE_MINCHISQ_ERR'])/t['SEDfix_age'])*10.  #log10((60+10)/60.)
table['y_stddev'] = np.log10((t['SEDfix_mass']+t['PHANGS_MASS_MINCHISQ_ERR'])/t['SEDfix_mass'])*10.
table['theta'] = np.radians(np.full(len(t),0.))

table['x_stddev'][table['x_stddev']<1.]=10.*minuncdex #impose min of minuncdex dex uncertainty
table['y_stddev'][table['y_stddev']<1.]=10.*minuncdex

#print(table['x_stddev'],t['SEDfix_age'],t['PHANGS_AGE_MINCHISQ_ERR'])

shape = (100, 200)
data = make_gaussian_sources_image(shape, table) / normfact
data[np.where(data==0.)]=0.
print('SEDfix_sum ',np.sum(data),' max ',np.max(data))

import matplotlib.pyplot as plt



fig, ax = plt.subplots()
plot1=ax.imshow(data[25:75,58:102], origin='lower', interpolation='nearest',cmap='plasma',extent=(5.8,10.2,2.5,7.5),norm=colors.LogNorm(vmin=0.1, vmax=150.0))
ax.set_aspect("auto")
ax.set_xlabel("log age [yr]")
ax.set_ylabel("log mass [Msun]")
plt.plot(np.log10(sfhage) + 6,masslim,linestyle='dashed')
plt.plot([5.8,7.0,7.0,5.8],[4.7,4.7,7.5,7.5],color='black')
plt.plot([10.2,10.0,10.0,10.2],[5.0,5.0,7.5,7.5],color='black')
plt.plot([7.5,7.5],[2.5,7.5],linestyle='dotted',color='black')
plt.plot([9.0,9.0],[2.5,7.5],linestyle='dotted',color='black')
ageKDESEDfix=np.sum(data,axis=0)/10.
massKDESEDfix=np.sum(data,axis=1)/10.
cb=fig.colorbar(plot1, ax=ax)
cb.ax.set_ylabel('age-mass number density')
plt.savefig('./agemassdensityplot_SEDfix.png')
plt.show()
dataSEDfix=data.copy()


table = QTable()
table[normtype] = np.full(len(t),1.)
table['x_mean'] = np.log10(t['PHANGS_AGE_MINCHISQ']*1.e6)*10.
table['y_mean'] = np.log10(t['PHANGS_MASS_MINCHISQ'])*10.
table['x_stddev'] = np.log10((t['PHANGS_AGE_MINCHISQ']+t['PHANGS_AGE_MINCHISQ_ERR'])/t['PHANGS_AGE_MINCHISQ'])*10.  #log10((60+10)/60.)                                                                                  
table['y_stddev'] = np.log10((t['PHANGS_MASS_MINCHISQ']+t['PHANGS_MASS_MINCHISQ_ERR'])/t['PHANGS_MASS_MINCHISQ'])*10.
table['theta'] = np.radians(np.full(len(t),0.))

table['x_stddev'][table['x_stddev']<1.]=10.*minuncdex #again impose 0.1 dex minimum uncertainity
table['y_stddev'][table['y_stddev']<1.]=10.*minuncdex

#print(table['x_stddev'],t['PHANGS_AGE_MINCHISQ'],t['PHANGS_AGE_MINCHISQ_ERR'])

shape = (100, 200)
data = make_gaussian_sources_image(shape, table) / normfact
data[np.where(data==0.)]=0.
print('Conv_sum ',np.sum(data),' max ',np.max(data))


fig, ax = plt.subplots()
plot2=ax.imshow(data[25:75,58:102], origin='lower', interpolation='nearest',cmap='plasma',extent=(5.8,10.2,2.5,7.5),norm=colors.LogNorm(vmin=0.1, vmax=150.0))
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
ageKDEConv=np.sum(data,axis=0)/10.
massKDEConv=np.sum(data,axis=1)/10.
plt.savefig('./agemassdensityplot_conventional.png')
plt.show()
dataConv=data.copy()


dataSEDmConv=dataSEDfix-dataConv
print('Difference ',np.sum(dataSEDmConv),' min ',np.min(dataSEDmConv),' max ',np.max(dataSEDmConv),'median ', np.median(dataSEDmConv))
fig, ax = plt.subplots()
plot3=ax.imshow(dataSEDmConv[25:75,58:102], origin='lower', interpolation='nearest',cmap='bwr_r',extent=(5.8,10.2,2.5,7.5), norm=colors.SymLogNorm(linthresh=3., linscale=2.,
                                              vmin=-30.0, vmax=30.0, base=10))
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
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional.png')
plt.show()

plt.plot(ageKDESEDfix-ageKDEConv)
plt.xlabel("log age [yr]")
plt.ylabel("mass-marginalized num. density diff")
plt.plot([58,102],[0,0],linestyle='dashed')
plt.xlim([58,102])
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional_massmarginalized.png')
plt.show()

plt.plot(massKDESEDfix-massKDEConv)
plt.xlabel("log mass [Msun]")
plt.ylabel("age-marginalized num. density diff")
plt.plot([25,75],[0,0],linestyle='dashed')
plt.xlim([25,75])
plt.savefig('./agemassdensityplot_SEDfix_minus_conventional_agemarginalized.png')
plt.show()
