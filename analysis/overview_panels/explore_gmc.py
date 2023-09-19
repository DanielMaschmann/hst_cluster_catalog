import numpy as np
from photometry_tools import helper_func as hf, plotting_tools
from photometry_tools.data_access import CatalogAccess
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from scipy.stats import gaussian_kde
from astropy.convolution import convolve
import multicolorfits as mcf

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import ConvexHull



ra_hum = np.load('../color_color/data_output/ra_hum.npy')
dec_hum = np.load('../color_color/data_output/dec_hum.npy')

galaxy_name = 'ngc1566'


signal_mask_hdu = fits.open('/home/benutzer/data/PHANGS_products/cloud_catalogs/v4p0_ST1p6/native_signalmask/ngc1566_12m+7m+tp_co21_native_signalmask.fits.bz2')
print(signal_mask_hdu[0].data.shape)

cloud_cat_hdu = fits.open('/home/benutzer/data/PHANGS_products/cloud_catalogs/v4p0_ST1p6/v4p0_gmccats/native/ngc1566_12m+7m+tp_co21_native_props.fits')

print(cloud_cat_hdu[1].data.names)


alma_hdu = fits.open('/home/benutzer/data/PHANGS-ALMA/delivery_v4p0/%s/%s_12m+7m+tp_co21_broad_tpeak.fits' %
                     (galaxy_name, galaxy_name))

alma_data = alma_hdu[0].data
alma_wcs = WCS(alma_hdu[0].header)


bkg_estimator = MedianBackground()
sigma_clip = SigmaClip(sigma=3.0)
print(alma_data.shape)
bkg = Background2D(alma_data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

print(bkg.background)
print(alma_data - bkg.background)
alma_data -= bkg.background
# exit()
#
mean, median, std = sigma_clipped_stats(alma_data, sigma=3.0)
levels_std = np.array(np.arange(start=0.3, stop=5, step=1), dtype=float)
levels = np.ones(len(levels_std)) * mean
levels += levels_std



figure = plt.figure(figsize=(15, 15))
fontsize = 23
ax_alma = figure.add_axes([0.05, 0.05, 0.9, 0.9], projection=alma_wcs)

# ax_alma.imshow(alma_data)
#
# ax_alma.contour(alma_data,
#                 levels=levels,
#                 colors='r', alpha=1, linewidths=2)

ax_alma.imshow(np.sum(signal_mask_hdu[0].data, axis=0))

ax_alma.scatter(cloud_cat_hdu[1].data['XCTR_PIX'], cloud_cat_hdu[1].data['YCTR_PIX'])



plt.show()


