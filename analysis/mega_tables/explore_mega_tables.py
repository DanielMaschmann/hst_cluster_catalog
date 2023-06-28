""" bla bla bla """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import photometry_tools
from photometry_tools.analysis_tools import AnalysisTools
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize, LogNorm
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astropy.table import Table
from mega_table import RadialMegaTable, TessellMegaTable
from photometry_tools.data_access import CatalogAccess
from photometry_tools import helper_func


def plot_hex(ax, pos_x, pos_y, hex_sep_x, hex_sep_y):

    # right line
    ax.plot([pos_x + hex_sep_x/2, pos_x + hex_sep_x/2],
             [pos_y - hex_sep_y/4, pos_y + hex_sep_y/4], color='k')
    # left line
    ax.plot([pos_x - hex_sep_x/2, pos_x - hex_sep_x/2],
             [pos_y - hex_sep_y/4, pos_y + hex_sep_y/4], color='k')
    # top lines
    ax.plot([pos_x, pos_x + hex_sep_x/2],
             [pos_y + hex_sep_y/2, pos_y + hex_sep_y/4], color='k')
    ax.plot([pos_x, pos_x - hex_sep_x/2],
             [pos_y + hex_sep_y/2, pos_y + hex_sep_y/4], color='k')
    # bottom lines
    ax.plot([pos_x, pos_x + hex_sep_x/2],
             [pos_y - hex_sep_y/2, pos_y - hex_sep_y/4], color='k')
    ax.plot([pos_x, pos_x - hex_sep_x/2],
             [pos_y - hex_sep_y/2, pos_y - hex_sep_y/4], color='k')




    # ax.plot([pos_x + hex_sep/2, pos_x + hex_sep/4],
    #          [pos_y, pos_y - hex_sep/2], color='k')
    # # left lines
    # ax.plot([pos_x - hex_sep/2, pos_x - hex_sep/4],
    #          [pos_y, pos_y + hex_sep/2], color='k')
    # ax.plot([pos_x - hex_sep/2, pos_x - hex_sep/4],
    #          [pos_y, pos_y - hex_sep/2], color='k')
    # # top lines
    # ax.plot([pos_x - hex_sep/4, pos_x + hex_sep/4],
    #          [pos_y + hex_sep/2, pos_y + hex_sep/2], color='k')
    # ax.plot([pos_x - hex_sep/4, pos_x + hex_sep/4],
    #          [pos_y - hex_sep/2, pos_y - hex_sep/2], color='k')
#



galaxy_name = 'ngc0628'
target = 'ngc0628c'


# load DSS image
# paths_dss = SkyView.get_images(position=galaxy_name, survey='DSS2 Red', radius=20*u.arcmin)
# data_dss = paths_dss[0][0].data
# wcs_dss = WCS(paths_dss[0][0].header)
dss_hdu = fits.open('dss_ngc0628.fits')
data_dss = dss_hdu[0].data
wcs_dss = WCS(dss_hdu[0].header)



# prepare the flux file
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)


catalog_access.load_hst_cc_list(target_list=[target])
cluster_class = catalog_access.get_hst_cc_class_human(target=target)
phangs_id = catalog_access.get_hst_cc_phangs_id(target=target)
age = catalog_access.get_hst_cc_age(target=target)
mstar = catalog_access.get_hst_cc_stellar_m(target=target)
ebv = catalog_access.get_hst_cc_ebv(target=target)
ra, dec = catalog_access.get_hst_cc_coords_world(target=target)


t = TessellMegaTable.read('/home/benutzer/data/PHANGS_products/mega_tables/v3p0/hexagon/NGC0628_base_hexagon_1p5kpc.ecsv')
print(t)
print(t.colnames)


# exit()


fig = plt.figure(figsize=(17, 17))
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=wcs_dss)
ax1.imshow(data_dss, cmap='Greys')

# pos_pix = wcs_dss.world_to_pixel(pos)
# ax1.scatter(pos_pix[0], pos_pix[1], color='r')


separation_coord = helper_func.calc_coord_separation(ra_ref=t['RA'][0], dec_ref=t['DEC'][0],
                                               ra=t['RA'][1], dec=t['DEC'][1])
sepaation_pix_x = separation_coord / wcs_dss.proj_plane_pixel_scales()[0]
sepaation_pix_y = separation_coord / wcs_dss.proj_plane_pixel_scales()[1]

pos = SkyCoord(ra=t['RA'][0], dec=t['DEC'][0], unit=(u.degree, u.degree), frame='fk5')

print(sepaation_pix_y)
sepaation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
sepaation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
sepaation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
sepaation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
print(sepaation_pix_y)


# # top and bottom
# ax1.scatter(pos_pix[0], pos_pix[1] - sepaation_pix/2, color='r')
# # left
# ax1.scatter(pos_pix[0] - sepaation_pix/2, pos_pix[1] + sepaation_pix/4, color='r')
# ax1.scatter(pos_pix[0] - sepaation_pix/2, pos_pix[1] - sepaation_pix/4, color='r')
# # right
# ax1.scatter(pos_pix[0] + sepaation_pix/2, pos_pix[1] + sepaation_pix/4, color='r')
# ax1.scatter(pos_pix[0] + sepaation_pix/2, pos_pix[1] - sepaation_pix/4, color='r')







# ax1.plot([pos_pix[0], pos_pix[0] + sepaation_pix/4],
#          [pos_pix[1], pos_pix[1] + sepaation_pix/2], color='k')
# ax1.plot([pos_pix[0] - sepaation_pix/2, pos_pix[0] + sepaation_pix/4],
#          [pos_pix[1], pos_pix[1] - sepaation_pix/2], color='k')
#









for hex_ra, hex_dec in zip(t['RA'], t['DEC']):
    pos = SkyCoord(ra=hex_ra, dec=hex_dec, unit=(u.degree, u.degree), frame='fk5')
    pos_pix = wcs_dss.world_to_pixel(pos)
    # ax1.scatter(pos_pix[0], pos_pix[1], color='k')
    plot_hex(ax=ax1, pos_x=pos_pix[0], pos_y=pos_pix[1], hex_sep_x=sepaation_pix_x, hex_sep_y=sepaation_pix_y)



# pos = SkyCoord(ra=t['RA'][0], dec=t['DEC'][0], unit=(u.degree, u.degree), frame='fk5')
# pos_pix = wcs_dss.world_to_pixel(pos)
# ax1.scatter(pos_pix[0], pos_pix[1], color='r')
#
# pos = SkyCoord(ra=t['RA'][1], dec=t['DEC'][1], unit=(u.degree, u.degree), frame='fk5')
# pos_pix = wcs_dss.world_to_pixel(pos)
# ax1.scatter(pos_pix[0], pos_pix[1], color='r')

#
# plt.show()
#
# exit()



id_dict = {}
for id in t['ID']:
    id_dict.update({'ra_%i' % id: [], 'dec_%i' % id: []})

for index in range(len(ra)):
    hex_id = t.find_coords_in_regions(ra=ra[index], dec=dec[index], fill_value=-1)
    id_dict['ra_%i' % hex_id].append(ra[index])
    id_dict['dec_%i' % hex_id].append(dec[index])


for id in t['ID']:
    pos = SkyCoord(ra=id_dict['ra_%i' % id], dec=id_dict['dec_%i' % id], unit=(u.degree, u.degree), frame='fk5')
    pos_pix = wcs_dss.world_to_pixel(pos)
    print(pos_pix)
    ax1.scatter(pos_pix[0], pos_pix[1])

plt.show()

# fig = t.show_tiles_on_sky()
# plt.show()



