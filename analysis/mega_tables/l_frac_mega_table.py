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


def get_slop_inter(x1, x2, y1, y2):
    slope = (y2 - y1) / (x2 - x1)
    intersect = y1 - x1 * (y2-y1) / (x2 - x1)
    return slope, intersect


def get_hex_2d_mask(data_x_len, data_y_len, pos_x, pos_y, hex_sep_x, hex_sep_y):
    x_dim = np.linspace(0, data_x_len, data_x_len)
    y_dim = np.linspace(0, data_y_len, data_y_len)
    x_grid, y_grid = np.meshgrid(x_dim, y_dim)

    slope_top_1, intersect_top_1 = get_slop_inter(x1=pos_x-hex_sep_x/2, x2=pos_x,
                                                  y1=pos_y+hex_sep_y/4, y2=pos_y+hex_sep_y/2)
    slope_top_2, intersect_top_2 = get_slop_inter(x1=pos_x, x2=pos_x+hex_sep_x/2,
                                                  y1=pos_y+hex_sep_y/2, y2=pos_y+hex_sep_y/4)
    slope_bottom_1, intersect_bottom_1 = get_slop_inter(x1=pos_x-hex_sep_x/2, x2=pos_x,
                                                        y1=pos_y-hex_sep_y/4, y2=pos_y-hex_sep_y/2)
    slope_bottom_2, intersect_bottom_2 = get_slop_inter(x1=pos_x, x2=pos_x+hex_sep_x/2,
                                                        y1=pos_y-hex_sep_y/2, y2=pos_y-hex_sep_y/4)

    mask_hex = ((x_grid > (pos_x - hex_sep_x/2)) &
                (x_grid < (pos_x + hex_sep_x/2)) &
                (y_grid < slope_top_1 * x_grid + intersect_top_1) &
                (y_grid < slope_top_2 * x_grid + intersect_top_2) &
                (y_grid > slope_bottom_1 * x_grid + intersect_bottom_1) &
                (y_grid > slope_bottom_2 * x_grid + intersect_bottom_2))
    return mask_hex


galaxy_name = 'ngc0628'
target = 'ngc0628c'

# prepare the flux file
cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
catalog_access = CatalogAccess(hst_cc_data_path=cluster_catalog_data_path, hst_obs_hdr_file_path=hst_obs_hdr_file_path)

catalog_access.load_hst_cc_list(target_list=[target])




# load mega tables
mega_table = TessellMegaTable.read('/home/benutzer/data/PHANGS_products/mega_tables/v3p0/hexagon/%s_base_hexagon_1p5kpc.ecsv' % galaxy_name.upper())

hex_sig_sfr = mega_table['Sigma_SFR']
hex_sig_sfr_err = mega_table['e_Sigma_SFR']
hex_sig_mol = mega_table['Sigma_mol']
hex_sig_mol_err = mega_table['e_Sigma_mol']
hex_sig_mstar = mega_table['Sigma_star']
hex_sig_mstar_err = mega_table['e_Sigma_star']


ra, dec = catalog_access.get_hst_cc_coords_world(target=target)
flux_v_band = catalog_access.get_hst_cc_band_flux(target=target, band='F555W')



# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name=galaxy_name,
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')
# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy', band_list=['F555W'])

v_band_data = phangs_photometry.hst_bands_data['F555W_data_img']
v_band_wcs = phangs_photometry.hst_bands_data['F555W_wcs_img']

# get the separation of the
separation_coord = helper_func.calc_coord_separation(ra_ref=mega_table['RA'][0], dec_ref=mega_table['DEC'][0],
                                                     ra=mega_table['RA'][1], dec=mega_table['DEC'][1])
separation_pix_x = separation_coord / v_band_wcs.proj_plane_pixel_scales()[0]
separation_pix_y = separation_coord / v_band_wcs.proj_plane_pixel_scales()[1]
pos = SkyCoord(ra=mega_table['RA'][0], dec=mega_table['DEC'][0], unit=(u.degree, u.degree), frame='fk5')
separation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
separation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
separation_pix_y /= np.cos(pos.dec.degree * np.pi/180)
separation_pix_y /= np.cos(pos.dec.degree * np.pi/180)


cluster_flux = np.zeros(len(hex_sig_sfr))
total_hex_flux = np.zeros(len(hex_sig_sfr))


# for index_cluster in range(len(ra)):
#     print('index_cluster ', index_cluster)
#     hex_id = mega_table.find_coords_in_regions(ra=ra[index_cluster], dec=dec[index_cluster], fill_value=-1)
#     cluster_flux[hex_id] += flux_v_band[index_cluster]
#
#
# for index_mega_table in range(len(mega_table['RA'])):
#     print('index_mega_table ', index_mega_table)
#
#     pos = SkyCoord(ra=mega_table['RA'][index_mega_table], dec=mega_table['DEC'][index_mega_table],
#                    unit=(u.degree, u.degree), frame='fk5')
#     hex_central_pos = v_band_wcs.world_to_pixel(pos)
#     mask_hex = get_hex_2d_mask(data_x_len=v_band_data.shape[0], data_y_len=v_band_data.shape[1],
#                                pos_x=hex_central_pos[0], pos_y=hex_central_pos[1],
#                                hex_sep_x=separation_pix_x, hex_sep_y=separation_pix_y)
#     total_hex_flux[index_mega_table] = np.nansum(v_band_data[mask_hex])
#
#
# np.save('cluster_flux.npy', cluster_flux)
# np.save('total_hex_flux.npy', total_hex_flux)

cluster_flux = np.load('cluster_flux.npy')
total_hex_flux = np.load('total_hex_flux.npy')

fraction = cluster_flux / total_hex_flux

plt.scatter(hex_sig_mol[fraction != 0], fraction[fraction != 0])

plt.show()


exit()





fig = plt.figure(figsize=(17, 17))
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=v_band_wcs)

ax1.imshow(v_band_data, cmap='Greys', vmin=0.05, vmax=5.0)



plt.show()


exit()




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



