from astropy.io import fits
import numpy as np
from photometry_tools import helper_func as hf
import photometry_tools
import matplotlib.pyplot as plt

def display_models(ax, y_color='nuvb',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-',
                   label_sol=None, label_sol50=None):

    y_model_sol = globals()['model_%s_sol' % y_color]
    y_model_sol50 = globals()['model_%s_sol50' % y_color]

    ax.plot(model_vi_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)
    ax.plot(model_vi_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
            color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)

    if age_dots_sol is None:
        age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
    for age in age_dots_sol:
        ax.scatter(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='b', s=80, zorder=20)

    if age_dots_sol50 is None:
        age_dots_sol50 = [500, 1000, 13750]
    for age in age_dots_sol50:
        ax.scatter(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age], color='tab:pink', s=80, zorder=20)

    if age_labels:
        label_dict = globals()['%s_label_dict' % y_color]
        for age in label_dict.keys():
            ax.text(model_vi_sol[age_mod_sol == age]+label_dict[age]['offsets'][0],
                    y_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][1],
                    label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize)


        annotation_dict = globals()['%s_annotation_dict' % y_color]
        for age in annotation_dict.keys():

            ax.annotate(' ', #annotation_dict[age]['label'],
                        xy=(model_vi_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'))
            ax.annotate(' ',
                        xy=(model_vi_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                        xytext=(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'))
            ax.text(model_vi_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40)


age_mod_sol = np.load('../color_color/data_output/age_mod_sol.npy')
model_nuvu_sol = np.load('../color_color/data_output/model_nuvu_sol.npy')
model_nuvb_sol = np.load('../color_color/data_output/model_nuvb_sol.npy')
model_ub_sol = np.load('../color_color/data_output/model_ub_sol.npy')
model_bv_sol = np.load('../color_color/data_output/model_bv_sol.npy')
model_bi_sol = np.load('../color_color/data_output/model_bi_sol.npy')
model_vi_sol = np.load('../color_color/data_output/model_vi_sol.npy')

age_mod_sol50 = np.load('../color_color/data_output/age_mod_sol50.npy')
model_nuvu_sol50 = np.load('../color_color/data_output/model_nuvu_sol50.npy')
model_nuvb_sol50 = np.load('../color_color/data_output/model_nuvb_sol50.npy')
model_ub_sol50 = np.load('../color_color/data_output/model_ub_sol50.npy')
model_bv_sol50 = np.load('../color_color/data_output/model_bv_sol50.npy')
model_bi_sol50 = np.load('../color_color/data_output/model_bi_sol50.npy')
model_vi_sol50 = np.load('../color_color/data_output/model_vi_sol50.npy')



cluster_catalog_data_path = '/home/benutzer/data/PHANGS_products/HST_catalogs'
hst_obs_hdr_file_path = '/home/benutzer/data/PHANGS_products/tables'
morph_mask_path = '/home/benutzer/data/PHANGS_products/environment_masks'
sample_table_path = '/home/benutzer/data/PHANGS_products/sample_table'
catalog_access = photometry_tools.data_access.CatalogAccess(hst_cc_data_path=cluster_catalog_data_path,
                                                            hst_obs_hdr_file_path=hst_obs_hdr_file_path,
                                                            morph_mask_path=morph_mask_path,
                                                            sample_table_path=sample_table_path)

hdu = fits.open('data/simulation_20230901.fits')
data_sim = hdu[1].data
print(data_sim.names)

age_sim = data_sim['age']
mass_sim = data_sim['mass']
ebv_sim = data_sim['attenuation']
flux_f275w_sim = data_sim['hst.wfc3.F275W']
flux_f336w_sim = data_sim['hst.wfc3.F336W']
flux_f438w_sim = data_sim['hst.wfc3.F438W']
flux_f555w_sim = data_sim['hst.wfc3.F555W']
flux_f814w_sim = data_sim['hst.wfc3.F814W']
mag_nuv_sim = hf.conv_mjy2vega(flux=flux_f275w_sim, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F275W'))
mag_u_sim = hf.conv_mjy2vega(flux=flux_f336w_sim, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F336W'))
mag_b_sim = hf.conv_mjy2vega(flux=flux_f438w_sim, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F438W'))
mag_v_sim = hf.conv_mjy2vega(flux=flux_f555w_sim, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F555W'))
mag_i_sim = hf.conv_mjy2vega(flux=flux_f814w_sim, ab_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W', mag_sys='AB'),
                             vega_zp=catalog_access.get_zp_mag(target='ngc7496', band='F814W'))
model_nuvb_sim = mag_nuv_sim - mag_b_sim
model_ub_sim = mag_u_sim - mag_b_sim
model_bv_sim = mag_b_sim - mag_v_sim
model_vi_sim = mag_v_sim - mag_i_sim


x_lim_vi = (-0.7, 2.4)

y_lim_nuvb = (3.2, -2.9)
y_lim_ub = (2.1, -2.2)
y_lim_bv = (1.9, -0.7)

n_bins_nuvbvi = 100
n_bins_ubvi = 100
n_bins_bvvi = 100
kernal_std = 2


fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(14, 20))
fontsize = 25


# ax[0].scatter(model_vi_sim, model_nuvb_sim)
# ax[1].scatter(model_vi_sim, model_ub_sim)
# ax[2].scatter(model_vi_sim, model_bv_sim)


hf.density_with_points(ax=ax[0], x=model_vi_sim,
                       y=model_nuvb_sim,
                       binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_nuvbvi),
                       biny=np.linspace(y_lim_nuvb[1], y_lim_nuvb[0], n_bins_nuvbvi),
                       kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

hf.density_with_points(ax=ax[1], x=model_vi_sim,
                       y=model_ub_sim,
                       binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_ubvi),
                       biny=np.linspace(y_lim_ub[1], y_lim_ub[0], n_bins_ubvi),
                       kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

hf.density_with_points(ax=ax[2], x=model_vi_sim,
                       y=model_bv_sim,
                       binx=np.linspace(x_lim_vi[0], x_lim_vi[1], n_bins_bvvi),
                       biny=np.linspace(y_lim_bv[1], y_lim_bv[0], n_bins_bvvi),
                       kernel_std=kernal_std, cmap='inferno', scatter_size=30, scatter_alpha=0.3)

vi_int = 1.75
nuvb_int = -2.2
ub_int = -1.8
bv_int = -0.5
av_value = 1
hf.plot_reddening_vect(ax=ax[0], x_color_1='v', x_color_2='i',  y_color_1='nuv', y_color_2='b',
                       x_color_int=vi_int, y_color_int=nuvb_int, av_val=1,
                       linewidth=4, line_color='k', text=True, fontsize=fontsize-4, x_text_offset=-0.1, y_text_offset=-0.3)
hf.plot_reddening_vect(ax=ax[1], x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                       x_color_int=vi_int, y_color_int=ub_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)
hf.plot_reddening_vect(ax=ax[2], x_color_1='v', x_color_2='i',  y_color_1='b', y_color_2='v',
                       x_color_int=vi_int, y_color_int=bv_int, av_val=1,
                       linewidth=4, line_color='k', text=False, fontsize=fontsize)

display_models(ax=ax[0], age_label_fontsize=fontsize+2, label_sol=r'BC03, Z$_{\odot}$', label_sol50=r'BC03, Z$_{\odot}/50\,(> 500\,{\rm Myr})$')
display_models(ax=ax[1], age_label_fontsize=fontsize+2, y_color='ub')
display_models(ax=ax[2], age_label_fontsize=fontsize+2, y_color='bv')


ax[0].set_xlim(x_lim_vi)
ax[1].set_xlim(x_lim_vi)
ax[2].set_xlim(x_lim_vi)

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])


ax[0].set_ylim(y_lim_nuvb)
ax[1].set_ylim(y_lim_ub)
ax[2].set_ylim(y_lim_bv)

ax[2].set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize)
ax[0].set_ylabel('NUV (F275W) - B (F438W/F435W'+'$^*$'+')',labelpad=27, fontsize=fontsize)
ax[1].set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')',labelpad=27, fontsize=fontsize)
ax[2].set_ylabel('B (F438W/F435W'+'$^*$'+') - V (F555W)', fontsize=fontsize)


ax[0].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[1].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax[2].tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

fig.subplots_adjust(left=0.1, bottom=0.04, right=0.995, top=0.98, wspace=0.01, hspace=0.01)
plt.savefig('plot_output/cc_sim.png')
