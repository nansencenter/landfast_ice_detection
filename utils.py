import os
from datetime import datetime
import glob
import time
import cv2
import numpy as np
from multiprocessing import Pool

import numpy as np
import pyproj
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from nansat import Nansat, NSR, Domain
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from scipy import ndimage as ndi
from osgeo import gdal
from nansat.vrt import VRT, NSR
from s1cs2lib import s1_nc_geolocation
from superlimo.lib import warp

def read_s1_nc(name, band_name='sar_primary'):
    with Dataset(name) as ds:
        b = ds[band_name][:]
    return b

def fill_gaps(array, mask, distance=15):
    """ Fill gaps in input raster

    Parameters
    ----------
    array : 2D numpy.array
        Ratser with deformation field
    mask : 2D numpy.array
        Where are gaps
    distance : int
        Minimum size of gap to fill

    Returns
    -------
    arra : 2D numpy.array
        Ratser with gaps filled

    """
    dist, indi = ndi.distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True)
    gpi = dist <= distance
    r,c = indi[:,gpi]
    array[gpi] = array[r,c]
    return array

class VRT2(VRT):
    def _init_from_dataset_params(self, x_size, y_size,
                                  geo_transform=None,
                                  projection=None,
                                  gcps=None,
                                  gcp_projection='',
                                  **kwargs):
        # x_size, y_size, geo_transform, projection, gcps=None, gcp_projection='', **kwargs
        super().__init__(x_size, y_size, **kwargs)
        # set dataset (geo-)metadata
        if projection:
            self.dataset.SetProjection(str(projection))
        if geo_transform:
            self.dataset.SetGeoTransform(geo_transform)
        if isinstance(gcps, (list, tuple)):
            self.dataset.SetGCPs(gcps, str(gcp_projection))
        self.dataset.SetMetadataItem(str('filename'), self.filename)

        # write file contents
        self.dataset.FlushCache()

def stack_nc_files(filenames, min_sigma0_hh=-25):
    arrays = []
    dates = []
    for filename in sorted(filenames):
        print('Input:', filename)
        array = read_s1_nc(filename)
        mask = array < min_sigma0_hh
        eroded_mask = ndi.binary_erosion(mask, structure=np.ones((5, 5)))
        eroded_mask = ndi.binary_dilation(eroded_mask, structure=np.ones((5, 5)))
        array[eroded_mask] = np.nan
        array = fill_gaps(array, np.isnan(array))
        arrays.append(array)
        name = os.path.basename(filename)
        date0 = datetime.strptime(name.split('_')[4], '%Y%m%dT%H%M%S')
        date1 = datetime.strptime(name.split('_')[5], '%Y%m%dT%H%M%S')
        dates.append([date0, date1])
    first_line = 0
    sar_grid_samples = []
    sar_grid_lines = []
    sar_grid_latitude = []
    sar_grid_longitude = []
    for filename in sorted(filenames):
        s1geo = s1_nc_geolocation(filename)
        s1geo.read_geolocation()
        sar_grid_samples.append(s1geo.sar_grid_sample)
        s1geo_sar_grid_line = np.array(s1geo.sar_grid_line)
        sar_grid_lines.append(s1geo_sar_grid_line + first_line)
        sar_grid_latitude.append(s1geo.sar_grid_latitude)
        sar_grid_longitude.append(s1geo.sar_grid_longitude)
        first_line += s1geo.raw_shape[0]
    return arrays, dates, sar_grid_samples, sar_grid_lines, sar_grid_latitude, sar_grid_longitude

def read_s1_safe_zip(ifile, pol='HH', **kwargs):
    n = Nansat(ifile)
    cal_avg = n.vrt.band_vrts[f'sigmaNought_{pol}'].vrt.dataset.ReadAsArray().mean()
    dn = n[f'DN_{pol}']
    sigma0 = dn.astype(float)**2 / cal_avg**2
    sigma0angle = np.linspace(2, -2, sigma0.shape[1])[None]
    sigma0db = 10 * np.log10(sigma0) - sigma0angle
    return n, sigma0db

def stack_safe_zip_files(filenames, **kwargs):
    arrays = []
    nansats = []
    dates = []
    for filename in filenames:
        print('Input:', filename)
        n, sigma0db = read_s1_safe_zip(filename, **kwargs)
        arrays.append(sigma0db)
        nansats.append(n)
        dates.append([n.time_coverage_start, n.time_coverage_end])
    first_line = 0
    sar_grid_samples = []
    sar_grid_lines = []
    sar_grid_latitude = []
    sar_grid_longitude = []

    for n, a in zip(nansats, arrays):
        gcps = n.vrt.dataset.GetGCPs()
        sar_grid_samples.extend([gcp.GCPPixel for gcp in gcps])
        sar_grid_lines.extend([gcp.GCPLine + first_line for gcp in gcps])
        sar_grid_latitude.extend([gcp.GCPY for gcp in gcps])
        sar_grid_longitude.extend([gcp.GCPX for gcp in gcps])
        first_line += a.shape[0]
    return arrays, dates, sar_grid_samples, sar_grid_lines, sar_grid_latitude, sar_grid_longitude

def get_stacked_nansat(filenames, stack_function):
    arrays, dates, sar_grid_samples, sar_grid_lines, sar_grid_latitude, sar_grid_longitude = stack_function(filenames)

    min_width = min(array.shape[1] for array in arrays)
    cropped_arrays = [array[:, :min_width] for array in arrays]
    stacked_array = np.vstack(cropped_arrays)

    sar_grid_samples = np.hstack(sar_grid_samples)
    sar_grid_lines = np.hstack(sar_grid_lines)
    sar_grid_latitude = np.hstack(sar_grid_latitude)
    sar_grid_longitude = np.hstack(sar_grid_longitude)

    # Create a list to hold the GCPs
    gcps = []
    # Iterate over the grid points and create GCPs
    for sample, line, lat, lon in zip(sar_grid_samples, sar_grid_lines, sar_grid_latitude, sar_grid_longitude):
        gcps.append(gdal.GCP(lon, lat, 0, sample, line))

    # Print the number of GCPs created
    print(f"Number of GCPs created: {len(gcps)}")

    x_size = stacked_array.shape[1]
    y_size = stacked_array.shape[0]
    vrttmp = VRT2.from_dataset_params(x_size, y_size, None, None, gcps=gcps, gcp_projection=NSR().ExportToWkt())
    d = Domain(ds=vrttmp.dataset)
    n = Nansat.from_domain(d, stacked_array)
    n.vrt.tps = True
    n.reproject_gcps()
    return n, dates[0][0], dates[-1][1]

def get_pm0_grids(dst_dom, pm_step, template_size):
    dst_shape = dst_dom.shape()

    c0pm_vec = np.arange(0, dst_shape[1], pm_step)
    r0pm_vec = np.arange(0, dst_shape[0], pm_step)
    c0pm, r0pm = np.meshgrid(c0pm_vec, r0pm_vec)
    x0pm, y0pm = dst_dom.transform_points(c0pm.ravel(), r0pm.ravel(), DstToSrc=0, dst_srs=NSR(dst_dom.vrt.get_projection()[0]))
    x0pm = x0pm.reshape(c0pm.shape)
    y0pm = y0pm.reshape(r0pm.shape)

    g0_pm = (
        (c0pm > template_size) *
        (c0pm < dst_shape[1] - template_size) *
        (r0pm > template_size) *
        (r0pm < dst_shape[0] - template_size)
    )
    return c0pm, r0pm, x0pm, y0pm, g0_pm

def get_1pmfg_grid_wo_ft(dst_dom, c0pm, r0pm, x0pm, y0pm, g0_pm, template_size, border):
    dst_shape = dst_dom.shape()
    x1pmfg, y1pmfg, c1pmfg, r1pmfg = x0pm, y0pm, c0pm, r0pm
    img_size = template_size + border * 2
    g1_pm = (
        g0_pm *
        (c1pmfg > img_size) *
        (c1pmfg < dst_shape[1] - img_size) *
        (r1pmfg > img_size) *
        (r1pmfg < dst_shape[0] - img_size)
    )
    return c1pmfg, r1pmfg, x1pmfg, y1pmfg, g1_pm

def get_landmask(landmask_file, dst_dom):
    if os.path.exists(landmask_file):
        landmask = np.load(landmask_file)['landmask']
    else:
        landmask = Nansat.from_domain(dst_dom).watermask()[1] == 2
        np.savez_compressed(landmask_file, landmask=landmask)
    return landmask

class PatternMatcher:
    def __init__(self, conf, cores=10):
        self.hs = conf.pm_template_size // 2
        self.border = conf.pm_border
        self.step = conf.pm_step
        self.sar_resolution = conf.sar_resolution
        self.min_time_delta = conf.pm_min_time_delta
        self.template_size = conf.pm_template_size
        self.cores = cores

    def set_initial_grid(self, dst_dom):
        self.c0pm, self.r0pm, x0pm, y0pm, g0_pm = get_pm0_grids(dst_dom, self.step, self.template_size)
        self.c1pmfg, self.r1pmfg, _, _, self.g1_pm = get_1pmfg_grid_wo_ft(dst_dom, self.c0pm, self.r0pm, x0pm, y0pm, g0_pm, self.template_size, self.border)

    def process_element(self, args):
        c0, r0, c1, r1 = args
        template = self.array0[r0-self.hs:r0+self.hs+1, c0-self.hs:c0+self.hs+1]
        image2 = self.array1[r1-self.hs-self.border:r1+self.hs+1+self.border, c1-self.hs-self.border:c1+self.hs+1+self.border]
        result = cv2.matchTemplate(image2, template, cv2.TM_CCOEFF_NORMED)
        mccr, mccc = np.unravel_index(result.argmax(), result.shape)
        dr, dc = mccr - self.border, mccc - self.border
        mcc = result[mccr, mccc]
        return [dc, dr, mcc]

    def pattern_matching(self, c0, r0, c1, r1):
        args = np.column_stack([c0, r0, c1, r1])
        t0 = time.time()
        with Pool(self.cores) as pool:
            corrections = pool.map(self.process_element, args)
        print('Elapsed time:', time.time() - t0)
        return np.array(corrections)

    def scale_array(self, array, percentiles):
        p1, p99 = np.nanpercentile(array, percentiles)
        return np.nan_to_num(np.clip((array - p1) / (p99 - p1) * 255, 0, 255), nan=0).astype(np.uint8)

    def get_drift(self, last_mosaic, new_mosaic, mosaic_time_diff):
        self.array0 = self.scale_array(last_mosaic, (0.5, 99.5))
        self.array1 = self.scale_array(new_mosaic, (0.5, 99.5))
        gpi = (
            self.g1_pm *
            np.isfinite(last_mosaic[self.r0pm, self.c0pm]) *
            np.isfinite(new_mosaic[self.r0pm, self.c0pm]) *
            (mosaic_time_diff[self.r0pm, self.c0pm] >= self.min_time_delta)
        )
        print('Number of valid pixels:', gpi.sum())

        corrections = self.pattern_matching(self.c0pm[gpi], self.r0pm[gpi], self.c1pmfg[gpi], self.r1pmfg[gpi])
        drift_col = np.zeros(self.c0pm.shape) + np.nan
        drift_row = np.zeros(self.c0pm.shape) + np.nan
        drift_cor = np.zeros(self.c0pm.shape) + np.nan
        if gpi.sum() > 0:
            drift_col[gpi] = corrections[:, 0]
            drift_row[gpi] = corrections[:, 1]
            drift_cor[gpi] = corrections[:, 2]

        drift_col = drift_col * self.sar_resolution * self.step / mosaic_time_diff[self.r0pm, self.c0pm]
        drift_row = drift_row * self.sar_resolution * self.step / mosaic_time_diff[self.r0pm, self.c0pm]
        return drift_col, drift_row, drift_cor

def create_destination_domain(conf):
    dst_dom = Domain(NSR(conf.proj4), f'-te {conf.extent} -tr {conf.sar_resolution} {conf.sar_resolution}')
    print('Destination domain size:', dst_dom.shape())
    dst_b_lon, dst_b_lat = dst_dom.get_border()
    dst_b_x, dst_b_y = pyproj.Proj(conf.proj4)(dst_b_lon, dst_b_lat, inverse=False)
    landmask = get_landmask(conf.landmask_file, dst_dom)
    dst_polygon = Polygon(zip(dst_b_x, dst_b_y))
    dst_polygon = gpd.GeoDataFrame(index=[0], crs=conf.proj4, geometry=[dst_polygon])
    return dst_dom, landmask, dst_polygon

def read_s1_sar_gdf(conf):
    s1_sar_proj4str = '+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
    gdf = gpd.read_feather(conf.s1_sar_gdf_file).set_crs(s1_sar_proj4str).to_crs(conf.proj4)
    gdf.index = gdf['date']
    gdf = gdf.sort_index(ascending=True)
    return gdf

def get_overlapping_gdf(conf, gdf, dst_polygon, plot=False):
    overlapping_gdf = gdf[gdf.intersects(dst_polygon.geometry[0])]
    overlapping_gdf = overlapping_gdf[overlapping_gdf.apply(lambda row: row.geometry.intersection(dst_polygon.geometry[0]).area / row.geometry.area > 0.1, axis=1)]
    print('Number of overlapping SAR scenes:', len(overlapping_gdf), 'Orbits:', len(overlapping_gdf['orbit'].unique()))

    if plot:
        # Create a new figure and axis with Cartopy projection
        cartopy_crs = ccrs.Projection(pyproj.CRS(conf.proj4))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': ccrs.NorthPolarStereo(-45)})
        dst_polygon.plot(ax=ax, color='blue', edgecolor='blue', alpha=0.3, transform=cartopy_crs)
        overlapping_gdf.plot(ax=ax, color='None', edgecolor='black', alpha=0.1, transform=cartopy_crs)
        ax.add_feature(cfeature.LAND, edgecolor='black')
        plt.show()
    return overlapping_gdf

class LoopProcessor:
    def __init__(self, conf, dst_dom, landmask, overlapping_gdf, period_start_date, stack_function):
        self.conf = conf
        self.dst_dom = dst_dom
        self.landmask = landmask
        self.overlapping_gdf = overlapping_gdf
        self.period_start_date = period_start_date
        self.stack_function = stack_function

    def loop(self, max_iter=None):
        state = 1
        counter = 0
        while state > 0:
            if max_iter is not None and counter >= max_iter:
                break
            state = self.get_state()
            self.forward(state)
            counter += 1

    def forward(self, state):
        if state == 1:
            self.save_first_mosaic()
        elif state == 2:
            self.create_mosaics()
            self.compute_drift()
            self.merge_mosaics()

    def get_state(self, plot=False, dst_polygon=None):
        self.last_mosaic_file, last_mosaic_date = self.get_last_mosaic_date()
        print(self.last_mosaic_file, last_mosaic_date)

        # Filter records with date larger than last_product_date
        filtered_gdf = self.overlapping_gdf[self.overlapping_gdf.index > last_mosaic_date]
        unique_orbits = filtered_gdf['orbit'].unique()

        if len(unique_orbits) == 0:
            print('No new orbits')
            # No processing
            self.last_mosaic_file = None
            self.orbit_date = None
            self.orbit_names = None
            return 0

        print('Number of unique orbits for processing:', len(unique_orbits))
        orbit = unique_orbits[0]
        orbit_gdf = filtered_gdf[filtered_gdf.orbit == orbit]
        self.orbit_date = filtered_gdf[filtered_gdf.orbit == orbit].index.min()
        self.orbit_names = orbit_gdf['name'].values

        if plot:
            cartopy_crs = ccrs.Projection(pyproj.CRS(self.conf.proj4))
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={'projection': ccrs.NorthPolarStereo(-45)})
            dst_polygon.plot(ax=ax, color='None', edgecolor='blue', transform=cartopy_crs)
            orbit_gdf.plot(ax=ax, color='None', edgecolor='black', transform=cartopy_crs)
            ax.add_feature(cfeature.LAND, edgecolor='black')
            plt.show()

        if self.last_mosaic_file is None:
            # First mosaic
            state = 1
        elif self.last_mosaic_file is not None and last_mosaic_date < self.orbit_date:
            # New orbit processing (drift and mosaic)
            state = 2
        return state

    def get_last_mosaic_date(self):
        product_files = sorted(glob.glob(f'{self.conf.odir}/mosaic_????????T??????_????????T??????.npz'))
        if len(product_files) > 0:
            last_product_file = product_files[-1]
        else:
            last_product_file = None

        if last_product_file is None:
            last_product_date = self.period_start_date
        else:
            last_product_date = datetime.strptime(last_product_file.split('.')[-2].split('_')[-1], '%Y%m%dT%H%M%S')
        return last_product_file, last_product_date

    def save_first_mosaic(self):
        print('The first mosaic processing')
        mosaic, mosaic_time, date0, date1 = self.warp_orbits()
        mosaic_file = f'{self.conf.odir}/mosaic_{date0.strftime("%Y%m%dT%H%M%S")}_{date1.strftime("%Y%m%dT%H%M%S")}.npz'
        np.savez_compressed(mosaic_file, **{self.conf.band_name: mosaic.astype(np.float32), 'time': mosaic_time})
        print('Saved mosaic:', mosaic_file)

    def warp_orbits(self):
        n, date0, date1 = get_stacked_nansat(self.orbit_names, self.stack_function)
        mosaic = warp(n, n[1], self.dst_dom, cval=self.conf.warp_cval)
        mosaic[self.landmask] = np.nan
        mosaic_time = np.full(mosaic.shape, np.int32((date1 - self.period_start_date).total_seconds()))
        mosaic_time[np.isnan(mosaic)] = 0
        return mosaic, mosaic_time, date0, date1

    def create_mosaics(self):
        with np.load(self.last_mosaic_file) as data:
            self.last_mosaic = data[self.conf.band_name]
            self.last_mosaic_time = data['time']
        # get new mosaic
        self.new_mosaic, self.new_mosaic_time, self.date0, self.date1 = self.warp_orbits()
        self.mosaic_time_diff = self.new_mosaic_time - self.last_mosaic_time
        self.mosaic_time_diff[~((self.new_mosaic_time > 0) & (self.last_mosaic_time > 0))] = 0

    def compute_drift(self):
        pm = PatternMatcher(self.conf, 4)
        pm.set_initial_grid(self.dst_dom)
        drift_col, drift_row, drift_cor = pm.get_drift(self.last_mosaic, self.new_mosaic, self.mosaic_time_diff)
        # saved drift
        drift_file = f'{self.conf.odir}/drift_{self.date0.strftime("%Y%m%dT%H%M%S")}_{self.date1.strftime("%Y%m%dT%H%M%S")}.npz'
        np.savez(drift_file, drift_col=drift_col, drift_row=drift_row, drift_cor=drift_cor)
        print('Save drift:', drift_file)

    def merge_mosaics(self):
        mosaic = np.array(self.last_mosaic)
        mosaic[np.isfinite(self.new_mosaic)] = self.new_mosaic[np.isfinite(self.new_mosaic)]
        mosaic_time = np.array(self.last_mosaic_time)
        mosaic_time[self.new_mosaic_time > 0] = self.new_mosaic_time[self.new_mosaic_time > 0]
        mosaic_file = f'{self.conf.odir}/mosaic_{self.date0.strftime("%Y%m%dT%H%M%S")}_{self.date1.strftime("%Y%m%dT%H%M%S")}.npz'
        np.savez_compressed(mosaic_file, **{self.conf.band_name: mosaic.astype(np.float32), 'time': mosaic_time})
        print('Save mosaic:', mosaic_file)