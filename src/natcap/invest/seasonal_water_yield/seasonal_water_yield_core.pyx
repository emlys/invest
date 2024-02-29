# cython: profile=False
# cython: language_level=2
import logging
import os
import collections
import sys
import gc
import pygeoprocessing

import numpy
cimport numpy
cimport cython
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from libcpp.pair cimport pair
from libcpp.stack cimport stack
from libcpp.queue cimport queue
from libc.time cimport time as ctime
from ..managed_raster.managed_raster cimport _ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport route

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

cdef int N_MONTHS = 12


cpdef calculate_local_recharge(
        precip_path_list, et0_path_list, qf_m_path_list, flow_dir_mfd_path,
        kc_path_list, alpha_month_map, float beta_i, float gamma, stream_path,
        target_li_path, target_li_avail_path, target_l_sum_avail_path,
        target_aet_path, target_pi_path):
    """
    Calculate the rasters defined by equations [3]-[7].

    Note all input rasters must be in the same coordinate system and
    have the same dimensions.

    Args:
        precip_path_list (list): list of paths to monthly precipitation
            rasters. (model input)
        et0_path_list (list): path to monthly ET0 rasters. (model input)
        qf_m_path_list (list): path to monthly quickflow rasters calculated by
            Equation [1].
        flow_dir_mfd_path (str): path to a PyGeoprocessing Multiple Flow
            Direction raster indicating flow directions for this analysis.
        alpha_month_map (dict): fraction of upslope annual available recharge
            that is available in month m (indexed from 1).
        beta_i (float):  fraction of the upgradient subsidy that is available
            for downgradient evapotranspiration.
        gamma (float): the fraction of pixel recharge that is available to
            downgradient pixels.
        stream_path (str): path to the stream raster where 1 is a stream,
            0 is not, and nodata is outside of the DEM.
        kc_path_list (str): list of rasters of the monthly crop factor for the
            pixel.
        target_li_path (str): created by this call, path to local recharge
            derived from the annual water budget. (Equation 3).
        target_li_avail_path (str): created by this call, path to raster
            indicating available recharge to a pixel.
        target_l_sum_avail_path (str): created by this call, the recursive
            upslope accumulation of target_li_avail_path.
        target_aet_path (str): created by this call, the annual actual
            evapotranspiration.
        target_pi_path (str): created by this call, the annual precipitation on
            a pixel.

        Returns:
            None.

    """
    cdef int i_n, flow_dir_nodata, flow_dir_mfd
    cdef int peak_pixel
    cdef long xs, ys, xs_root, ys_root, xoff, yoff
    cdef int flow_dir_s
    cdef long xi, yi, xj, yj
    cdef int flow_dir_j, p_ij_base
    cdef int n_dir
    cdef long raster_x_size, raster_y_size, win_xsize, win_ysize
    cdef double pet_m, p_m, qf_m, et0_m, aet_i, p_i, qf_i, l_i, l_avail_i
    cdef float qf_nodata, kc_nodata

    cdef float mfd_direction_array[8]

    cdef queue[pair[long, long]] work_queue
    cdef _ManagedRaster et0_m_raster, qf_m_raster, kc_m_raster

    cdef numpy.ndarray[numpy.npy_float32, ndim=1] alpha_month_array = (
        numpy.array(
            [x[1] for x in sorted(alpha_month_map.items())],
            dtype=numpy.float32))

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # we know the PyGeoprocessing MFD raster flow dir type is a 32 bit int.
    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_mfd_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']
    cdef ManagedFlowDirRaster flow_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path, 1, 0)

    # make sure that user input nodata values are defined
    # set to -1 if not defined
    # precipitation and evapotranspiration data should
    # always be non-negative
    et0_m_raster_list = []
    et0_m_nodata_list = []
    for et0_path in et0_path_list:
        et0_m_raster_list.append(_ManagedRaster(et0_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(et0_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        et0_m_nodata_list.append(nodata)

    precip_m_raster_list = []
    precip_m_nodata_list = []
    for precip_m_path in precip_path_list:
        precip_m_raster_list.append(_ManagedRaster(precip_m_path, 1, 0))
        nodata = pygeoprocessing.get_raster_info(precip_m_path)['nodata'][0]
        if nodata is None:
            nodata = -1
        precip_m_nodata_list.append(nodata)

    qf_m_raster_list = []
    qf_m_nodata_list = []
    for qf_m_path in qf_m_path_list:
        qf_m_raster_list.append(_ManagedRaster(qf_m_path, 1, 0))
        qf_m_nodata_list.append(
            pygeoprocessing.get_raster_info(qf_m_path)['nodata'][0])

    kc_m_raster_list = []
    kc_m_nodata_list = []
    for kc_m_path in kc_path_list:
        kc_m_raster_list.append(_ManagedRaster(kc_m_path, 1, 0))
        kc_m_nodata_list.append(
            pygeoprocessing.get_raster_info(kc_m_path)['nodata'][0])

    target_nodata = -1e32
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_raster = _ManagedRaster(
        target_li_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_li_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_li_avail_raster = _ManagedRaster(
        target_li_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_l_sum_avail_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    cdef _ManagedRaster target_l_sum_avail_raster = _ManagedRaster(
        target_l_sum_avail_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_aet_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_aet_raster = _ManagedRaster(
        target_aet_path, 1, 1)

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_pi_path, gdal.GDT_Float32, [target_nodata],
        fill_value_list=[target_nodata])
    cdef _ManagedRaster target_pi_raster = _ManagedRaster(
        target_pi_path, 1, 1)


    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_mfd_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                'peak point detection %.2f%% complete',
                100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))

        # search block for a peak pixel where no other pixel drains to it.
        for ys in xrange(win_ysize):
            ys_root = yoff + ys
            for xs in xrange(win_xsize):
                xs_root = xoff + xs

                if flow_raster.is_local_high_point(xs_root, ys_root):
                    work_queue.push(
                        pair[long, long](xs_root, ys_root))

                while work_queue.size() > 0:
                    xi = work_queue.front().first
                    yi = work_queue.front().second
                    work_queue.pop()

                    l_sum_avail_i = target_l_sum_avail_raster.get(xi, yi)
                    if not is_close(l_sum_avail_i, target_nodata):
                        # already defined
                        continue

                    # Equation 7, calculate L_sum_avail_i if possible, skip
                    # otherwise
                    upslope_defined = 1
                    # initialize to 0 so we indicate we haven't tracked any
                    # mfd values yet
                    l_sum_avail_i = 0
                    for neighbor in flow_raster.get_upslope_neighbors(xi, yi):
                        # pixel flows inward, check upslope
                        l_sum_avail_j = target_l_sum_avail_raster.get(
                            neighbor.x, neighbor.y)
                        if is_close(l_sum_avail_j, target_nodata):
                            upslope_defined = 0
                            break
                        l_avail_j = target_li_avail_raster.get(
                            neighbor.x, neighbor.y)
                        # A step of Equation 7
                        l_sum_avail_i += (
                            l_sum_avail_j + l_avail_j) * neighbor.flow_proportion
                    # calculate l_sum_avail_i by summing all the valid
                    # directions then normalizing by the sum of the mfd
                    # direction weights (Equation 8)
                    if upslope_defined:
                        # Equation 7
                        target_l_sum_avail_raster.set(xi, yi, l_sum_avail_i)
                    else:
                        # if not defined, we'll get it on another pass
                        continue

                    aet_i = 0
                    p_i = 0
                    qf_i = 0

                    for m_index in range(12):
                        precip_m_raster = (
                            <_ManagedRaster?>precip_m_raster_list[m_index])
                        qf_m_raster = (
                            <_ManagedRaster?>qf_m_raster_list[m_index])
                        et0_m_raster = (
                            <_ManagedRaster?>et0_m_raster_list[m_index])
                        kc_m_raster = (
                            <_ManagedRaster?>kc_m_raster_list[m_index])

                        et0_nodata = et0_m_nodata_list[m_index]
                        precip_nodata = precip_m_nodata_list[m_index]
                        qf_nodata = qf_m_nodata_list[m_index]
                        kc_nodata = kc_m_nodata_list[m_index]

                        p_m = precip_m_raster.get(xi, yi)
                        if not is_close(p_m, precip_nodata):
                            p_i += p_m
                        else:
                            p_m = 0

                        qf_m = qf_m_raster.get(xi, yi)
                        if not is_close(qf_m, qf_nodata):
                            qf_i += qf_m
                        else:
                            qf_m = 0

                        kc_m = kc_m_raster.get(xi, yi)
                        pet_m = 0
                        et0_m = et0_m_raster.get(xi, yi)
                        if not (
                                is_close(kc_m, kc_nodata) or
                                is_close(et0_m, et0_nodata)):
                            # Equation 6
                            pet_m = kc_m * et0_m

                        # Equation 4/5
                        aet_i += min(
                            pet_m,
                            p_m - qf_m +
                            alpha_month_array[m_index]*beta_i*l_sum_avail_i)

                    l_i = (p_i - qf_i - aet_i)
                    l_avail_i = min(gamma * l_i, l_i)

                    target_pi_raster.set(xi, yi, p_i)
                    target_aet_raster.set(xi, yi, aet_i)
                    target_li_raster.set(xi, yi, l_i)
                    target_li_avail_raster.set(xi, yi, l_avail_i)

                    for neighbor in flow_raster.get_downslope_neighbors(xi, yi):
                        work_queue.push(pair[long, long](neighbor.x, neighbor.y))


def route_baseflow_sum(
        flow_dir_mfd_path, l_path, l_avail_path, l_sum_path,
        stream_path, target_b_path, target_b_sum_path):
    """Route Baseflow through MFD as described in Equation 11.

    Args:
        flow_dir_mfd_path (string): path to a pygeoprocessing multiple flow
            direction raster.
        l_path (string): path to local recharge raster.
        l_avail_path (string): path to local recharge raster that shows
            recharge available to the pixel.
        l_sum_path (string): path to upslope sum of l_path.
        stream_path (string): path to stream raster, 1 stream, 0 no stream,
            and nodata.
        target_b_path (string): path to created raster for per-pixel baseflow.
        target_b_sum_path (string): path to created raster for per-pixel
            upslope sum of baseflow.

    Returns:
        None.
    """
    cdef float target_nodata = -1e32

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_sum_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_path, target_b_path, gdal.GDT_Float32,
        [target_nodata], fill_value_list=[target_nodata])

    cdef _ManagedRaster target_b_sum_raster = _ManagedRaster(
        target_b_sum_path, 1, 1)
    cdef _ManagedRaster target_b_raster = _ManagedRaster(
        target_b_path, 1, 1)
    cdef _ManagedRaster l_raster = _ManagedRaster(l_path, 1, 0)
    cdef _ManagedRaster l_avail_raster = _ManagedRaster(l_avail_path, 1, 0)
    cdef _ManagedRaster l_sum_raster = _ManagedRaster(l_sum_path, 1, 0)
    cdef ManagedFlowDirRaster flow_dir_mfd_raster = ManagedFlowDirRaster(
        flow_dir_mfd_path, 1, 0)
    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, 0)


    def seed_fn(int x, int y):
        """Determine whether the pixel at (x, y) is a seed pixel.

        Args:
            x (int): x coordinate of the pixel
            y (int): y coordinate of the pixel

        Returns:
            True if the pixel qualifies as a seed pixel, False otherwise
        """
        cdef int stream_val
        cdef int flow_dir_s = <int>flow_dir_mfd_raster.get(x, y)
        if is_close(flow_dir_s, flow_dir_mfd_raster.nodata):
            return False

        # search for a pixel that has no downslope neighbors,
        # or whose downslope neighbors all have nodata in the stream raster (?)
        for neighbor in flow_dir_mfd_raster.get_downslope_neighbors(x, y):
            stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
            if stream_val != stream_raster.nodata:
                return False
        return True


    def route_fn(xi, yi):
        cdef int stream_val
        cdef float b_i, b_sum_i, l_j, l_avail_j, l_sum_j

        to_push = []
        b_sum_i = target_b_sum_raster.get(xi, yi)
        if not is_close(b_sum_i, target_nodata):
            return []

        b_sum_i = 0.0
        downslope_defined = 1
        for neighbor in flow_dir_mfd_raster.get_downslope_neighbors(xi, yi):
            stream_val = <int>stream_raster.get(neighbor.x, neighbor.y)
            if stream_val:
                b_sum_i += neighbor.flow_proportion
            else:
                b_sum_j = target_b_sum_raster.get(neighbor.x, neighbor.y)
                if is_close(b_sum_j, target_nodata):
                    downslope_defined = 0
                    break
                l_j = l_raster.get(neighbor.x, neighbor.y)
                l_avail_j = l_avail_raster.get(neighbor.x, neighbor.y)
                l_sum_j = l_sum_raster.get(neighbor.x, neighbor.y)

                if l_sum_j != 0 and (l_sum_j - l_j) != 0:
                    b_sum_i += neighbor.flow_proportion * (
                        (1 - l_avail_j / l_sum_j) * (
                            b_sum_j / (l_sum_j - l_j)))
                else:
                    b_sum_i += neighbor.flow_proportion

        if not downslope_defined:
            return []

        l_i = l_raster.get(xi, yi)
        l_sum_i = l_sum_raster.get(xi, yi)
        b_sum_i = l_sum_i * b_sum_i

        if l_sum_i != 0:
            b_i = max(b_sum_i * l_i / l_sum_i, 0.0)
        else:
            b_i = 0.0

        target_b_raster.set(xi, yi, b_i)
        target_b_sum_raster.set(xi, yi, b_sum_i)

        for neighbor in flow_dir_mfd_raster.get_upslope_neighbors(xi, yi):
            to_push.append(neighbor.y * flow_dir_mfd_raster.raster_x_size + neighbor.x)

        return to_push

    route(flow_dir_mfd_path, seed_fn, route_fn)

