# cython: profile=False
# cython: language_level=2
import tempfile
import logging
import os
import collections

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from libcpp.stack cimport stack
from libc.stdlib cimport free
from libc.math cimport exp
from ..managed_raster.managed_raster cimport _ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport INFLOW_OFFSETS
from ..managed_raster.managed_raster cimport NeighborTuple

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

# Within a stream, the effective retention is 0
cdef int STREAM_EFFECTIVE_RETENTION = 0

def ndr_eff_calculation(
        mfd_flow_direction_path, stream_path, retention_eff_lulc_path,
        crit_len_path, effective_retention_path):
    """Calculate flow downhill effective_retention to the channel.

        Args:
            mfd_flow_direction_path (string): a path to a raster with
                pygeoprocessing.routing MFD flow direction values.
            stream_path (string): a path to a raster where 1 indicates a
                stream all other values ignored must be same dimensions and
                projection as mfd_flow_direction_path.
            retention_eff_lulc_path (string): a path to a raster indicating
                the maximum retention efficiency that the landcover on that
                pixel can accumulate.
            crit_len_path (string): a path to a raster indicating the critical
                length of the retention efficiency that the landcover on this
                pixel.
            effective_retention_path (string): path to a raster that is
                created by this call that contains a per-pixel effective
                sediment retention to the stream.

        Returns:
            None.

    """
    cdef float effective_retention_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, effective_retention_path, gdal.GDT_Float32,
        [effective_retention_nodata])
    fp, to_process_flow_directions_path = tempfile.mkstemp(
        suffix='.tif', prefix='flow_to_process',
        dir=os.path.dirname(effective_retention_path))
    os.close(fp)

    cdef long n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_cols, n_rows = flow_dir_info['raster_size']

    cdef stack[long] processing_stack
    stream_info = pygeoprocessing.get_raster_info(stream_path)
    # cell sizes must be square, so no reason to test at this point.
    cdef float cell_size = abs(stream_info['pixel_size'][0])

    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, False)
    cdef _ManagedRaster crit_len_raster = _ManagedRaster(
        crit_len_path, 1, False)
    cdef float crit_len_nodata = pygeoprocessing.get_raster_info(
        crit_len_path)['nodata'][0]
    cdef _ManagedRaster retention_eff_lulc_raster = _ManagedRaster(
        retention_eff_lulc_path, 1, False)
    cdef float retention_eff_nodata = pygeoprocessing.get_raster_info(
        retention_eff_lulc_path)['nodata'][0]
    cdef _ManagedRaster effective_retention_raster = _ManagedRaster(
        effective_retention_path, 1, True)
    cdef ManagedFlowDirRaster mfd_flow_direction_raster = ManagedFlowDirRaster(
        mfd_flow_direction_path, 1, False)

    # create direction raster in bytes
    def _mfd_to_flow_dir_op(mfd_array):
        result = numpy.zeros(mfd_array.shape, dtype=numpy.int8)
        for i in range(8):
            result[:] |= (((mfd_array >> (i*4)) & 0xF) > 0) << i
        return result

    # convert mfd raster to binary mfd
    # each value is an 8-digit binary number
    # where 1 indicates that the pixel drains in that direction
    # and 0 indicates that it does not drain in that direction
    pygeoprocessing.raster_calculator(
        [(mfd_flow_direction_path, 1)], _mfd_to_flow_dir_op,
        to_process_flow_directions_path, gdal.GDT_Byte, None)

    cdef _ManagedRaster to_process_flow_directions_raster = _ManagedRaster(
        to_process_flow_directions_path, 1, True)

    cdef long col_index, row_index, win_xsize, win_ysize, xoff, yoff
    cdef long global_col, global_row
    cdef unsigned long flat_index
    cdef long outflow_weight, flow_dir
    cdef long ds_col, ds_row, i
    cdef float current_step_factor, step_size, crit_len
    cdef long neighbor_row, neighbor_col
    cdef int neighbor_outflow_dir, neighbor_outflow_dir_mask, neighbor_process_flow_dir
    cdef int outflow_dirs, dir_mask

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        for row_index in range(win_ysize):
            global_row = yoff + row_index
            for col_index in range(win_xsize):
                global_col = xoff + col_index
                outflow_dirs = <int>to_process_flow_directions_raster.get(
                    global_col, global_row)
                should_seed = 0
                # see if this pixel drains to nodata or the edge, if so it's
                # a drain
                downslope_neighbors_array = (
                    mfd_flow_direction_raster.get_downslope_neighbors(
                        global_col, global_row, skip_oob=False))
                length = sizeof(downslope_neighbors_array) /  sizeof(NeighborTuple)
                for neighbor_idx in range(length):
                    neighbor = downslope_neighbors_array[neighbor_idx]

                    if (neighbor.x < 0 or neighbor.x >= n_cols or
                        neighbor.y < 0 or neighbor.y >= n_rows or
                        to_process_flow_directions_raster.get(
                            neighbor.x, neighbor.y) == 0):
                        should_seed = 1
                        outflow_dirs &= ~(1 << neighbor.direction)
                free(downslope_neighbors_array)
                if should_seed:
                    # mark all outflow directions processed
                    to_process_flow_directions_raster.set(
                        global_col, global_row, outflow_dirs)
                    processing_stack.push(global_row * n_cols + global_col)

        while processing_stack.size() > 0:
            # loop invariant, we don't push a cell on the stack that
            # hasn't already been set for processing.
            flat_index = processing_stack.top()
            processing_stack.pop()
            global_row = flat_index / n_cols
            global_col = flat_index % n_cols

            crit_len = <float>crit_len_raster.get(global_col, global_row)
            retention_eff_lulc = retention_eff_lulc_raster.get(
                global_col, global_row)
            flow_dir = <int>mfd_flow_direction_raster.get(
                    global_col, global_row)
            if stream_raster.get(global_col, global_row) == 1:
                # if it's a stream effective retention is 0.
                effective_retention_raster.set(global_col, global_row, STREAM_EFFECTIVE_RETENTION)
            elif (is_close(crit_len, crit_len_nodata) or
                  is_close(retention_eff_lulc, retention_eff_nodata) or
                  flow_dir == 0):
                # if it's nodata, effective retention is nodata.
                effective_retention_raster.set(
                    global_col, global_row, effective_retention_nodata)
            else:
                working_retention_eff = 0.0
                has_outflow = False
                downslope_neighbors_array = (
                    mfd_flow_direction_raster.get_downslope_neighbors(
                        global_col, global_row, skip_oob=False))
                length = sizeof(downslope_neighbors_array) /  sizeof(NeighborTuple)
                for neighbor_idx in range(length):
                    neighbor = downslope_neighbors_array[neighbor_idx]

                    has_outflow = True
                    if (neighbor.x < 0 or neighbor.x >= n_cols or
                        neighbor.y < 0 or neighbor.y >= n_rows):
                        continue

                    if neighbor.direction % 2 == 1:
                        step_size = <float>(cell_size * 1.41421356237)
                    else:
                        step_size = cell_size
                    # guard against a critical length factor that's 0
                    if crit_len > 0:
                        current_step_factor = <float>(
                            exp(-5 * step_size / crit_len))
                    else:
                        current_step_factor = 0.0

                    neighbor_effective_retention = (
                        effective_retention_raster.get(
                            neighbor.x, neighbor.y))

                    # Case 1: downslope neighbor is a stream pixel
                    if neighbor_effective_retention == STREAM_EFFECTIVE_RETENTION:
                        intermediate_retention = (
                            retention_eff_lulc * (1 - current_step_factor))

                    # Case 2: the current LULC's retention exceeds the neighbor's retention.
                    elif retention_eff_lulc > neighbor_effective_retention:
                        intermediate_retention = (
                            (neighbor_effective_retention * current_step_factor) +
                            (retention_eff_lulc * (1 - current_step_factor)))

                    # Case 3: the other 2 cases have not been hit.
                    else:
                        intermediate_retention = neighbor_effective_retention

                    working_retention_eff += (
                        intermediate_retention * neighbor.flow_proportion)

                free(downslope_neighbors_array)
                if has_outflow:
                    effective_retention_raster.set(
                        global_col, global_row, working_retention_eff)
                else:
                    raise Exception("got to a cell that has no outflow!")
            # search upslope to see if we need to push a cell on the stack
            upslope_neighbors_array = (
                mfd_flow_direction_raster.get_upslope_neighbors(
                    global_col, global_row))
            length = sizeof(upslope_neighbors_array) /  sizeof(NeighborTuple)
            for neighbor_idx in range(length):
                neighbor = upslope_neighbors_array[neighbor_idx]

                neighbor_outflow_dir = INFLOW_OFFSETS[neighbor.direction]
                neighbor_outflow_dir_mask = 1 << neighbor_outflow_dir
                neighbor_process_flow_dir = <int>(
                    to_process_flow_directions_raster.get(
                        neighbor.x, neighbor.y))
                if neighbor_process_flow_dir == 0:
                    # skip, due to loop invariant this must be a nodata pixel
                    continue
                if neighbor_process_flow_dir & neighbor_outflow_dir_mask == 0:
                    # no outflow
                    continue
                # mask out the outflow dir that this iteration processed
                neighbor_process_flow_dir &= ~neighbor_outflow_dir_mask
                to_process_flow_directions_raster.set(
                    neighbor.x, neighbor.y, neighbor_process_flow_dir)
                if neighbor_process_flow_dir == 0:
                    # if 0 then all downslope have been processed,
                    # push on stack, otherwise another downslope pixel will
                    # pick it up
                    processing_stack.push(neighbor.y * n_cols + neighbor.x)
            free(upslope_neighbors_array)
    to_process_flow_directions_raster.close()
    os.remove(to_process_flow_directions_path)
