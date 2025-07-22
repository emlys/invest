"""InVEST Sediment Delivery Ratio (SDR) module.

The SDR method in this model is based on:
    Winchell, M. F., et al. "Extension and validation of a geographic
    information system-based method for calculating the Revised Universal
    Soil Loss Equation length-slope factor for erosion risk assessments in
    large watersheds." Journal of Soil and Water Conservation 63.3 (2008):
    105-111.
"""
import logging
import os

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from .. import gettext
from .. import spec
from .. import utils
from .. import validation
from ..unit_registry import u
from . import sdr_core

LOGGER = logging.getLogger(__name__)

# Target nodata is for general rasters that are positive, and _IC_NODATA are
# for rasters that are any range
_TARGET_NODATA = -1.0
_BYTE_NODATA = 255
_IC_NODATA = float(numpy.finfo('float32').min)


def preprocess(args, f_reg):
    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])
    # Test to see if c or p values are outside of 0..1
    for key in ['usle_c', 'usle_p']:
        for lulc_code, row in biophysical_df.iterrows():
            if row[key] < 0 or row[key] > 1:
                raise ValueError(
                    f'A value in the biophysical table is not a number '
                    f'within range 0..1. The offending value is in '
                    f'column "{key}", lucode row "{lulc_code}", '
                    f'and has value "{row[key]}"')

    input_raster_key_list = ['dem', 'lulc', 'erosivity', 'erodibility']
    base_list = [args[f"{file_key}_path"] for file_key in input_raster_key_list]
    aligned_list = [f_reg[f"aligned_{file_key}_path"] for file_key in input_raster_key_list]

    # all continuous rasters can use bilinear, but lulc should be mode
    interpolation_list = ['bilinear', 'mode', 'bilinear', 'bilinear']

    if args['flow_dir_algorithm'] == 'MFD':
        d_dn_func = pygeoprocessing.routing.distance_to_channel_mfd
        flow_dir_task = 'mfd_flow_dir_task'
        flow_accumulation_task = 'mfd_flow_accumulation_task'
        stream_task = 'mfd_stream_task'
        d_dn_task = 'mfd_d_dn_task'
    else:
        d_dn_func = pygeoprocessing.routing.distance_to_channel_d8
        flow_dir_task = 'd8_flow_dir_task'
        flow_accumulation_task = 'd8_flow_accumulation_task'
        stream_task = 'd8_stream_task'
        d_dn_task = 'd8_d_dn_task'

    if 'drainage_path' in args and args['drainage_path'] != '':
        input_raster_key_list.append('drainage')
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])
        interpolation_list.append('near')
        drainage_raster_path = f_reg['stream_and_drainage_path']
        drainage_task = 'drainage_task'
    else:
        drainage_raster_path = f_reg['stream_path']
        drainage_task = stream_task

    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
    min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))

    return dict(
        biophysical_df=biophysical_df,
        input_raster_key_list=input_raster_key_list,
        base_list=base_list,
        aligned_list=aligned_list,
        interpolation_list=interpolation_list,
        drainage_raster_path=drainage_raster_path,
        drainage_task=drainage_task,
        target_pixel_size=(min_pixel_size, -min_pixel_size),
        target_sr_wkt=dem_raster_info['projection_wkt'],
        lulc_to_c=biophysical_df['usle_c'].to_dict(),
        lulc_to_cp=(biophysical_df['usle_c'] * biophysical_df['usle_p']).to_dict(),
        d_dn_func=d_dn_func,
        flow_dir_task=flow_dir_task,
        flow_accumulation_task=flow_accumulation_task,
        stream_task=stream_task,
        d_dn_task=d_dn_task,
        threshold_flow_accumulation=float(args['threshold_flow_accumulation']),
        l_max=float(args['l_max']),
        k_param=float(args['k_param']),
        ic_0_param=float(args['ic_0_param']),
        sdr_max=float(args['sdr_max']))



# raster_map op for building a mask where all pixels in the stack are valid.
def _create_mutual_mask_op(*arrays): return 1


# raster_map op for using a mask raster to mask out another raster.
def _mask_single_raster_op(source_array, mask_array): return source_array


def _avoided_export_op(avoided_erosion, sdr, sed_deposition):
    """raster_map equation: calculate total retention.

    Args:
        avoided_erosion (numpy.array): Avoided erosion values.
        sdr (numpy.array): SDR values.
        sed_deposition (numpy.array): Sediment deposition values.

    Returns:
        A ``numpy.array`` of computed total retention matching the shape of
        the input numpy arrays.
    """
    # avoided_erosion represents RLKS - RKLSCP (where RKLSCP is also
    # known as a modified USLE)
    return avoided_erosion * sdr + sed_deposition


def add_drainage_op(stream, drainage):
    """raster_map equation: add drainage mask to stream layer.

    Args:
        stream (numpy.array): binary array where 1 indicates
            a stream, and 0 is a valid landscape pixel but not a stream.
        drainage (numpy.array): binary array where 1 indicates any water
            reaching that pixel drains to a stream.

    Returns:
        numpy.array combination of stream and drainage
    """
    return numpy.where(drainage == 1, 1, stream)


# raster_map equation: calculate USLE
def usle_op(rkls, cp_factor): return rkls * cp_factor


# raster_map equation: calculate the inverse ws factor
def inverse_ws_op(w_factor, s_factor): return 1 / (w_factor * s_factor)


def _calculate_what_drains_to_stream(
        flow_dir_path, dist_to_channel_path, target_mask_path):
    """Create a mask indicating regions that do or do not drain to a stream.

    This is useful because the distance-to-stream functions may leave
    some unexpected regions as nodata if they do not drain to a stream.  This
    may be confusing behavior, so this mask is intended to locate what drains
    to a stream and what does not. A pixel doesn't drain to a stream if it has
    a defined flow direction but undefined distance to stream.

    Args:
        flow_dir_path (string): The path to a flow direction raster
            (MFD or D8). This raster must have a nodata value defined.
        dist_to_channel_path (string): The path to a distance-to-channel
            raster.  This raster must have a nodata value defined.
        target_mask_path (string): The path to where the mask raster should be
            written.

    Returns:
        ``None``
    """
    flow_dir_nodata = pygeoprocessing.get_raster_info(
        flow_dir_path)['nodata'][0]
    dist_to_channel_nodata = pygeoprocessing.get_raster_info(
        dist_to_channel_path)['nodata'][0]

    def _what_drains_to_stream(flow_dir, dist_to_channel):
        """Determine which pixels do and do not drain to a stream.

        Args:
            flow_dir (numpy.array): A numpy array of flow direction values.
            dist_to_channel (numpy.array): A numpy array of calculated
                distances to the nearest channel.

        Returns:
            A ``numpy.array`` of dtype ``numpy.uint8`` with pixels where:

                * ``255`` where ``flow_dir`` is nodata (and thus
                  ``dist_to_channel`` is also nodata).
                * ``0`` where ``flow_dir`` has data and ``dist_to_channel``
                  does not
                * ``1`` where ``flow_dir`` has data, and
                  ``dist_to_channel`` also has data.
        """
        drains_to_stream = numpy.full(
            flow_dir.shape, _BYTE_NODATA, dtype=numpy.uint8)
        valid_flow_dir = ~pygeoprocessing.array_equals_nodata(
            flow_dir, flow_dir_nodata)
        valid_dist_to_channel = (
            ~pygeoprocessing.array_equals_nodata(
                dist_to_channel, dist_to_channel_nodata) &
            valid_flow_dir)

        # Nodata where both flow_dir and dist_to_channel are nodata
        # 1 where flow_dir and dist_to_channel have values (drains to stream)
        # 0 where flow_dir has data and dist_to_channel doesn't (doesn't drain)
        drains_to_stream[valid_flow_dir & valid_dist_to_channel] = 1
        drains_to_stream[valid_flow_dir & ~valid_dist_to_channel] = 0
        return drains_to_stream

    pygeoprocessing.raster_calculator(
        [(flow_dir_path, 1), (dist_to_channel_path, 1)],
        _what_drains_to_stream, target_mask_path, gdal.GDT_Byte, _BYTE_NODATA)


def _calculate_ls_factor(
        flow_accumulation_path, slope_path, l_max,
        target_ls_factor_path):
    """Calculate LS factor.

    Calculates the LS factor using Equation 3 from "Extension and
    validation of a geographic information system-based method for calculating
    the Revised Universal Soil Loss Equation length-slope factor for erosion
    risk assessments in large watersheds".

    The equation for this is::

                 (upstream_area + pixel_area)^(m+1) - upstream_area^(m+1)
        LS = S * --------------------------------------------------------
                       (pixel_area^(m+2)) * aspect_dir * 22.13^(m)

    Where

        * ``S`` is the slope factor defined in equation 4 from the same paper,
          calculated by the following where ``b`` is the slope in radians:

          * ``S = 10.8 * sin(b) + 0.03`` where slope < 9%
          * ``S = 16.8 * sin(b) - 0.50`` where slope >= 9%

        * ``upstream_area`` is interpreted as the square root of the
          catchment area, to match SAGA-GIS's method for calculating LS
          Factor.
        * ``pixel_area`` is the area of the pixel in square meters.
        * ``m`` is the slope-length exponent of the RUSLE LS-factor,
          which, as discussed in Oliveira et al. 2013 is a function of the
          on-pixel slope theta:

          * ``m = 0.2`` when ``theta <= 1%``
          * ``m = 0.3`` when ``1% < theta <= 3.5%``
          * ``m = 0.4`` when ``3.5% < theta <= 5%``
          * ``m = 0.5`` when ``5% < theta <= 9%``
          * ``m = (beta / (1+beta)`` when ``theta > 9%``, where
            ``beta = (sin(theta) / 0.0896) / (3*sin(theta)^0.8 + 0.56)``

        * ``aspect_dir`` is calculated by ``|sin(alpha)| + |cos(alpha)|``
          for the given pixel.

    Oliveira et al can be found at:

        Oliveira, A.H., Silva, M.A. da, Silva, M.L.N., Curi, N., Neto, G.K.,
        Freitas, D.A.F. de, 2013. Development of Topographic Factor Modeling
        for Application in Soil Erosion Models, in: Intechopen (Ed.), Soil
        Processes and Current Trends in Quality Assessment. p. 28.

    Args:
        flow_accumulation_path (string): path to raster, pixel values are the
            contributing upslope area at that cell. Pixel size is square.
        slope_path (string): path to slope raster as a percent
        l_max (float): if the calculated value of L exceeds this value
            it is clamped to this value.
        target_ls_factor_path (string): path to output ls_prime_factor
            raster

    Returns:
        None

    """
    cell_size = abs(pygeoprocessing.get_raster_info(
        flow_accumulation_path)['pixel_size'][0])
    cell_area = cell_size ** 2

    def ls_factor_function(percent_slope, flow_accumulation):
        """Calculate the LS factor.

        Args:
            percent_slope (numpy.ndarray): slope in percent
            flow_accumulation (numpy.ndarray): upslope pixels
            l_max (float): max L factor, clamp to this value if L exceeds it

        Returns:
            ls_factor

        """
        # Although Desmet & Govers (1996) discusses "upstream contributing
        # area", this is not strictly defined. We decided to use the square
        # root of the upstream contributing area here as an estimate, which
        # matches the SAGA LS Factor option "square root of catchment area".
        # See the InVEST ADR-0001 for more information.
        # We subtract 1 from the flow accumulation because FA includes itself
        # in its count of pixels upstream and our LS factor equation wants only
        # those pixels that are strictly upstream.
        contributing_area = numpy.sqrt((flow_accumulation - 1) * cell_area)
        slope_in_radians = numpy.arctan(percent_slope / 100)

        aspect_length = (numpy.fabs(numpy.sin(slope_in_radians)) +
                         numpy.fabs(numpy.cos(slope_in_radians)))

        # From Equation 4 in "Extension and validation of a geographic
        # information system ..."
        slope_factor = numpy.where(
            percent_slope < 9,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        beta = (
            (numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        # Set m value via lookup table: Table 1 in
        # InVEST Sediment Model_modifications_10-01-2012_RS.docx
        # note slope_table in percent
        slope_table = numpy.array([1., 3.5, 5., 9.])
        m_table = numpy.array([0.2, 0.3, 0.4, 0.5])
        # mask where slopes are larger than lookup table
        big_slope_mask = percent_slope > slope_table[-1]
        m_indexes = numpy.digitize(
            percent_slope[~big_slope_mask], slope_table, right=True)
        m_exp = numpy.empty(big_slope_mask.shape, dtype=numpy.float32)
        m_exp[big_slope_mask] = (
            beta[big_slope_mask] / (1 + beta[big_slope_mask]))
        m_exp[~big_slope_mask] = m_table[m_indexes]

        l_factor = (
            ((contributing_area + cell_area)**(m_exp+1) -
             contributing_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (aspect_length**m_exp) *
             (22.13**m_exp)))

        # threshold L factor to l_max
        l_factor[l_factor > l_max] = l_max

        return l_factor * slope_factor

    pygeoprocessing.raster_map(
        op=ls_factor_function,
        rasters=[slope_path, flow_accumulation_path],
        target_path=target_ls_factor_path)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate potential soil loss (tons / (ha * year)) using RKLS.

    (revised universal soil loss equation with no C or P).

    Args:
        ls_factor_path (string): path to LS raster that has square pixels in
            meter units.
        erosivity_path (string): path to erosivity raster
            (MJ * mm / (ha * hr * yr))
        erodibility_path (string): path to erodibility raster
            (t * ha * hr / (MJ * ha * mm))
        stream_path (string): path to drainage raster
            (1 is drainage, 0 is not)
        rkls_path (string): path to RKLS raster

    Returns:
        None

    """
    erosivity_nodata = pygeoprocessing.get_raster_info(
        erosivity_path)['nodata'][0]
    erodibility_nodata = pygeoprocessing.get_raster_info(
        erodibility_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(
        stream_path)['nodata'][0]

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculate the RKLS equation.

        Args:
            ls_factor (numpy.ndarray): length/slope factor. unitless.
            erosivity (numpy.ndarray): related to peak rainfall events. units:
                MJ * mm / (ha * hr * yr)
            erodibility (numpy.ndarray): related to the potential for soil to
                erode. units: t * ha * hr / (MJ * ha * mm)
            stream (numpy.ndarray): stream mask (1 stream, 0 no stream)

        Returns:
            numpy.ndarray of RKLS values in tons / (ha * year))
        """
        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            ~pygeoprocessing.array_equals_nodata(ls_factor, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(stream, stream_nodata))
        if erosivity_nodata is not None:
            nodata_mask &= ~pygeoprocessing.array_equals_nodata(
                erosivity, erosivity_nodata)
        if erodibility_nodata is not None:
            nodata_mask &= ~pygeoprocessing.array_equals_nodata(
                erodibility, erodibility_nodata)

        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = _TARGET_NODATA

        rkls[valid_mask] = (           # rkls units are tons / (ha * year)
            ls_factor[valid_mask] *    # unitless
            erosivity[valid_mask] *    # MJ * mm / (ha * hr * yr)
            erodibility[valid_mask])   # t * ha * hr / (MJ * ha * mm)
        return rkls

    # aligning with index 3 that's the stream and the most likely to be
    # aligned with LULCs
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            ls_factor_path, erosivity_path, erodibility_path, stream_path]],
        rkls_function, rkls_path, gdal.GDT_Float32, _TARGET_NODATA)


def threshold_slope_op(slope):
    """raster_map equation: convert slope to m/m and clamp at 0.005 and 1.0.

    As desribed in Cavalli et al., 2013.
    """
    slope_m = slope / 100
    slope_m[slope_m < 0.005] = 0.005
    slope_m[slope_m > 1] = 1
    return slope_m


def _calculate_w(
        lulc_to_c, lulc_path, w_factor_path,
        out_thresholded_w_factor_path):
    """W factor: map C values from LULC and lower threshold to 0.001.

    W is a factor in calculating d_up accumulation for SDR.

    Args:
        lulc_to_c (dict): mapping of LULC codes to C values
        lulc_path (string): path to LULC raster
        w_factor_path (string): path to outputed raw W factor
        out_thresholded_w_factor_path (string): W factor from `w_factor_path`
            thresholded to be no less than 0.001.

    Returns:
        None

    """
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_c:
            lulc_to_c = lulc_to_c.copy()
            lulc_to_c[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_c, w_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)

    pygeoprocessing.raster_map(
        op=lambda w_val: numpy.where(w_val < 0.001, 0.001, w_val),
        rasters=[w_factor_path],
        target_path=out_thresholded_w_factor_path)


def _calculate_cp(lulc_to_cp, lulc_path, cp_factor_path):
    """Map LULC to C*P value.

    Args:
        lulc_to_cp (dict): mapping of lulc codes to CP values
        lulc_path (string): path to LULC raster
        cp_factor_path (string): path to output raster of LULC mapped to C*P
            values

    Returns:
        None

    """
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_cp:
            lulc_to_cp[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_cp, cp_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)


def _calculate_bar_factor(
        flow_direction_path, factor_path, flow_accumulation_path,
        accumulation_path, out_bar_path, flow_dir_algorithm):
    """Route user defined source across DEM.

    Used for calculating S and W bar in the SDR operation.

    Args:
        dem_path (string): path to DEM raster
        factor_path (string): path to arbitrary factor raster
        flow_accumulation_path (string): path to flow accumulation raster
        flow_direction_path (string): path to flow direction path (in radians)
        accumulation_path (string): path to a raster that can be used to
            save the accumulation of the factor.  Temporary file.
        out_bar_path (string): path to output raster that is the result of
            the factor accumulation raster divided by the flow accumulation
            raster.
        flow_dir_algorithm (string): flow direction algorithm, 'D8' or 'MFD'

    Returns:
        None.

    """
    LOGGER.debug(f"doing flow accumulation on {factor_path}")

    if flow_dir_algorithm == 'D8':
        flow_accum_func = pygeoprocessing.routing.flow_accumulation_d8
    else:  # MFD
        flow_accum_func = pygeoprocessing.routing.flow_accumulation_mfd
    # manually setting compression to DEFLATE because we got some LZW
    # errors when testing with large data.
    flow_accum_func(
        (flow_direction_path, 1), accumulation_path,
        weight_raster_path_band=(factor_path, 1),
        raster_driver_creation_tuple=('GTIFF', [
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=3']))

    pygeoprocessing.raster_map(
        op=lambda base_accum, flow_accum: base_accum / flow_accum,
        rasters=[accumulation_path, flow_accumulation_path],
        target_path=out_bar_path)


def _calculate_d_up(
        w_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate w_bar * s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        pygeoprocessing.get_raster_info(w_bar_path)['pixel_size'][0])**2
    pygeoprocessing.raster_map(
        op=lambda w_bar, s_bar, flow_accum: (
            w_bar * s_bar * numpy.sqrt(flow_accum * cell_area)),
        rasters=[w_bar_path, s_bar_path, flow_accumulation_path],
        target_path=out_d_up_path)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    """Calculate log10(d_up/d_dn)."""
    # ic can be positive or negative, so float.min is a reasonable nodata value
    d_dn_nodata = pygeoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(d_up, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(d_dn, d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        ic_array[:] = _IC_NODATA
        ic_array[valid_mask] = numpy.log10(
            d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    pygeoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, _IC_NODATA)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    """Derive SDR from k, ic0, ic; 0 on the stream and clamped to sdr_max."""
    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(ic_factor, _IC_NODATA) & (stream != 1))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            sdr_max / (1+numpy.exp((ic_0-ic_factor[valid_mask])/k_factor)))
        result[stream == 1] = 0.0
        return result

    pygeoprocessing.raster_calculator(
        [(ic_path, 1), (stream_path, 1)], sdr_op, out_sdr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_e_prime(usle_path, sdr_path, stream_path, target_e_prime):
    """Calculate USLE * (1-SDR)."""
    def e_prime_op(usle, sdr, streams):
        """Wash that does not reach stream."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(usle, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(sdr, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * (1-sdr[valid_mask])
        # set to 0 on streams, to prevent nodata propagating up/down slope
        # in calculate_sediment_deposition. This makes sense intuitively:
        # E'_i represents the sediment export from pixel i that does not
        # reach a stream, which is 0 if pixel i is already in a stream.
        result[streams == 1] = 0
        return result

    pygeoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1), (stream_path, 1)], e_prime_op,
        target_e_prime, gdal.GDT_Float32, _TARGET_NODATA)


def _generate_report(
        watersheds_path, usle_path, sed_export_path,
        sed_deposition_path, avoided_export_path, avoided_erosion_path,
        watershed_results_sdr_path):
    """Create summary vector with totals for rasters.

    Args:
        watersheds_path (string): The path to the watersheds vector.
        usle_path (string): The path to the computed USLE raster.
        sed_export_path (string): The path to the sediment export raster.
        sed_deposition_path (string): The path to the sediment deposition
            raster.
        avoided_export_path (string): The path to the total retention raster.
        avoided_erosion_path (string): The path to the avoided local
            erosion raster.
        watershed_results_sdr_path (string): The path to where the watersheds
            vector will be created.  This path must end in ``.shp`` as it will
            be written as an ESRI Shapefile.

    Returns:
        ``None``
    """
    original_datasource = gdal.OpenEx(watersheds_path, gdal.OF_VECTOR)
    if os.path.exists(watershed_results_sdr_path):
        LOGGER.warning(f'overwriting results at {watershed_results_sdr_path}')
        os.remove(watershed_results_sdr_path)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(
        watershed_results_sdr_path, original_datasource)

    target_layer = target_vector.GetLayer()
    target_layer.SyncToDisk()

    # It's worth it to check if the geometries don't significantly overlap.
    # On large rasters, this can save a TON of time rasterizing even a
    # relatively simple vector.
    geometries_might_overlap = utils.geometries_overlap(
        watershed_results_sdr_path)
    fields_and_rasters = [
        ('usle_tot', usle_path), ('sed_export', sed_export_path),
        ('sed_dep', sed_deposition_path), ('avoid_exp', avoided_export_path),
        ('avoid_eros', avoided_erosion_path)]

    # Using the list option for raster path bands so that we can reduce
    # rasterizations, which are costly on large datasets.
    zonal_stats_results = pygeoprocessing.zonal_statistics(
        [(raster_path, 1) for (_, raster_path) in fields_and_rasters],
        watershed_results_sdr_path,
        polygons_might_overlap=geometries_might_overlap)

    field_summaries = {
        field: stats for ((field, _), stats) in
        zip(fields_and_rasters, zonal_stats_results)}

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_layer.CreateField(field_def)

    # Since pixel values are t/(ha•yr), raster sum is (t•px)/(ha•yr).
    # To convert to t/yr, multiply by ha/px.
    raster_info = pygeoprocessing.get_raster_info(usle_path)
    pixel_area = abs(numpy.prod(raster_info['pixel_size']))
    ha_per_px = pixel_area / 10000

    target_layer.ResetReading()
    for feature in target_layer:
        feature_id = feature.GetFID()
        for field_name in field_summaries:
            feature.SetField(
                field_name,
                float(field_summaries[field_name][feature_id]['sum']
                      * ha_per_px))
        target_layer.SetFeature(feature)
    target_vector = None
    target_layer = None


MODEL_SPEC = spec.ModelSpec(
    model_id="sdr",
    model_title=gettext("Sediment Delivery Ratio"),
    userguide="sdr.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=(),
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["dem_path", "erosivity_path", "erodibility_path"],
        ["lulc_path", "biophysical_table_path"],
        ["watersheds_path", "drainage_path"],
        ["flow_dir_algorithm", "threshold_flow_accumulation", "k_param",
         "sdr_max", "ic_0_param", "l_max"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.PROJECTED_DEM,
        spec.SingleBandRasterInput(
            id="erosivity_path",
            name=gettext("erosivity"),
            about=gettext(
                "Map of rainfall erosivity, reflecting the intensity and duration of"
                " rainfall in the area of interest."
            ),
            data_type=float,
            units=u.megajoule * u.millimeter / (u.hectare * u.hour * u.year),
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="erodibility_path",
            name=gettext("soil erodibility"),
            about=gettext(
                "Map of soil erodibility, the susceptibility of soil particles to"
                " detachment and transport by rainfall and runoff."
            ),
            data_type=float,
            units=u.metric_ton * u.hectare * u.hour/ (u.hectare * u.megajoule * u.millimeter),
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="lulc_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code. All values in this raster must have"
                " corresponding entries in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.VectorInput(
            id="watersheds_path",
            name=gettext("Watersheds"),
            about=gettext(
                "Map of the boundaries of the watershed(s) over which to aggregate"
                " results. Each watershed should contribute to a point of interest where"
                " water quality will be analyzed."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
            projected=True
        ),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC code to biophysical properties of that LULC"
                " class. All values in the LULC raster must have corresponding entries in"
                " this table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.RatioInput(
                    id="usle_c",
                    about=gettext("Cover-management factor for the USLE"),
                    units=None
                ),
                spec.RatioInput(
                    id="usle_p",
                    about=gettext("Support practice factor for the USLE"),
                    units=None
                )
            ],
            index_col="lucode"
        ),
        spec.THRESHOLD_FLOW_ACCUMULATION,
        spec.NumberInput(
            id="k_param",
            name=gettext("Borselli k parameter"),
            about=gettext("Borselli k parameter."),
            units=u.none
        ),
        spec.RatioInput(
            id="sdr_max",
            name=gettext("maximum SDR value"),
            about=gettext("The maximum SDR value that a pixel can have."),
            units=None
        ),
        spec.NumberInput(
            id="ic_0_param",
            name=gettext("Borselli IC0 parameter"),
            about=gettext("Borselli IC0 parameter."),
            units=u.none
        ),
        spec.NumberInput(
            id="l_max",
            name=gettext("maximum l value"),
            about=gettext(
                "The maximum allowed value of the slope length parameter (L) in the LS"
                " factor."
            ),
            units=u.none,
            expression="value > 0"
        ),
        spec.SingleBandRasterInput(
            id="drainage_path",
            name=gettext("drainages"),
            about=gettext(
                "Map of locations of artificial drainages that drain to the watershed."
                " Pixels with 1 are drainages and are treated like streams. Pixels with 0"
                " are not drainages."
            ),
            required=False,
            data_type=int,
            units=None,
            projected=None
        ),
        spec.FLOW_DIR_ALGORITHM
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="avoided_erosion.tif",
            about=gettext(
                "The contribution of vegetation to keeping soil from eroding from each"
                " pixel. (Eq. (82))"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="avoided_export.tif",
            about=gettext(
                "The contribution of vegetation to keeping erosion from entering a"
                " stream. This combines local/on-pixel sediment retention with trapping"
                " of erosion from upslope of the pixel. (Eq. (83))"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="rkls.tif",
            about=gettext(
                "Total potential soil loss per pixel in the original land cover from the"
                " RKLS equation. Equivalent to the soil loss for bare soil. (Eq. (68),"
                " without applying the C or P factors)."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="sed_deposition.tif",
            about=gettext(
                "The total amount of sediment deposited on the pixel from upslope sources"
                " as a result of trapping. (Eq. (80))"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="sed_export.tif",
            about=gettext(
                "The total amount of sediment exported from each pixel that reaches the"
                " stream. (Eq. (76))"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.STREAM.model_copy(update=dict(id="stream.tif")),
        spec.SingleBandRasterOutput(
            id="stream_and_drainage.tif",
            about=gettext(
                "This raster is the union of that layer with the calculated stream"
                " layer(Eq. (85)). Values of 1 represent streams, values of 0 are"
                " non-stream pixels."
            ),
            created_if="drainage_path",
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="usle.tif",
            about=gettext(
                "Total potential soil loss per hectare in the original land cover"
                " calculated from the USLE equation. (Eq. (68))"
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.VectorOutput(
            id="watershed_results_sdr.shp",
            about=gettext("Table containing biophysical values for each watershed"),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="sed_export",
                    about=gettext(
                        "Total amount of sediment exported to the stream per watershed."
                        " (Eq. (77) with sum calculated over the watershed area)"
                    ),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="usle_tot",
                    about=gettext(
                        "Total amount of potential soil loss in each watershed calculated"
                        " by the USLE equation. (Sum of USLE from (68) over the watershed"
                        " area)"
                    ),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="avoid_exp",
                    about=gettext("The sum of avoided export in the watershed."),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="avoid_eros",
                    about=gettext("The sum of avoided local erosion in the watershed"),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="sed_dep",
                    about=gettext(
                        "Total amount of sediment deposited on the landscape in each"
                        " watershed, which does not enter the stream."
                    ),
                    units=u.metric_ton
                )
            ]
        ),
        spec.DirectoryOutput(
            id="intermediate_outputs",
            about=None,
            contents=[
                spec.SingleBandRasterOutput(
                    id="cp.tif",
                    about=gettext(
                        "CP factor derived by mapping usle_c and usle_p from the"
                        " biophysical table to the LULC raster."
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="d_dn.tif",
                    about=gettext(
                        "Downslope factor of the index of connectivity (Eq. (74))"
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="d_up.tif",
                    about=gettext(
                        "Upslope factor of the index of connectivity (Eq. (73))"
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="e_prime.tif",
                    about=gettext(
                        "Sediment downslope deposition, the amount of sediment from a"
                        " given pixel that does not reach a stream (Eq. (78))"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare / u.year
                ),
                spec.SingleBandRasterOutput(
                    id="f.tif",
                    about=gettext(
                        "Map of sediment flux for sediment that does not reach the stream"
                        " (Eq. (81))"
                    ),
                    data_type=float,
                    units=u.metric_ton / u.hectare / u.year
                ),
                spec.FLOW_ACCUMULATION.model_copy(update=dict(id="flow_accumulation.tif")),
                spec.FLOW_DIRECTION.model_copy(update=dict(id="flow_direction.tif")),
                spec.SingleBandRasterOutput(
                    id="ic.tif",
                    about=gettext("Index of connectivity (Eq. (70))"),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="ls.tif",
                    about=gettext("LS factor for USLE (Eq. (69))"),
                    data_type=float,
                    units=u.none
                ),
                spec.FILLED_DEM.model_copy(update=dict(id="pit_filled_dem.tif")),
                spec.SingleBandRasterOutput(
                    id="s_accumulation.tif",
                    about=gettext(
                        "Flow accumulation weighted by the thresholded slope. Used in"
                        " calculating s_bar."
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="s_bar.tif",
                    about=gettext(
                        "Mean thresholded slope gradient of the upslope contributing area"
                        " (in eq. (73))"
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="sdr_factor.tif",
                    about=gettext("Sediment delivery ratio (Eq. (75))"),
                    data_type=float,
                    units=None
                ),
                spec.SLOPE,
                spec.SingleBandRasterOutput(
                    id="slope_threshold.tif",
                    about=gettext(
                        "Percent slope, thresholded to be no less than 0.005 and no"
                        " greater than 1 (eq. (71)). 1 is equivalent to a 45 degree"
                        " slope."
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="w_accumulation.tif",
                    about=gettext(
                        "Flow accumulation weighted by the thresholded cover-management"
                        " factor. Used in calculating w_bar."
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="w_bar.tif",
                    about=gettext(
                        "Mean thresholded cover-management factor for upslope"
                        " contributing area (in eq. (73))"
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="w.tif",
                    about=gettext(
                        "Cover-management factor derived by mapping usle_c from the"
                        " biophysical table to the LULC raster."
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="w_threshold.tif",
                    about=gettext(
                        "Cover-management factor thresholded to be no less than 0.001"
                        " (eq. (72))"
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="weighted_avg_aspect.tif",
                    about=gettext(
                        "Average aspect weighted by flow direction (in eq. (69))"
                    ),
                    data_type=float,
                    units=u.none
                ),
                spec.SingleBandRasterOutput(
                    id="what_drains_to_stream.tif",
                    about=gettext(
                        "Map of which pixels drain to a stream. A value of 1 means that"
                        " at least some of the runoff from that pixel drains to a stream"
                        " in stream.tif. A value of 0 means that it does not drain at all"
                        " to any stream in stream.tif."
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="ws_inverse.tif",
                    about=gettext(
                        "Inverse of the thresholded cover-management factor times the"
                        " thresholded slope (in eq. (74))"
                    ),
                    data_type=float,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="aligned_dem.tif",
                    about=gettext(
                        "Copy of the input DEM, clipped to the extent of the other raster"
                        " inputs."
                    ),
                    data_type=float,
                    units=u.meter
                ),
                spec.SingleBandRasterOutput(
                    id="aligned_drainage.tif",
                    about=gettext(
                        "Copy of the input drainage map, clipped to the extent of the"
                        " other raster inputs and aligned to the DEM."
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="aligned_erodibility.tif",
                    about=gettext(
                        "Copy of the input erodibility map, clipped to the extent of the"
                        " other raster inputs and aligned to the DEM."
                    ),
                    data_type=float,
                    units=u.metric_ton * u.hectare * u.hour / (u.hectare * u.megajoule * u.millimeter)
                ),
                spec.SingleBandRasterOutput(
                    id="aligned_erosivity.tif",
                    about=gettext(
                        "Copy of the input erosivity map, clipped to the extent of the"
                        " other raster inputs and aligned to the DEM."
                    ),
                    data_type=float,
                    units=u.megajoule * u.millimeter / (u.hectare * u.hour * u.year)
                ),
                spec.SingleBandRasterOutput(
                    id="aligned_lulc.tif",
                    about=gettext(
                        "Copy of the input Land Use Land Cover map, clipped to the extent"
                        " of the other raster inputs and aligned to the DEM."
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="mask.tif",
                    about=gettext(
                        "A raster aligned to the DEM and clipped to the extent of the"
                        " other raster inputs. Pixel values indicate where a nodata value"
                        " exists in the stack of aligned rasters (pixel value of 0), or"
                        " if all values in the stack of rasters at this pixel location"
                        " are valid."
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="masked_dem.tif",
                    about=gettext(
                        "A copy of the aligned DEM, masked using the mask raster."
                    ),
                    data_type=float,
                    units=u.meter
                ),
                spec.SingleBandRasterOutput(
                    id="masked_drainage.tif",
                    about=gettext(
                        "A copy of the aligned drainage map, masked using the mask"
                        " raster."
                    ),
                    data_type=int,
                    units=None
                ),
                spec.SingleBandRasterOutput(
                    id="masked_erodibility.tif",
                    about=gettext(
                        "A copy of the aligned erodibility map, masked using the mask"
                        " raster."
                    ),
                    data_type=float,
                    units=u.metric_ton * u.hectare * u.hour / (u.hectare * u.megajoule * u.millimeter)
                ),
                spec.SingleBandRasterOutput(
                    id="masked_erosivity.tif",
                    about=gettext(
                        "A copy of the aligned erosivity map, masked using the mask"
                        " raster."
                    ),
                    data_type=float,
                    units=u.megajoule * u.millimeter / (u.hectare * u.hour * u.year)
                ),
                spec.SingleBandRasterOutput(
                    id="masked_lulc.tif",
                    about=gettext(
                        "A copy of the aligned Land Use Land Cover map, masked using the"
                        " mask raster."
                    ),
                    data_type=int,
                    units=None
                )
            ]
        ),
        spec.TASKGRAPH_DIR
    ],
    preprocessing_function=preprocess,
    tasks=[
        spec.Task(
            key='align_task',
            func=pygeoprocessing.align_and_resize_raster_stack,
            kwarg_keys=dict(
                base_raster_path_list='vals.base_list',
                target_raster_path_list='vals.aligned_list',
                resample_method_list='vals.interpolation_list',
                target_pixel_size='vals.target_pixel_size',
                bounding_box_mode='intersection',
                target_projection_wkt='vals.target_sr_wkt',
                base_vector_path_list=('args.watersheds_path',),
                raster_align_index=0,
                vector_mask_options=dict(mask_vector_path='args.watersheds_path')
            ),
            target_path_list='vals.aligned_list',
            task_name='align input rasters'
        ),
        spec.Task(
            key='mutual_mask_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_create_mutual_mask_op,
                rasters='vals.aligned_list',
                target_path='paths.mask_path',
                target_nodata=0),
            target_path_list=['paths.mask_path'],
            dependent_task_list=['align_task'],
            task_name='create mask'
        ),
        spec.Task(
            key='mask_dem',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_mask_single_raster_op,
                rasters=['paths.aligned_dem_path', 'paths.mask_path'],
                target_path='paths.masked_dem_path'),
            target_path_list=['paths.masked_dem_path'],
            dependent_task_list=['mutual_mask_task', 'align_task'],
            task_name=f'mask dem'
        ),
        spec.Task(
            key='mask_lulc',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_mask_single_raster_op,
                rasters=['paths.aligned_lulc_path', 'paths.mask_path'],
                target_path='paths.masked_lulc_path'),
            target_path_list=['paths.masked_lulc_path'],
            dependent_task_list=['mutual_mask_task', 'align_task'],
            task_name=f'mask lulc'
        ),
        spec.Task(
            key='mask_erosivity',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_mask_single_raster_op,
                rasters=['paths.aligned_erosivity_path', 'paths.mask_path'],
                target_path='paths.masked_erosivity_path'),
            target_path_list=['paths.masked_erosivity_path'],
            dependent_task_list=['mutual_mask_task', 'align_task'],
            task_name=f'mask erosivity'
        ),
        spec.Task(
            key='mask_erodibility',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_mask_single_raster_op,
                rasters=['paths.aligned_erodibility_path', 'paths.mask_path'],
                target_path='paths.masked_erodibility_path'),
            target_path_list=['paths.masked_erodibility_path'],
            dependent_task_list=['mutual_mask_task', 'align_task'],
            task_name=f'mask erodibility'
        ),
        spec.Task(
            key=f"mask_drainage",
            run_if="'drainage_path' in args and args['drainage_path'] != ''",
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_mask_single_raster_op,
                rasters=['paths.aligned_drainage_path', 'paths.mask_path'],
                target_path='paths.masked_drainage_path'),
            target_path_list=['paths.masked_drainage_path'],
            dependent_task_list=['mutual_mask_task', 'align_task'],
            task_name=f'mask drainage'
        ),
        spec.Task(
            key='pit_fill_task',
            func=pygeoprocessing.routing.fill_pits,
            kwarg_keys=dict(
                dem_raster_path_band=('paths.masked_dem_path', 1),
                target_filled_dem_raster_path='paths.pit_filled_dem_path'),
            target_path_list=['paths.pit_filled_dem_path'],
            dependent_task_list=['mask_dem'],
            task_name='fill pits'
        ),
        spec.Task(
            key='slope_task',
            func=pygeoprocessing.calculate_slope,
            kwarg_keys=dict(
                base_elevation_raster_path_band=('paths.pit_filled_dem_path', 1),
                target_slope_path='paths.slope_path'),
            dependent_task_list=['pit_fill_task'],
            target_path_list=['paths.slope_path'],
            task_name='calculate slope'
        ),
        spec.Task(
            key='threshold_slope_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=threshold_slope_op,
                rasters=['paths.slope_path'],
                target_path='paths.thresholded_slope_path'),
            target_path_list=['paths.thresholded_slope_path'],
            dependent_task_list=['slope_task'],
            task_name='threshold slope'
        ),
        spec.Task(
            key='mfd_flow_dir_task',
            run_if="args['flow_dir_algorithm'] == 'MFD'",
            func=pygeoprocessing.routing.flow_dir_mfd,
            kwarg_keys=dict(
                dem_raster_path_band=('paths.pit_filled_dem_path', 1),
                target_flow_dir_path='paths.flow_direction_path'),
            target_path_list=['paths.flow_direction_path'],
            dependent_task_list=['pit_fill_task'],
            task_name='flow direction calculation'
        ),
        spec.Task(
            key='mfd_flow_accumulation_task',
            run_if="args['flow_dir_algorithm'] == 'MFD'",
            func=pygeoprocessing.routing.flow_accumulation_mfd,
            kwarg_keys=dict(
                flow_dir_mfd_raster_path_band=('paths.flow_direction_path', 1),
                target_flow_accum_raster_path='paths.flow_accumulation_path'),
            target_path_list=['paths.flow_accumulation_path'],
            dependent_task_list=['vals.flow_dir_task'],
            task_name='flow accumulation calculation'
        ),
        spec.Task(
            key='mfd_stream_task',
            run_if="args['flow_dir_algorithm'] == 'MFD'",
            func=pygeoprocessing.routing.extract_streams_mfd,
            kwarg_keys=dict(
                flow_accum_raster_path_band=('paths.flow_accumulation_path', 1),
                flow_dir_mfd_path_band=('paths.flow_direction_path', 1),
                flow_threshold='vals.threshold_flow_accumulation',
                target_stream_raster_path='paths.stream_path',
                trace_threshold_proportion=0.7),
            target_path_list=['paths.stream_path'],
            dependent_task_list=['vals.flow_accumulation_task'],
            task_name='extract streams'
        ),
        spec.Task(
            key='d8_flow_dir_task',
            run_if="args['flow_dir_algorithm'] == 'D8'",
            func=pygeoprocessing.routing.flow_dir_d8,
            kwarg_keys=dict(
                dem_raster_path_band=('paths.pit_filled_dem_path', 1),
                target_flow_dir_path='paths.flow_direction_path'),
            target_path_list=['paths.flow_direction_path'],
            dependent_task_list=['pit_fill_task'],
            task_name='flow direction calculation'
        ),
        spec.Task(
            key='d8_flow_accumulation_task',
            run_if="args['flow_dir_algorithm'] == 'D8'",
            func=pygeoprocessing.routing.flow_accumulation_d8,
            kwarg_keys=dict(
                flow_dir_raster_path_band=('paths.flow_direction_path', 1),
                target_flow_accum_raster_path='paths.flow_accumulation_path'),
            target_path_list=['paths.flow_accumulation_path'],
            dependent_task_list=['vals.flow_dir_task'],
            task_name='flow accumulation calculation'
        ),
        spec.Task(
            key='d8_stream_task',
            run_if="args['flow_dir_algorithm'] == 'D8'",
            func=pygeoprocessing.routing.extract_streams_d8,
            kwarg_keys=dict(
                flow_accum_raster_path_band=('paths.flow_accumulation_path', 1),
                flow_threshold='args.threshold_flow_accumulation',
                target_stream_raster_path='paths.stream_path'),
            target_path_list=['paths.stream_path'],
            dependent_task_list=['vals.flow_accumulation_task'],
            task_name='extract streams'
        ),
        spec.Task(
            key='ls_factor_task',
            func=_calculate_ls_factor,
            kwarg_keys=dict(
                flow_accumulation_path='paths.flow_accumulation_path',
                slope_path='paths.slope_path',
                l_max='vals.l_max',
                target_ls_factor_path='paths.ls_path'),
            target_path_list=['paths.ls_path'],
            dependent_task_list=['vals.flow_accumulation_task', 'slope_task'],
            task_name='ls factor calculation'
        ),
        spec.Task(
            key='drainage_task',
            run_if="'drainage_path' in args and args['drainage_path'] != ''",
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=add_drainage_op,
                rasters=['paths.stream_path', 'paths.masked_drainage_path'],
                target_path='paths.stream_and_drainage_path',
                target_dtype=numpy.uint8),
            target_path_list=['paths.stream_and_drainage_path'],
            dependent_task_list=['vals.stream_task', 'mask_drainage'],
            task_name='add drainage'
        ),
        spec.Task(
            key='threshold_w_task',
            func=_calculate_w,
            kwarg_keys=dict(
                lulc_to_c='vals.lulc_to_c',
                lulc_path='paths.masked_lulc_path',
                w_factor_path='paths.w_path',
                out_thresholded_w_factor_path='paths.thresholded_w_path'),
            target_path_list=['paths.w_path', 'paths.thresholded_w_path'],
            dependent_task_list=['mask_lulc'],
            task_name='calculate W'
        ),
        spec.Task(
            key='cp_task',
            func=_calculate_cp,
            kwarg_keys=dict(
                lulc_to_cp='vals.lulc_to_cp',
                lulc_path='paths.masked_lulc_path',
                cp_factor_path='paths.cp_factor_path'),
            target_path_list=['paths.cp_factor_path'],
            dependent_task_list=['mask_lulc'],
            task_name='calculate CP'
        ),
        spec.Task(
            key='rkls_task',
            func=_calculate_rkls,
            kwarg_keys=dict(
                ls_factor_path='paths.ls_path',
                erosivity_path='paths.masked_erosivity_path',
                erodibility_path='paths.masked_erodibility_path',
                stream_path='vals.drainage_raster_path',
                rkls_path='paths.rkls_path'),
            target_path_list=['paths.rkls_path'],
            dependent_task_list=[
                'mask_erosivity', 'mask_erodibility',
                'vals.drainage_task', 'ls_factor_task'],
            task_name='calculate RKLS'
        ),
        spec.Task(
            key='usle_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=usle_op,
                rasters=['paths.rkls_path', 'paths.cp_factor_path'],
                target_path='paths.usle_path'),
            target_path_list=['paths.usle_path'],
            dependent_task_list=['rkls_task', 'cp_task'],
            task_name='calculate USLE'
        ),
        spec.Task(
            key='w_bar_task',
            func=_calculate_bar_factor,
            kwarg_keys=dict(
                flow_direction_path='paths.flow_direction_path',
                factor_path='paths.thresholded_w_path',
                flow_accumulation_path='paths.flow_accumulation_path',
                accumulation_path='paths.w_accumulation_path',
                out_bar_path='paths.w_bar_path',
                flow_dir_algorithm='args.flow_dir_algorithm'),
            target_path_list=['paths.w_accumulation_path', 'paths.w_bar_path'],
            dependent_task_list=[
                'threshold_w_task', 'vals.flow_accumulation_task', 'vals.flow_dir_task'],
            task_name=f'calculate w_bar'
        ),
        spec.Task(
            key='s_bar_task',
            func=_calculate_bar_factor,
            kwarg_keys=dict(
                flow_direction_path='paths.flow_direction_path',
                factor_path='paths.thresholded_slope_path',
                flow_accumulation_path='paths.flow_accumulation_path',
                accumulation_path='paths.s_accumulation_path',
                out_bar_path='paths.s_bar_path',
                flow_dir_algorithm='args.flow_dir_algorithm'),
            target_path_list=['paths.s_accumulation_path', 'paths.s_bar_path'],
            dependent_task_list=[
                'threshold_slope_task', 'vals.flow_accumulation_task', 'vals.flow_dir_task'],
            task_name=f'calculate s_bar'
        ),
        spec.Task(
            key='d_up_task',
            func=_calculate_d_up,
            kwarg_keys=dict(
                w_bar_path='paths.w_bar_path',
                s_bar_path='paths.s_bar_path',
                flow_accumulation_path='paths.flow_accumulation_path',
                out_d_up_path='paths.d_up_path'),
            target_path_list=['paths.d_up_path'],
            dependent_task_list=['s_bar_task', 'w_bar_task', 'vals.flow_accumulation_task'],
            task_name='calculate Dup'
        ),
        spec.Task(
            key='inverse_ws_factor_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=inverse_ws_op,
                rasters=['paths.thresholded_w_path',
                         'paths.thresholded_slope_path'],
                target_path='paths.ws_inverse_path'),
            target_path_list=['paths.ws_inverse_path'],
            dependent_task_list=['threshold_slope_task', 'threshold_w_task'],
            task_name='calculate inverse ws factor'
        ),

        spec.Task(
            key='d8_d_dn_task',
            func=pygeoprocessing.routing.distance_to_channel_d8,
            run_if="args['flow_dir_algorithm'] == 'D8'",
            kwarg_keys=dict(
                flow_dir_d8_raster_path_band=('paths.flow_direction_path', 1),
                channel_raster_path_band=('vals.drainage_raster_path', 1),
                target_distance_to_channel_raster_path='paths.d_dn_path',
                weight_raster_path_band=('paths.ws_inverse_path', 1)),
            target_path_list=['paths.d_dn_path'],
            dependent_task_list=[
                'vals.flow_dir_task', 'vals.drainage_task',
                'inverse_ws_factor_task'],
            task_name='calculating d_dn'
        ),
        spec.Task(
            key='mfd_d_dn_task',
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            run_if="args['flow_dir_algorithm'] == 'MFD'",
            kwarg_keys=dict(
                flow_dir_mfd_raster_path_band=('paths.flow_direction_path', 1),
                channel_raster_path_band=('vals.drainage_raster_path', 1),
                target_distance_to_channel_raster_path='paths.d_dn_path',
                weight_raster_path_band=('paths.ws_inverse_path', 1)),
            target_path_list=['paths.d_dn_path'],
            dependent_task_list=[
                'vals.flow_dir_task', 'vals.drainage_task',
                'inverse_ws_factor_task'],
            task_name='calculating d_dn'
        ),

        spec.Task(
            key='ic_task',
            func=_calculate_ic,
            kwarg_keys=dict(
                d_up_path='paths.d_up_path',
                d_dn_path='paths.d_dn_path',
                out_ic_factor_path='paths.ic_path'),
            target_path_list=['paths.ic_path'],
            dependent_task_list=['d_up_task', 'vals.d_dn_task'],
            task_name='calculate ic'
        ),
        spec.Task(
            key='sdr_task',
            func=_calculate_sdr,
            kwarg_keys=dict(
                k_factor='vals.k_param',
                ic_0='vals.ic_0_param',
                sdr_max='vals.sdr_max',
                ic_path='paths.ic_path',
                stream_path='vals.drainage_raster_path',
                out_sdr_path='paths.sdr_path'),
            target_path_list=['paths.sdr_path'],
            dependent_task_list=['ic_task'],
            task_name='calculate sdr'
        ),
        spec.Task(
            key='sed_export_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=numpy.multiply,  # export = USLE * SDR
                rasters=['paths.usle_path', 'paths.sdr_path'],
                target_path='paths.sed_export_path'),
            target_path_list=['paths.sed_export_path'],
            dependent_task_list=['usle_task', 'sdr_task'],
            task_name='calculate sed export'
        ),
        spec.Task(
            key='e_prime_task',
            func=_calculate_e_prime,
            kwarg_keys=dict(
                usle_path='paths.usle_path',
                sdr_path='paths.sdr_path',
                stream_path='vals.drainage_raster_path',
                target_e_prime='paths.e_prime_path'),
            target_path_list=['paths.e_prime_path'],
            dependent_task_list=['usle_task', 'sdr_task'],
            task_name='calculate export prime'
        ),
        spec.Task(
            key='sed_deposition_task',
            func=sdr_core.calculate_sediment_deposition,
            kwarg_keys=dict(
                flow_direction_path='paths.flow_direction_path',
                e_prime_path='paths.e_prime_path',
                f_path='paths.f_path',
                sdr_path='paths.sdr_path',
                target_sediment_deposition_path='paths.sed_deposition_path',
                algorithm='args.flow_dir_algorithm'),
            dependent_task_list=['e_prime_task', 'sdr_task', 'vals.flow_dir_task'],
            target_path_list=['paths.sed_deposition_path', 'paths.f_path'],
            task_name='sediment deposition'
        ),
        spec.Task(
            key='avoided_erosion_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=numpy.subtract,  # avoided erosion = rkls - usle
                rasters=['paths.rkls_path', 'paths.usle_path'],
                target_path='paths.avoided_erosion_path'),
            dependent_task_list=['rkls_task', 'usle_task'],
            target_path_list=['paths.avoided_erosion_path'],
            task_name='calculate avoided erosion'
        ),
        spec.Task(
            key='avoided_export_task',
            func=pygeoprocessing.raster_map,
            kwarg_keys=dict(
                op=_avoided_export_op,
                rasters=['paths.avoided_erosion_path',
                         'paths.sdr_path',
                         'paths.sed_deposition_path'],
                target_path='paths.avoided_export_path'),
            dependent_task_list=['avoided_erosion_task', 'sdr_task',
                                 'sed_deposition_task'],
            target_path_list=['paths.avoided_export_path'],
            task_name='calculate total retention'
        ),
        spec.Task(
            key='what_drains_to_stream_task',
            func=_calculate_what_drains_to_stream,
            kwarg_keys=dict(
                flow_dir_path='paths.flow_direction_path',
                dist_to_channel_path='paths.d_dn_path',
                target_mask_path='paths.drainage_mask'),
            target_path_list=['paths.drainage_mask'],
            dependent_task_list=['vals.flow_dir_task', 'vals.d_dn_task'],
            task_name='write mask of what drains to stream'
        ),
        spec.Task(
            key='generate_report_task',
            func=_generate_report,
            kwarg_keys=dict(
                watersheds_path='args.watersheds_path',
                usle_path='paths.usle_path',
                sed_export_path='paths.sed_export_path',
                sed_deposition_path='paths.sed_deposition_path',
                avoided_export_path='paths.avoided_export_path',
                avoided_erosion_path='paths.avoided_erosion_path',
                watershed_results_sdr_path='paths.watershed_results_sdr_path'),
            target_path_list=['paths.watershed_results_sdr_path'],
            dependent_task_list=[
                'usle_task', 'sed_export_task', 'avoided_export_task',
                'sed_deposition_task', 'avoided_erosion_task'],
            task_name='generate report'
        )
    ],
    intermediate_dir_name='intermediate_outputs',
    output_base_files={
        'rkls_path': 'rkls.tif',
        'sed_export_path': 'sed_export.tif',
        'sed_deposition_path': 'sed_deposition.tif',
        'stream_and_drainage_path': 'stream_and_drainage.tif',
        'stream_path': 'stream.tif',
        'usle_path': 'usle.tif',
        'watershed_results_sdr_path': 'watershed_results_sdr.shp',
        'avoided_export_path': 'avoided_export.tif',
        'avoided_erosion_path': 'avoided_erosion.tif',
    },
    intermediate_base_files={
        'aligned_dem_path': 'aligned_dem.tif',
        'aligned_drainage_path': 'aligned_drainage.tif',
        'aligned_erodibility_path': 'aligned_erodibility.tif',
        'aligned_erosivity_path': 'aligned_erosivity.tif',
        'aligned_lulc_path': 'aligned_lulc.tif',
        'mask_path': 'mask.tif',
        'masked_dem_path': 'masked_dem.tif',
        'masked_drainage_path': 'masked_drainage.tif',
        'masked_erodibility_path': 'masked_erodibility.tif',
        'masked_erosivity_path': 'masked_erosivity.tif',
        'masked_lulc_path': 'masked_lulc.tif',
        'cp_factor_path': 'cp.tif',
        'd_dn_path': 'd_dn.tif',
        'd_up_path': 'd_up.tif',
        'f_path': 'f.tif',
        'flow_accumulation_path': 'flow_accumulation.tif',
        'flow_direction_path': 'flow_direction.tif',
        'ic_path': 'ic.tif',
        'ls_path': 'ls.tif',
        'pit_filled_dem_path': 'pit_filled_dem.tif',
        's_accumulation_path': 's_accumulation.tif',
        's_bar_path': 's_bar.tif',
        'sdr_path': 'sdr_factor.tif',
        'slope_path': 'slope.tif',
        'thresholded_slope_path': 'slope_threshold.tif',
        'thresholded_w_path': 'w_threshold.tif',
        'w_accumulation_path': 'w_accumulation.tif',
        'w_bar_path': 'w_bar.tif',
        'w_path': 'w.tif',
        'ws_inverse_path': 'ws_inverse.tif',
        'e_prime_path': 'e_prime.tif',
        'drainage_mask': 'what_drains_to_stream.tif',
    }
)

def execute(args):
    """Sediment Delivery Ratio.

    This function calculates the sediment export and retention of a landscape
    using the sediment delivery ratio model described in the InVEST user's
    guide.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['erosivity_path'] (string): path to rainfall erosivity index
            raster
        args['erodibility_path'] (string): a path to soil erodibility raster
        args['lulc_path'] (string): path to land use/land cover raster
        args['watersheds_path'] (string): path to vector of the watersheds
        args['biophysical_table_path'] (string): path to CSV file with
            biophysical information of each land use classes.  contain the
            fields 'usle_c' and 'usle_p'
        args['threshold_flow_accumulation'] (number): number of upslope pixels
            on the dem to threshold to a stream.
        args['k_param'] (number): k calibration parameter
        args['sdr_max'] (number): max value the SDR
        args['ic_0_param'] (number): ic_0 calibration parameter
        args['drainage_path'] (string): (optional) path to drainage raster that
            is used to add additional drainage areas to the internally
            calculated stream layer
        args['l_max'] (number): the maximum allowed value of the slope length
            parameter (L) in the LS factor. If the calculated value of L
            exceeds 'l_max' it will be clamped to this value.
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.

    Returns:
        None.

    """
    MODEL_SPEC.execute(args)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(args, MODEL_SPEC)
