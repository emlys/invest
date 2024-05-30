"""
    Calculate local recharge, and intermediate results::

        l_i = p_i - qf_i - aet_i

        aet_i = (sum over m ∈ M of aet_im)

        aet_im = min(pet_im, (p_im - qf_im + a_m * b_i * l_sum_avail_i))

        pet_im = kc_im * et0_im

        l_sum_avail_i = sum over j ∈ J of p_ij * (l_avail_j + l_sum_avail_j)

        l_avail_i = min(γ * l_i, l_i)

    where:

    - ``p_i`` is the annual precipitation on pixel ``i``
    - ``qf_i`` is the annual quickflow on pixel ``i``
    - ``aet_i`` is the annual actual evapotranspiration on pixel ``i``
    - ``aet_im`` is the actual evapotranspiration on pixel ``i`` in month ``m``
    - ``pet_im`` is the potential evapotranspiration on pixel ``i`` in month ``m``
    - ``p_im`` is the precipitation on pixel ``i`` in month ``m``
    - ``qf_im`` is the quickflow on pixel ``i`` in month ``m``
    - ``a_m`` is the alpha value for month ``m``
    - ``b_i`` is the beta factor
    - ``kc_im`` is the crop factor on pixel ``i`` in month ``m`` 
        (from ``kc_path_list``)
    - ``et0_im`` is the ET0 on pixel ``i`` in month ``m`` (from ``et0_path_list``)
    - ``M`` is the set of months 1 - 12
    - ``p_ij`` is the proportion of flow from pixel i to pixel j

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


"""Calculate baseflow and baseflow sum.

    Baseflow sum is defined recursively in terms of each pixel's
    downslope neighbors::

        b_sum_i = l_sum_i * sum over j ∈ J of

          ⎧ p_ij * (1 - l_avail_j / l_sum_j) *
          ⎪         b_sum_j / (l_sum_j - l_j) ..... if j is not a stream pixel
          ⎨
          ⎩ p_ij .................................. if j is a stream pixel


    where:

    - ``p_ij`` is the proportion of flow from pixel ``i`` into pixel ``j``
    - ``J`` is the set of pixels that are immediate downslope neighbors of
      pixel ``i``
    - ``l_avail_j`` is the available recharge on pixel ``j``
      (``l_avail_path``)
    - ``l_sum_j`` is the cumulative upstream recharge on pixel ``j``
      (``l_sum_path``)
    - ``l_j`` is the local recharge on pixel ``j`` (``l_path``)

    Therefore, for an outlet pixel ``k`` whose downslope neighbors ``J`` are
    all stream pixels, ``b_sum_k = l_sum_k``. Calculation begins from these
    outlet pixels and proceeds recursively upslope.

    Baseflow is calculated as:

        b_i = max(b_sum_i * l_i / l_sum_i, 0)


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