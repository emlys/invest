"""InVEST Annual Water Yield model."""
import logging
import math
import os
import pickle

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from . import gettext
from . import spec_utils
from . import utils
from . import validation
from .model_metadata import MODEL_METADATA
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

BASE_OUTPUT_FIELDS = {
    "precip_mn": {
        "type": "number",
        "units": u.mm,
        "about": "Mean precipitation per pixel in the subwatershed.",
    },
    "PET_mn": {
        "type": "number",
        "units": u.mm,
        "about": "Mean potential evapotranspiration per pixel in the subwatershed.",
    },
    "AET_mn": {
        "type": "number",
        "units": u.mm,
        "about": "Mean actual evapotranspiration per pixel in the subwatershed.",
    },
    "wyield_mn": {
        "type": "number",
        "units": u.mm,
        "about": "Mean water yield per pixel in the subwatershed.",
    },
    "wyield_vol": {
        "type": "number",
        "units": u.m**3,
        "about": "Total volume of water yield in the subwatershed.",
    }
}
SCARCITY_OUTPUT_FIELDS = {
    "consum_vol": {
        "type": "number",
        "units": u.m**3,
        "about": "Total water consumption for each watershed.",
        "created_if": "demand_table_path"
    },
    "consum_mn": {
        "type": "number",
        "units": u.meter**3/u.hectare,
        "about": "Mean water consumptive volume per pixel per watershed.",
        "created_if": "demand_table_path"
    },
    "rsupply_vl": {
        "type": "number",
        "units": u.m**3,
        "about": "Total realized water supply (water yield – consumption) volume for each watershed.",
        "created_if": "demand_table_path"
    },
    "rsupply_mn": {
        "type": "number",
        "units": u.m**3/u.hectare,
        "about": "Mean realized water supply (water yield – consumption) volume per pixel per watershed.",
        "created_if": "demand_table_path"
    }
}
VALUATION_OUTPUT_FIELDS = {
    "hp_energy": {
        "type": "number",
        "units": u.kilowatt_hour,
        "created_if": "valuation_table_path",
        "about": "The amount of ecosystem service in energy production terms. This is the amount of energy produced annually by the hydropower station that can be attributed to each watershed based on the watershed’s water yield contribution.",
    },
    "hp_val": {
        "type": "number",
        "units": u.currency,
        "created_if": "valuation_table_path",
        "about": "The amount of ecosystem service in economic terms. This shows the value of the landscape per watershed according to its ability to yield water for hydropower production over the specified timespan, and with respect to the discount rate.",
    }
}
SUBWATERSHED_OUTPUT_FIELDS = {
    "subws_id": {
        "type": "integer",
        "about": gettext("Unique identifier for each subwatershed.")
    },
    **BASE_OUTPUT_FIELDS,
    **SCARCITY_OUTPUT_FIELDS,

}
WATERSHED_OUTPUT_FIELDS = {
    "ws_id": {
        "type": "integer",
        "about": gettext("Unique identifier for each watershed.")
    },
    **BASE_OUTPUT_FIELDS,
    **SCARCITY_OUTPUT_FIELDS,
    **VALUATION_OUTPUT_FIELDS
}

MODEL_SPEC = {
    "model_name": MODEL_METADATA["annual_water_yield"].model_title,
    "pyname": MODEL_METADATA["annual_water_yield"].pyname,
    "userguide": MODEL_METADATA["annual_water_yield"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["lulc_path",
                         "depth_to_root_rest_layer_path",
                         "precipitation_path",
                         "pawc_path",
                         "eto_path",
                         "watersheds_path",
                         "sub_watersheds_path"],
        "different_projections_ok": False,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "lulc_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": spec_utils.LULC['about'] + " " + gettext(
                "All values in this raster must have corresponding entries "
                "in the Biophysical Table.")
        },
        "depth_to_root_rest_layer_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.millimeter
            }},
            "projected": True,
            "about": gettext(
                "Map of root restricting layer depth, the soil depth at "
                "which root penetration is strongly inhibited because of "
                "physical or chemical characteristics."),
            "name": gettext("root restricting layer depth")
        },
        "precipitation_path": {
            **spec_utils.PRECIP,
            "projected": True
        },
        "pawc_path": {
            "type": "raster",
            "bands": {1: {"type": "ratio"}},
            "projected": True,
            "about": gettext(
                "Map of plant available water content, the fraction of "
                "water that can be stored in the soil profile that is "
                "available to plants."),
            "name": gettext("plant available water content")
        },
        "eto_path": {
            **spec_utils.ET0,
            "projected": True
        },
        "watersheds_path": {
            "projected": True,
            "type": "vector",
            "fields": {
                "ws_id": {
                    "type": "integer",
                    "about": gettext("Unique identifier for each watershed.")
                }
            },
            "geometries": spec_utils.POLYGON,
            "about": gettext(
                "Map of watershed boundaries, such that each watershed drains "
                "to a point of interest where hydropower production will be "
                "analyzed."),
            "name": gettext("watersheds")
        },
        "sub_watersheds_path": {
            "projected": True,
            "type": "vector",
            "fields": {
                "subws_id": {
                    "type": "integer",
                    "about": gettext("Unique identifier for each subwatershed.")
                }
            },
            "geometries": spec_utils.POLYGONS,
            "required": False,
            "about": gettext(
                "Map of subwatershed boundaries within each watershed in "
                "the Watersheds map."),
            "name": gettext("sub-watersheds")
        },
        "biophysical_table_path": {
            "type": "csv",
            "columns": {
                "lucode": spec_utils.LULC_TABLE_COLUMN,
                "lulc_veg": {
                    "type": "integer",
                    "about": gettext(
                        "Code indicating whether the the LULC class is "
                        "vegetated for the purpose of AET. Enter 1 for all "
                        "vegetated classes except wetlands, and 0 for all "
                        "other classes, including wetlands, urban areas, "
                        "water bodies, etc.")
                },
                "root_depth": {
                    "type": "number",
                    "units": u.millimeter,
                    "about": gettext(
                        "Maximum root depth for plants in this LULC class. "
                        "Only used for classes with a 'lulc_veg' value of 1.")
                },
                "kc": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("Crop coefficient for this LULC class.")}
            },
            "index_col": "lucode",
            "about": gettext(
                "Table of biophysical parameters for each LULC class. All "
                "values in the LULC raster must have corresponding entries "
                "in this table."),
            "name": gettext("biophysical table")
        },
        "seasonality_constant": {
            "expression": "value > 0",
            "type": "number",
            "units": u.none,
            "about": gettext(
                "The seasonality factor, representing hydrogeological "
                "characterisitics and the seasonal distribution of "
                "precipitation. Values typically range from 1 - 30."),
            "name": gettext("z parameter")
        },
        "demand_table_path": {
            "type": "csv",
            "columns": {
                "lucode": {
                    "about": gettext("LULC code corresponding to the LULC raster"),
                    "type": "integer"
                },
                "demand": {
                    "about": gettext(
                        "Average consumptive water use in this LULC class."),
                    "type": "number",
                    "units": u.meter**3/u.year/u.pixel
                }
            },
            "index_col": "lucode",
            "required": False,
            "about": gettext(
                "A table of water demand for each LULC class. Each LULC code "
                "in the LULC raster must have a corresponding row in this "
                "table."),
            "name": gettext("water demand table")
        },
        "valuation_table_path": {
            "type": "csv",
            "columns": {
                "ws_id": {
                    "type": "integer",
                    "about": gettext(
                        "Unique identifier for the hydropower station. This "
                        "must match the 'ws_id' value for the corresponding "
                        "watershed in the Watersheds vector. Each watershed "
                        "in the Watersheds vector must have its 'ws_id' "
                        "entered in this column.")
                },
                "efficiency": {
                    "type": "ratio",
                    "about": gettext(
                        "Turbine efficiency, the proportion of potential "
                        "energy captured and converted to electricity by the "
                        "turbine.")
                },
                "fraction": {
                    "type": "ratio",
                    "about": gettext(
                        "The proportion of inflow water volume that is used "
                        "to generate energy.")
                },
                "height": {
                    "type": "number",
                    "units": u.meter,
                    "about": gettext(
                        "The head, measured as the average annual effective "
                        "height of water behind each dam at the turbine "
                        "intake.")
                },
                "kw_price": {
                    "type": "number",
                    "units": u.currency/u.kilowatt_hour,
                    "about": gettext(
                        "The price of power produced by the station. Must be "
                        "in the same currency used in the 'cost' column.")
                },
                "cost": {
                    "type": "number",
                    "units": u.currency/u.year,
                    "about": gettext(
                        "Annual maintenance and operations cost of running "
                        "the hydropower station. Must be in the same currency "
                        "used in the 'kw_price' column.")
                },
                "time_span": {
                    "type": "number",
                    "units": u.year,
                    "about": gettext(
                        "Number of years over which to value the "
                        "hydropower station. This is either the station's "
                        "expected lifespan or the duration of the land use "
                        "scenario of interest.")
                },
                "discount": {
                    "type": "percent",
                    "about": gettext(
                        "The annual discount rate, applied for each year in "
                        "the time span.")
                }
            },
            "index_col": "ws_id",
            "required": False,
            "about": gettext(
                "A table mapping each watershed to the associated valuation "
                "parameters for its hydropower station."),
            "name": gettext("hydropower valuation table")
        }
    },
    "outputs": {
        "output": {
            "type": "directory",
            "contents": {
                "watershed_results_wyield.shp": {
                    "fields": {**WATERSHED_OUTPUT_FIELDS},
                    "geometries": spec_utils.POLYGON,
                    "about": "Shapefile containing biophysical output values per watershed."
                },
                "watershed_results_wyield.csv": {
                    "columns": {**WATERSHED_OUTPUT_FIELDS},
                    "index_col": "ws_id",
                    "about": "Table containing biophysical output values per watershed."
                },
                "subwatershed_results_wyield.shp": {
                    "fields": {**SUBWATERSHED_OUTPUT_FIELDS},
                    "geometries": spec_utils.POLYGON,
                    "about": "Shapefile containing biophysical output values per subwatershed."
                },
                "subwatershed_results_wyield.csv": {
                    "columns": {**SUBWATERSHED_OUTPUT_FIELDS},
                    "index_col": "subws_id",
                    "about": "Table containing biophysical output values per subwatershed."
                },
                "per_pixel": {
                    "type": "directory",
                    "about": "Outputs in the per_pixel folder can be useful for intermediate calculations but should NOT be interpreted at the pixel level, as model assumptions are based on processes understood at the subwatershed scale.",
                    "contents": {
                        "fractp.tif": {
                            "about": (
                                "The fraction of precipitation that actually "
                                "evapotranspires at the pixel level."),
                            "bands": {1: {"type": "ratio"}}
                        },
                        "aet.tif": {
                            "about": "Estimated actual evapotranspiration per pixel.",
                            "bands": {
                                1: {"type": "number", "units": u.millimeter}
                            }
                        },
                        "wyield.tif": {
                            "about": "Estimated water yield per pixel.",
                            "bands": {
                                1: {"type": "number", "units": u.millimeter}
                            }
                        }
                    }
                }
            }
        },
        "intermediate": {
            "type": "directory",
            "contents": {
                "clipped_lulc.tif": {
                    "about": "Aligned and clipped copy of LULC input.",
                    "bands": {1: {"type": "integer"}}
                },
                "depth_to_root_rest_layer.tif": {
                    "about": (
                        "Aligned and clipped copy of root restricting "
                        "layer depth input."),
                    "bands": {
                        1: {"type": "number", "units": u.millimeter}
                    }
                },
                "eto.tif": {
                    "about": "Aligned and clipped copy of ET0 input.",
                    "bands": {
                        1: {"type": "number", "units": u.millimeter}
                    }
                },
                "kc_raster.tif": {
                    "about": "Map of KC values.",
                    "bands": {
                        1: {"type": "number", "units": u.none}
                    }
                },
                "pawc.tif": {
                    "about": "Aligned and clipped copy of PAWC input.",
                    "bands": {1: {"type": "ratio"}},
                },
                "pet.tif": {
                    "about": "Map of potential evapotranspiration.",
                    "bands": {
                        1: {"type": "number", "units": u.millimeter}
                    }
                },
                "precip.tif": {
                    "about": "Aligned and clipped copy of precipitation input.",
                    "bands": {
                        1: {"type": "number", "units": u.millimeter}
                    }
                },
                "root_depth.tif": {
                    "about": "Map of root depth.",
                    "bands": {
                        1: {"type": "number", "units": u.millimeter}
                    }
                },
                "veg.tif": {
                    "about": "Map of vegetated state.",
                    "bands": {1: {"type": "integer"}},
                }
            }
        },
        "taskgraph_dir": spec_utils.TASKGRAPH_DIR
    }
}


def execute(args):
    """Annual Water Yield: Reservoir Hydropower Production.

    Executes the hydropower/annual water yield model

    Args:
        args['workspace_dir'] (string): a path to the directory that will write
            output and other temporary files during calculation. (required)

        args['lulc_path'] (string): a path to a land use/land cover raster
            whose LULC indexes correspond to indexes in the biophysical table
            input. Used for determining soil retention and other biophysical
            properties of the landscape. (required)

        args['depth_to_root_rest_layer_path'] (string): a path to an input
            raster describing the depth of "good" soil before reaching this
            restrictive layer (required)

        args['precipitation_path'] (string): a path to an input raster
            describing the average annual precipitation value for each cell
            (mm) (required)

        args['pawc_path'] (string): a path to an input raster describing the
            plant available water content value for each cell. Plant Available
            Water Content fraction (PAWC) is the fraction of water that can be
            stored in the soil profile that is available for plants' use.
            PAWC is a fraction from 0 to 1 (required)

        args['eto_path'] (string): a path to an input raster describing the
            annual average evapotranspiration value for each cell. Potential
            evapotranspiration is the potential loss of water from soil by
            both evaporation from the soil and transpiration by healthy
            Alfalfa (or grass) if sufficient water is available (mm)
            (required)

        args['watersheds_path'] (string): a path to an input shapefile of the
            watersheds of interest as polygons. (required)

        args['sub_watersheds_path'] (string): a path to an input shapefile of
            the subwatersheds of interest that are contained in the
            ``args['watersheds_path']`` shape provided as input. (optional)

        args['biophysical_table_path'] (string): a path to an input CSV table
            of land use/land cover classes, containing data on biophysical
            coefficients such as root_depth (mm) and Kc, which are required.
            A column with header LULC_veg is also required which should
            have values of 1 or 0, 1 indicating a land cover type of
            vegetation, a 0 indicating non vegetation or wetland, water.
            NOTE: these data are attributes of each LULC class rather than
            attributes of individual cells in the raster map (required)

        args['seasonality_constant'] (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation (required)

        args['results_suffix'] (string): a string that will be concatenated
            onto the end of file names (optional)

        args['demand_table_path'] (string): (optional) if a non-empty string,
            a path to an input CSV
            table of LULC classes, showing consumptive water use for each
            landuse / land-cover type (cubic meters per year) to calculate
            water scarcity.

        args['valuation_table_path'] (string): (optional) if a non-empty
            string, a path to an input CSV table of
            hydropower stations with the following fields to calculate
            valuation: 'ws_id', 'time_span', 'discount', 'efficiency',
            'fraction', 'cost', 'height', 'kw_price'
            Required if ``calculate_valuation`` is True.

        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None

    """
    # valuation_df is passed to create_vector_output()
    # which computes valuation if valuation_df is not None.
    valuation_df = None
    if 'valuation_table_path' in args and args['valuation_table_path'] != '':
        LOGGER.info(
            'Checking that watersheds have entries for every `ws_id` in the '
            'valuation table.')
        # Open/read in valuation parameters from CSV file
        valuation_df = validation.get_validated_dataframe(
            args['valuation_table_path'],
            **MODEL_SPEC['args']['valuation_table_path'])
        watershed_vector = gdal.OpenEx(
            args['watersheds_path'], gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        missing_ws_ids = []
        for watershed_feature in watershed_layer:
            watershed_ws_id = watershed_feature.GetField('ws_id')
            if watershed_ws_id not in valuation_df.index:
                missing_ws_ids.append(watershed_ws_id)
        watershed_feature = None
        watershed_layer = None
        watershed_vector = None
        if missing_ws_ids:
            raise ValueError(
                'The following `ws_id`s exist in the watershed vector file '
                'but are not found in the valuation table. Check your '
                'valuation table to see if they are missing: '
                f'"{", ".join(str(x) for x in sorted(missing_ws_ids))}"')

    # Construct folder paths
    workspace_dir = args['workspace_dir']
    output_dir = os.path.join(workspace_dir, 'output')
    per_pixel_output_dir = os.path.join(output_dir, 'per_pixel')
    intermediate_dir = os.path.join(workspace_dir, 'intermediate')
    pickle_dir = os.path.join(intermediate_dir, '_tmp_zonal_stats')
    utils.make_directories(
        [workspace_dir, output_dir, per_pixel_output_dir,
         intermediate_dir, pickle_dir])

    # Append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Paths for targets of align_and_resize_raster_stack
    clipped_lulc_path = os.path.join(
        intermediate_dir, f'clipped_lulc{file_suffix}.tif')
    eto_path = os.path.join(intermediate_dir, f'eto{file_suffix}.tif')
    precip_path = os.path.join(intermediate_dir, f'precip{file_suffix}.tif')
    depth_to_root_rest_layer_path = os.path.join(
        intermediate_dir, f'depth_to_root_rest_layer{file_suffix}.tif')
    pawc_path = os.path.join(intermediate_dir, f'pawc{file_suffix}.tif')
    tmp_pet_path = os.path.join(intermediate_dir, f'pet{file_suffix}.tif')

    # Paths for output rasters
    fractp_path = os.path.join(
        per_pixel_output_dir, f'fractp{file_suffix}.tif')
    wyield_path = os.path.join(
        per_pixel_output_dir, f'wyield{file_suffix}.tif')
    aet_path = os.path.join(per_pixel_output_dir, f'aet{file_suffix}.tif')
    demand_path = os.path.join(intermediate_dir, f'demand{file_suffix}.tif')
    veg_raster_path = os.path.join(intermediate_dir, f'veg{file_suffix}.tif')
    root_raster_path = os.path.join(
        intermediate_dir, f'root_depth{file_suffix}.tif')
    kc_raster_path = os.path.join(
        intermediate_dir, f'kc_raster{file_suffix}.tif')

    watersheds_path = args['watersheds_path']
    watershed_results_vector_path = os.path.join(
        output_dir, f'watershed_results_wyield{file_suffix}.shp')
    watershed_paths_list = [
        (watersheds_path, 'ws_id', watershed_results_vector_path)]

    sub_watersheds_path = None
    if 'sub_watersheds_path' in args and args['sub_watersheds_path'] != '':
        sub_watersheds_path = args['sub_watersheds_path']
        subwatershed_results_vector_path = os.path.join(
            output_dir, f'subwatershed_results_wyield{file_suffix}.shp')
        watershed_paths_list.append(
            (sub_watersheds_path, 'subws_id',
             subwatershed_results_vector_path))

    seasonality_constant = float(args['seasonality_constant'])

    # Initialize a TaskGraph
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    graph = taskgraph.TaskGraph(
        os.path.join(args['workspace_dir'], 'taskgraph_cache'), n_workers)

    base_raster_path_list = [
        args['eto_path'],
        args['precipitation_path'],
        args['depth_to_root_rest_layer_path'],
        args['pawc_path'],
        args['lulc_path']]

    aligned_raster_path_list = [
        eto_path,
        precip_path,
        depth_to_root_rest_layer_path,
        pawc_path,
        clipped_lulc_path]

    target_pixel_size = pygeoprocessing.get_raster_info(
        args['lulc_path'])['pixel_size']
    align_raster_stack_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        args=(base_raster_path_list, aligned_raster_path_list,
              ['near'] * len(base_raster_path_list),
              target_pixel_size, 'intersection'),
        kwargs={'raster_align_index': 4,
                'base_vector_path_list': [watersheds_path]},
        target_path_list=aligned_raster_path_list,
        task_name='align_raster_stack')
    # Joining now since this task will always be the root node
    # and it's useful to have the raster info available.
    align_raster_stack_task.join()

    nodata_dict = {
        'out_nodata': -1,
        'precip': pygeoprocessing.get_raster_info(precip_path)['nodata'][0],
        'eto': pygeoprocessing.get_raster_info(eto_path)['nodata'][0],
        'depth_root': pygeoprocessing.get_raster_info(
            depth_to_root_rest_layer_path)['nodata'][0],
        'pawc': pygeoprocessing.get_raster_info(pawc_path)['nodata'][0],
        'lulc': pygeoprocessing.get_raster_info(clipped_lulc_path)['nodata'][0]}

    # Open/read in the csv file into a dictionary and add to arguments
    bio_df = validation.get_validated_dataframe(args['biophysical_table_path'],
                                         **MODEL_SPEC['args']['biophysical_table_path'])
    bio_lucodes = set(bio_df.index.values)
    bio_lucodes.add(nodata_dict['lulc'])
    LOGGER.debug(f'bio_lucodes: {bio_lucodes}')

    if 'demand_table_path' in args and args['demand_table_path'] != '':
        demand_df = validation.get_validated_dataframe(
            args['demand_table_path'], **MODEL_SPEC['args']['demand_table_path'])
        demand_reclassify_dict = dict(
            [(lucode, row['demand']) for lucode, row in demand_df.iterrows()])
        demand_lucodes = set(demand_df.index.values)
        demand_lucodes.add(nodata_dict['lulc'])
        LOGGER.debug(f'demand_lucodes: {demand_lucodes}', )
    else:
        demand_lucodes = None

    # Break the bio_df into three separate dictionaries based on
    # Kc, root_depth, and LULC_veg fields to use for reclassifying
    Kc_dict = {}
    root_dict = {}
    vegetated_dict = {}

    for lulc_code, row in bio_df.iterrows():
        Kc_dict[lulc_code] = row['kc']

        # Catch invalid LULC_veg values with an informative error.
        if row['lulc_veg'] not in set([0, 1]):
            # If the user provided an invalid LULC_veg value, raise an
            # informative error.
            raise ValueError(
                f'LULC_veg value must be either 1 or 0, not {row["lulc_veg"]}')
        vegetated_dict[lulc_code] = row['lulc_veg']

        # If LULC_veg value is 1 get root depth value
        if vegetated_dict[lulc_code] == 1:
            root_dict[lulc_code] = row['root_depth']
        # If LULC_veg value is 0 then we do not care about root
        # depth value so will just substitute in a 1. This
        # value will not end up being used.
        else:
            root_dict[lulc_code] = 1

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    # Create Kc raster from table values to use in future calculations
    LOGGER.info("Reclassifying temp_Kc raster")
    create_Kc_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), Kc_dict, kc_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[kc_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_Kc_raster')

    # Create root raster from table values to use in future calculations
    LOGGER.info("Reclassifying tmp_root raster")
    create_root_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), root_dict, root_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[root_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_root_raster')

    # Create veg raster from table values to use in future calculations
    # of determining which AET equation to use
    LOGGER.info("Reclassifying tmp_veg raster")
    create_veg_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), vegetated_dict, veg_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[veg_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_veg_raster')

    dependent_tasks_for_watersheds_list = []

    LOGGER.info('Calculate PET from Ref Evap times Kc')
    calculate_pet_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.multiply,  # PET = ET0 * KC
            rasters=[eto_path, kc_raster_path],
            target_path=tmp_pet_path,
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[tmp_pet_path],
        dependent_task_list=[create_Kc_raster_task],
        task_name='calculate_pet')
    dependent_tasks_for_watersheds_list.append(calculate_pet_task)

    # List of rasters to pass into the vectorized fractp operation
    raster_list = [
        kc_raster_path, eto_path, precip_path, root_raster_path,
        depth_to_root_rest_layer_path, pawc_path, veg_raster_path]

    LOGGER.debug('Performing fractp operation')
    calculate_fractp_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(x, 1) for x in raster_list]
              + [(nodata_dict, 'raw'), (seasonality_constant, 'raw')],
              fractp_op, fractp_path, gdal.GDT_Float32,
              nodata_dict['out_nodata']),
        target_path_list=[fractp_path],
        dependent_task_list=[
            create_Kc_raster_task, create_veg_raster_task,
            create_root_raster_task, align_raster_stack_task],
        task_name='calculate_fractp')

    LOGGER.info('Performing wyield operation')
    calculate_wyield_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=wyield_op,
            rasters=[fractp_path, precip_path],
            target_path=wyield_path,
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[wyield_path],
        dependent_task_list=[calculate_fractp_task, align_raster_stack_task],
        task_name='calculate_wyield')
    dependent_tasks_for_watersheds_list.append(calculate_wyield_task)

    LOGGER.debug('Performing aet operation')
    calculate_aet_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.multiply,  # AET = fractp * precip
            rasters=[fractp_path, precip_path],
            target_path=aet_path,
            target_nodata=nodata_dict['out_nodata']),
        target_path_list=[aet_path],
        dependent_task_list=[
            calculate_fractp_task, create_veg_raster_task,
            align_raster_stack_task],
        task_name='calculate_aet')
    dependent_tasks_for_watersheds_list.append(calculate_aet_task)

    # list of rasters that will always be summarized with zonal stats
    raster_names_paths_list = [
        ('precip_mn', precip_path),
        ('PET_mn', tmp_pet_path),
        ('AET_mn', aet_path),
        ('wyield_mn', wyield_path)]

    demand = False
    if 'demand_table_path' in args and args['demand_table_path'] != '':
        demand = True
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Demand'}
        # Create demand raster from table values to use in future calculations
        create_demand_raster_task = graph.add_task(
            func=utils.reclassify_raster,
            args=((clipped_lulc_path, 1), demand_reclassify_dict, demand_path,
                  gdal.GDT_Float32, nodata_dict['out_nodata'],
                  reclass_error_details),
            target_path_list=[demand_path],
            dependent_task_list=[align_raster_stack_task],
            task_name='create_demand_raster')
        dependent_tasks_for_watersheds_list.append(create_demand_raster_task)
        raster_names_paths_list.append(('demand', demand_path))

    # Aggregate results to watershed polygons, and do the optional
    # scarcity and valuation calculations.
    for base_ws_path, ws_id_name, target_ws_path in watershed_paths_list:
        # make a copy so we don't modify the original
        # do zonal stats with the copy so that FIDS are correct
        copy_watersheds_vector_task = graph.add_task(
            func=copy_vector,
            args=[base_ws_path, target_ws_path],
            target_path_list=[target_ws_path],
            task_name='create copy of watersheds vector')

        # Do zonal stats with the input shapefiles provided by the user
        # and store results dictionaries in pickles
        zonal_stats_pickle = os.path.join(
            pickle_dir,
            f'zonal_stats{file_suffix}.pickle')
        zonal_stats_task = graph.add_task(
            func=zonal_stats_tofile,
            args=(target_ws_path, raster_names_paths_list, zonal_stats_pickle),
            target_path_list=[zonal_stats_pickle],
            dependent_task_list=[
                *dependent_tasks_for_watersheds_list,
                copy_watersheds_vector_task],
            task_name=f'{ws_id_name}_zonalstats')

        # Add the zonal stats data to the output vector's attribute table
        # Compute optional scarcity and valuation
        write_output_vector_attributes_task = graph.add_task(
            func=write_output_vector_attributes,
            args=(target_ws_path, ws_id_name, zonal_stats_pickle,
                  valuation_df),
             kwargs={'demand': demand},
            target_path_list=[target_ws_path],
            dependent_task_list=[
                zonal_stats_task, copy_watersheds_vector_task],
            task_name=f'create_{ws_id_name}_vector_output')

        # Export a CSV with all the fields present in the output vector
        target_basename = os.path.splitext(target_ws_path)[0]
        target_csv_path = target_basename + '.csv'
        create_output_table_task = graph.add_task(
            func=convert_vector_to_csv,
            args=(target_ws_path, target_csv_path),
            target_path_list=[target_csv_path],
            dependent_task_list=[write_output_vector_attributes_task],
            task_name=f'create_{ws_id_name}_table_output')

    graph.join()


# wyield equation to pass to raster_map
def wyield_op(fractp, precip): return (1 - fractp) * precip


def copy_vector(base_vector_path, target_vector_path):
    """Wrapper around CreateCopy that handles opening & closing the dataset.

    Args:
        base_vector_path: path to the vector to copy
        target_vector_path: path to copy the vector to

    Returns:
        None
    """
    esri_shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    base_dataset = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    esri_shapefile_driver.CreateCopy(target_vector_path, base_dataset)
    base_dataset = None


def write_output_vector_attributes(target_vector_path, ws_id_name,
                                   pickle_path, valuation_df, demand=False):
    """Add data attributes to the vector outputs of this model.

    Join results of zonal stats to copies of the watershed shapefiles.
    Also do optional scarcity and valuation calculations.

    Args:
        target_vector_path (string): Path to the watersheds vector to modify
        ws_id_name (string): Either 'ws_id' or 'subws_id', which are required
            names of a unique ID field in the watershed and subwatershed
            shapefiles, respectively. Used to determine if the polygons
            represent watersheds or subwatersheds.
        stats_path_list (list): List of file paths to pickles storing the zonal
            stats results.
        valuation_df (pandas.DataFrame): dataframe built from
            args['valuation_table_path']. Or None if valuation table was not
            provided.

    Returns:
        None

    """

    with open(pickle_path, 'rb') as picklefile:
        stats = pickle.load(picklefile)

    new_fields = ['wyield_mn', 'wyield_vol', 'precip_mn', 'PET_mn', 'AET_mn']
    if 'demand' in stats:
        new_fields += ['consum_mn', 'consum_vol', 'rsupply_mn', 'rsupply_vl']
    if valuation_df is not None and ws_id_name == 'ws_id':
        new_fields += ['hp_energy', 'hp_val']

    def stats_op(feature):
        fid = feature.GetFID()
        # Using the unique feature ID, index into the
        # dictionary to get the corresponding value
        # only write a value if zonal stats found valid pixels in the polygon:
        if stats['wyield_mn'][fid]['count'] > 0:
            wyield_mn = (
                stats['wyield_mn'][fid]['sum'] /
                stats['wyield_mn'][fid]['count'])
            # Calculate water yield volume,
            # 1000 is for converting the mm of wyield to meters
            wyield_vol = wyield_mn * feature.GetGeometryRef().Area() / 1000
            feature.SetField('wyield_mn', wyield_mn)
            feature.SetField('wyield_vol', wyield_vol)

        if 'demand' in stats and stats['demand'][fid]['count'] > 0:
            consum_vol = stats['demand'][fid]['sum']
            consum_mn = consum_vol / stats['demand'][fid]['count']
            feature.SetField('consum_mn', consum_mn)
            feature.SetField('consum_vol', consum_vol)

        if ('demand' in stats and stats['demand'][fid]['count'] > 0 and
                stats['wyield_mn'][fid]['count'] > 0):
            # Calculate realized supply
            # these values won't exist if the polygon feature only
            # covers nodata raster values, so check before doing math.
            rsupply_vol = wyield_vol - consum_vol
            rsupply_mn = wyield_mn - consum_mn
            feature.SetField('rsupply_mn', rsupply_mn)
            feature.SetField('rsupply_vl', rsupply_vol)

            if valuation_df is not None and ws_id_name == 'ws_id':
                # Compute hydropower energy production (KWH)
                # This is from the equation given in the Users' Guide
                ws_id = feature.GetField('ws_id')
                energy = (
                    valuation_df['efficiency'][ws_id] * valuation_df['fraction'][ws_id] *
                    valuation_df['height'][ws_id] * rsupply_vol * 0.00272)

                # Divide by 100 because it is input at a percent and we need
                # decimal value
                disc = valuation_df['discount'][ws_id] / 100
                # To calculate the summation of the discount rate term over the life
                # span of the dam we can use a geometric series
                ratio = 1 / (1 + disc)
                dsum = 0
                if ratio != 1:
                    dsum = (1 - math.pow(ratio, valuation_df['time_span'][ws_id])) / (1 - ratio)

                npv = ((valuation_df['kw_price'][ws_id] * energy) - valuation_df['cost'][ws_id]) * dsum

                # Get the volume field index and add value
                feature.SetField('hp_energy', energy)
                feature.SetField('hp_val', npv)

        if stats['precip_mn'][fid]['count'] > 0:
            feature.SetField('precip_mn',
                (stats['precip_mn'][fid]['sum'] /
                    stats['precip_mn'][fid]['count']))

        if stats['PET_mn'][fid]['count'] > 0:
            feature.SetField('PET_mn',
                (stats['PET_mn'][fid]['sum'] / stats['PET_mn'][fid]['count']))

        if stats['AET_mn'][fid]['count'] > 0:
            feature.SetField('AET_mn',
                (stats['AET_mn'][fid]['sum'] / stats['AET_mn'][fid]['count']))

    utils.vector_apply(target_vector_path, stats_op,
        new_fields=new_fields)


def convert_vector_to_csv(base_vector_path, target_csv_path):
    """Create a CSV with all the fields present in vector attribute table.

    Args:
        base_vector_path (string):
            Path to the watershed shapefile in the output workspace.
        target_csv_path (string):
            Path to a CSV to create in the output workspace.

    Returns:
        None

    """
    watershed_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    csv_driver = gdal.GetDriverByName('CSV')
    _ = csv_driver.CreateCopy(target_csv_path, watershed_vector)


def zonal_stats_tofile(base_vector_path, raster_path_list, target_stats_pickle):
    """Calculate zonal statistics for watersheds and write results to a file.

    Args:
        base_vector_path (string): Path to the watershed shapefile in the
            output workspace.
        raster_path (string): Path to raster to aggregate.
        target_stats_pickle (string): Path to pickle file to store dictionary
            returned by zonal stats.

    Returns:
        None

    """
    stats = {}
    for key, raster_path in raster_path_list:
        stats[key] = pygeoprocessing.zonal_statistics(
            (raster_path, 1), base_vector_path, ignore_nodata=True)
    with open(target_stats_pickle, 'wb') as picklefile:
        picklefile.write(pickle.dumps(stats))


def fractp_op(
        Kc, eto, precip, root, soil, pawc, veg,
        nodata_dict, seasonality_constant):
    """Calculate actual evapotranspiration fraction of precipitation.

    Args:
        Kc (numpy.ndarray): Kc (plant evapotranspiration
          coefficient) raster values
        eto (numpy.ndarray): potential evapotranspiration raster
          values (mm)
        precip (numpy.ndarray): precipitation raster values (mm)
        root (numpy.ndarray): root depth (maximum root depth for
           vegetated land use classes) raster values (mm)
        soil (numpy.ndarray): depth to root restricted layer raster
            values (mm)
        pawc (numpy.ndarray): plant available water content raster
           values
        veg (numpy.ndarray): 1 or 0 where 1 depicts the land type as
            vegetation and 0 depicts the land type as non
            vegetation (wetlands, urban, water, etc...). If 1 use
            regular AET equation if 0 use: AET = Kc * ETo
        nodata_dict (dict): stores nodata values keyed by raster names
        seasonality_constant (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation.

    Returns:
        numpy.ndarray (float) of actual evapotranspiration as fraction
            of precipitation.

    """
    # Kc, root, & veg were created by reclassify_raster, which set nodata
    # to out_nodata. All others are products of align_and_resize_raster_stack
    # and retain their original nodata values.
    # out_nodata is defined above and should never be None.
    valid_mask = (
        ~pygeoprocessing.array_equals_nodata(Kc, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(root, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(veg, nodata_dict['out_nodata']) &
        ~pygeoprocessing.array_equals_nodata(precip, 0))
    if nodata_dict['eto'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(eto, nodata_dict['eto'])
    if nodata_dict['precip'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(precip, nodata_dict['precip'])
    if nodata_dict['depth_root'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(
            soil, nodata_dict['depth_root'])
    if nodata_dict['pawc'] is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(pawc, nodata_dict['pawc'])

    # Compute Budyko Dryness index
    # Use the original AET equation if the land cover type is vegetation
    # If not vegetation (wetlands, urban, water, etc...) use
    # Alternative equation Kc * Eto
    phi = (Kc[valid_mask] * eto[valid_mask]) / precip[valid_mask]
    pet = Kc[valid_mask] * eto[valid_mask]

    # Calculate plant available water content (mm) using the minimum
    # of soil depth and root depth
    awc = numpy.where(
        root[valid_mask] < soil[valid_mask], root[valid_mask],
        soil[valid_mask]) * pawc[valid_mask]
    climate_w = (
        (awc / precip[valid_mask]) * seasonality_constant) + 1.25
    # Capping to 5 to set to upper limit if exceeded
    climate_w[climate_w > 5] = 5

    # Compute evapotranspiration partition of the water balance
    aet_p = (
        1 + (pet / precip[valid_mask])) - (
            (1 + (pet / precip[valid_mask]) ** climate_w) ** (
                1 / climate_w))

    # We take the minimum of the following values (phi, aet_p)
    # to determine the evapotranspiration partition of the
    # water balance (see users guide)
    veg_result = numpy.where(phi < aet_p, phi, aet_p)
    # Take the minimum of precip and Kc * ETo to avoid x / p > 1
    nonveg_result = Kc[valid_mask] * eto[valid_mask]
    nonveg_mask = precip[valid_mask] < Kc[valid_mask] * eto[valid_mask]
    nonveg_result[nonveg_mask] = precip[valid_mask][nonveg_mask]
    nonveg_result_fract = nonveg_result / precip[valid_mask]

    # If veg is 1 use the result for vegetated areas else use result
    # for non veg areas
    result = numpy.where(
        veg[valid_mask] == 1,
        veg_result, nonveg_result_fract)

    fractp = numpy.empty(valid_mask.shape, dtype=numpy.float32)
    fractp[:] = nodata_dict['out_nodata']
    fractp[valid_mask] = result
    return fractp


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
    return validation.validate(
        args, MODEL_SPEC['args'], MODEL_SPEC['args_with_spatial_overlap'])
