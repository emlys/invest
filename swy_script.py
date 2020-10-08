import logging

from natcap.invest.seasonal_water_yield import seasonal_water_yield

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

args = {
        "workspace_dir": '/Users/emily/Documents/seasonal_water_yield_workspace',
        "results_suffix": 'a',
        "n_workers": -1,
        "threshold_flow_accumulation": 1000,
        "kc_dir": '/Users/emily/Downloads/Seasonal_Water_Yield/Kc_monthly',
        "et0_dir": '/Users/emily/Downloads/Seasonal_Water_Yield/ET0_monthly',
        "precip_dir": '/Users/emily/Downloads/Seasonal_Water_Yield/Precipitation_monthly',
        "dem_raster_path": '/Users/emily/Downloads/Seasonal_Water_Yield/DEM_gura.tif',
        "lulc_raster_path": '/Users/emily/Downloads/Seasonal_Water_Yield/land_use_gura.tif',
        "soil_group_path": '/Users/emily/Downloads/Seasonal_Water_Yield/soil_group_gura.tif',
        "aoi_path": '/Users/emily/Downloads/Seasonal_Water_Yield/watershed_gura.shp',
        "biophysical_table_path": '/Users/emily/Downloads/Seasonal_Water_Yield/biophysical_table_gura_SWY.csv',
        "rain_events_table_path": '/Users/emily/Downloads/Seasonal_Water_Yield/rain_events_gura.csv',
        "alpha_m": 1,
        "beta_i": 1,
        "gamma": 1,
        "user_defined_local_recharge": False,
        "l_path": None,
        "user_defined_climate_zones": False,
        "climate_zone_table_path": None,
        "climate_zone_raster_path": None,
        "monthly_alpha": False,
        "monthly_alpha_path": None
    }

seasonal_water_yield.execute(args)
