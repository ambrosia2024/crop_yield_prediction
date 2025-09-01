
import numpy as np
import pandas as pd


def prepare_features_and_targets(dataset):
    """
    Prepared features and target from the raw data.
    """
    X_list, y_list, years_list = [], [], []

    targets_array = dataset.targets()
    indices_list = list(dataset.indices())  # [(adm_id, year), ...]

    for i, idx in enumerate(indices_list):
        adm_id, year = idx
        target = targets_array[i]

        features = {}

        # Soil
        soil_row = dataset._dfs_x['soil'].loc[adm_id]
        for col in soil_row.index:
            features[f'soil_{col}'] = soil_row[col]

        # Meteorological
        meteo_rows = dataset._dfs_x['meteo'].loc[adm_id].loc[year]
        features['meteo_tmin_mean'] = meteo_rows['tmin'].mean()
        features['meteo_tmax_mean'] = meteo_rows['tmax'].mean()
        features['meteo_tavg_mean'] = meteo_rows['tavg'].mean()
        features['meteo_prec_sum'] = meteo_rows['prec'].sum()
        features['meteo_cwb_sum'] = meteo_rows['cwb'].sum()
        features['meteo_rad_sum'] = meteo_rows['rad'].sum()

        # Remote sensing
        for key in ['fpar', 'ndvi', 'ssm']:
            try:
                rs_rows = dataset._dfs_x[key].loc[adm_id].loc[year]
                features[f'{key}_mean'] = rs_rows.iloc[:, 0].mean() if not rs_rows.empty else np.nan
            except KeyError:
                features[f'{key}_mean'] = np.nan

        # Crop season
        try:
            cs_row = dataset._dfs_x['crop_season'].loc[(adm_id, year)]
            for col in cs_row.index:
                value = cs_row[col]
                if isinstance(value, pd.Timestamp):
                    value = (value - pd.Timestamp("1970-01-01")).days
                elif pd.isnull(value):
                    value = np.nan
                features[f'crop_{col}'] = value
        except KeyError:
            for col in dataset._dfs_x['crop_season'].columns:
                features[f'crop_{col}'] = np.nan

        X_list.append(list(features.values()))
        y_list.append(target)
        years_list.append(year)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, years_list