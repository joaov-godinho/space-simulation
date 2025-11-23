from pandas.io.sas.sas_constants import sas_release_length
from skyfield.api import EarthSatellite, load
import numpy as np
import pandas as pd

def satellites_to_dataframe(satellites):
    data = []
    for sat in satellites:
        data.append({
            'name': sat.name,
            'catalog_number': sat.model.satnum,
            'epoch': sat.epoch.utc_datetime(),
            'object': sat
        })
    return pd.DataFrame(data)

def load_tles_from_url(url='https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'):
    try:
        ts = load.timescale()

        satellites = load.tle_file(url)

        print(f"Successfully loaded {len(satellites)} satellites from CelesTrak.")
        return ts, satellites
    except Exception as e:
        print(f"Error loading TLEs:{e}")
        return None, None

def get_initial_state_and_time(satellite, ts, time_offset_seconds=0):
    t0_epoch_skyfield = satellite.epoch

    if time_offset_seconds != 0:
        t0 = ts.tt(jd=t0_epoch_skyfield.tt + time_offset_seconds / (24*3600))
    else:
        t0 = t0_epoch_skyfield

    geocentric_position = satellite.at(t0)

    r_vector = geocentric_position.position.km
    v_vector = geocentric_position.velocity.km_per_s

    state_initial = np.hstack((r_vector, v_vector))

    return t0, state_initial