from src.data_handler import get_initial_state_and_time
from src.orbital_mechanics import runge_kutta_4
import numpy as np
import pandas as pd
from skyfield.api import wgs84

# Constantes de Simulação
DELTA_T_SECONDS = 60
DURATION_HOURS = 1
NUM_STEPS = int(DURATION_HOURS * 3600 / DELTA_T_SECONDS)


def run_multi_object_simulation_and_validate(ts, list_of_satellites):

    all_results = []

    time_series = np.arange(0, DURATION_HOURS * 3600 + DELTA_T_SECONDS, DELTA_T_SECONDS)

    for i, satellite in enumerate(list_of_satellites):
        print(f"  -> Propagando e Validando {satellite.name} ({i + 1}/{len(list_of_satellites)})")

        # 1. Obter o estado inicial (r, v) e tempo inicial
        t0, state_initial = get_initial_state_and_time(satellite, ts)

        # 2. PROCESSO DE PROPAGAÇÃO
        # 2.1. Propagação do RK4 (com J2)
        _, state_history_rk4 = runge_kutta_4(
            state_initial, DELTA_T_SECONDS, NUM_STEPS
        )

        # 2.2. Propagação com Skyfield (Referência SGP4/SDP4)
        r_skyfield_history = np.zeros((NUM_STEPS + 1, 3))
        t_skyfield_series = ts.tt_jd(t0.tt + time_series / (24 * 3600))

        for j, t_skyfield in enumerate(t_skyfield_series):
            geocentric_position = satellite.at(t_skyfield)
            r_vector = geocentric_position.position.km
            r_skyfield_history[j, :] = r_vector

        # 3. ANÁLISE DE ERRO
        # ------------------

        final_r_rk4 = state_history_rk4[-1, 0:3]
        final_r_skyfield = r_skyfield_history[-1, :]

        difference_vector = final_r_rk4 - final_r_skyfield
        error_magnitude = np.linalg.norm(difference_vector)  # Erro em km

        # 4. ARMAZENAMENTO DOS RESULTADOS
        # --------------------------------

        df_orbit = pd.DataFrame(state_history_rk4, columns=['rx', 'ry', 'rz', 'vx', 'vy', 'vz'])
        df_orbit['time_step'] = time_series[:len(df_orbit)]  # Garante que o tamanho da série temporal é correto
        df_orbit['satellite_name'] = satellite.name
        df_orbit['satellite_id'] = satellite.model.satnum
        df_orbit['error_km'] = error_magnitude  # Adiciona o erro para cada ponto

        all_results.append(df_orbit)

    final_result_df = pd.concat(all_results, ignore_index=True)
    return final_result_df