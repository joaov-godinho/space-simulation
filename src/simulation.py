"""
Módulo: simulation.py
Descrição: Orquestrador principal das simulações de tráfego espacial.

Este módulo é responsável por:
1. Gerenciar a execução em lote (batch) de múltiplos satélites.
2. Executar o 'Motor Físico' (RK4) para cada objeto.
3. Executar o 'Modelo de Referência' (SGP4/Skyfield) para fins de validação.
4. Calcular métricas de erro e consolidar os resultados em estruturas de dados (Pandas)
   para análise e visualização posterior.
"""

from src.data_handler import get_initial_state_and_time
from src.orbital_mechanics import runge_kutta_4
import numpy as np
import pandas as pd
from skyfield.api import wgs84

# --- CONSTANTES DE CONFIGURAÇÃO DA SIMULAÇÃO ---
# Passo de tempo (Delta T): Intervalo entre cada cálculo do integrador.
# 60s é um bom equilíbrio entre precisão e performance para LEO.
DELTA_T_SECONDS = 60

# Duração total da simulação.
DURATION_HOURS = 1

# Número total de passos (iterações) que o loop do RK4 executará.
NUM_STEPS = int(DURATION_HOURS * 3600 / DELTA_T_SECONDS)


def run_multi_object_simulation_and_validate(ts, list_of_satellites):
    """
    Executa a simulação e validação para uma lista de objetos espaciais.
    
    Lógica do Algoritmo:
    Para cada satélite na lista:
      1. Obtém o estado inicial (posição/velocidade) do TLE.
      2. Propaga a órbita usando nosso modelo numérico (RK4 + J2).
      3. Propaga a órbita usando o modelo analítico padrão (SGP4 via Skyfield).
      4. Compara os resultados finais para calcular o erro de precisão.
      5. Armazena a trajetória completa e metadados.
    
    Args:
        ts (Timescale): Objeto de tempo do Skyfield.
        list_of_satellites (list): Lista de objetos EarthSatellite (já filtrados).
        
    Returns:
        pd.DataFrame: DataFrame único contendo as trajetórias e erros de todos os satélites.
    """

    all_results = []

    # Cria o array de tempo simulado [0, 60, 120, ..., 3600]
    time_series = np.arange(0, DURATION_HOURS * 3600 + DELTA_T_SECONDS, DELTA_T_SECONDS)

    # Loop de Iteração sobre o "Tráfego" (Lista de Satélites)
    for i, satellite in enumerate(list_of_satellites):
        print(f"  -> Propagando e Validando {satellite.name} ({i + 1}/{len(list_of_satellites)})")

        # ---------------------------------------------------------
        # 1. DEFINIÇÃO DAS CONDIÇÕES INICIAIS
        # ---------------------------------------------------------
        # Converte o TLE (elementos orbitais) em vetores cartesianos [r, v]
        # t0 é o instante exato (Epoch) onde a simulação começa para este satélite.
        t0, state_initial = get_initial_state_and_time(satellite, ts)

        # ---------------------------------------------------------
        # 2. PROCESSO DE PROPAGAÇÃO (MOTOR FÍSICO)
        # ---------------------------------------------------------
        
        # 2.1. Simulação Numérica (Nosso Modelo)
        # Chama o integrador RK4 implementado em orbital_mechanics.py
        # Retorna o histórico completo de posições e velocidades.
        _, state_history_rk4 = runge_kutta_4(
            state_initial, DELTA_T_SECONDS, NUM_STEPS
        )

        # 2.2. Simulação de Referência (Benchmark SGP4)
        # Preparamos arrays para guardar a "verdade" do Skyfield
        r_skyfield_history = np.zeros((NUM_STEPS + 1, 3))
        
        # Gera os tempos exatos para consultar o SGP4
        t_skyfield_series = ts.tt_jd(t0.tt + time_series / (24 * 3600))

        # Consulta o Skyfield para cada passo de tempo
        for j, t_skyfield in enumerate(t_skyfield_series):
            geocentric_position = satellite.at(t_skyfield)
            r_vector = geocentric_position.position.km
            r_skyfield_history[j, :] = r_vector

        # ---------------------------------------------------------
        # 3. ANÁLISE DE VALIDAÇÃO (CÁLCULO DE ERRO)
        # ---------------------------------------------------------
        
        # Pega a última posição calculada pelo nosso RK4
        final_r_rk4 = state_history_rk4[-1, 0:3]
        
        # Pega a última posição calculada pelo Skyfield (Referência)
        final_r_skyfield = r_skyfield_history[-1, :]

        # Calcula a distância Euclidiana (magnitude do vetor diferença)
        # Isso quantifica o quanto nosso modelo desviou do padrão ouro.
        difference_vector = final_r_rk4 - final_r_skyfield
        error_magnitude = np.linalg.norm(difference_vector)  # Erro em km

        # ---------------------------------------------------------
        # 4. ESTRUTURAÇÃO E ARMAZENAMENTO (BIG DATA)
        # ---------------------------------------------------------
        
        # Converte a matriz NumPy do RK4 em um DataFrame Pandas
        df_orbit = pd.DataFrame(state_history_rk4, columns=['rx', 'ry', 'rz', 'vx', 'vy', 'vz'])
        
        # Enriquece os dados com metadados e resultados da validação
        df_orbit['time_step'] = time_series[:len(df_orbit)] 
        df_orbit['satellite_name'] = satellite.name
        df_orbit['satellite_id'] = satellite.model.satnum
        
        # Armazena o erro calculado (igual para todas as linhas deste satélite)
        # Útil para filtrar depois: "Mostre apenas satélites com erro > 1km"
        df_orbit['error_km'] = error_magnitude 

        all_results.append(df_orbit)

    # Consolidação: Junta todas as trajetórias individuais em uma única tabela mestre
    final_result_df = pd.concat(all_results, ignore_index=True)
    
    return final_result_df