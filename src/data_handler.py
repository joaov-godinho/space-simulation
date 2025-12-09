from pandas.io.sas.sas_constants import sas_release_length
from skyfield.api import EarthSatellite, load
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite

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

def load_tles_smart(url='https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', 
                    filename='active_satellites.txt', 
                    max_days=1.0):
    """
    Carrega TLEs de forma inteligente:
    - Se o arquivo local existir e for recente (menos de 'max_days'), usa ele.
    - Caso contrário, baixa da internet e atualiza o arquivo local.
    
    Args:
        url (str): URL do CelesTrak.
        filename (str): Nome do arquivo para salvar localmente (cache).
        max_days (float): Idade máxima do arquivo em dias antes de forçar atualização.
        
    Returns:
        tuple: (ts, satellites)
    """
    
    # Caminho completo para salvar na pasta 'data' (para organização)
    # Garante que a pasta data existe
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    filepath = os.path.join(data_dir, filename)
    
    download_needed = True
    
    # 1. Verifica se o arquivo existe
    if os.path.exists(filepath):
        # 2. Verifica a idade do arquivo
        file_mod_time = os.path.getmtime(filepath)
        file_age = datetime.now() - datetime.fromtimestamp(file_mod_time)
        
        if file_age < timedelta(days=max_days):
            print(f"Cache válido encontrado: '{filename}' (Idade: {file_age}). Usando arquivo local.")
            download_needed = False
        else:
            print(f"Cache expirado: '{filename}' (Idade: {file_age}). Baixando atualização...")
    else:
        print(f"Arquivo não encontrado locally. Baixando de {url}...")

    # 3. Carrega os dados (baixando ou lendo local)
    ts = load.timescale()
    
    try:
        # O Skyfield tem uma função 'reload' se passarmos o arquivo, 
        # mas aqui vamos controlar manualmente para garantir o comportamento.
        if download_needed:
            # Baixa e salva o arquivo usando a função interna do Skyfield ou requests
            # O load.tle_file aceita reload=True para forçar download
            satellites = load.tle_file(url, filename=filepath, reload=True)
        else:
            # Carrega direto do arquivo local sem tentar conectar
            satellites = load.tle_file(filepath, reload=False)
            
        print(f"Sucesso: {len(satellites)} satélites carregados.")
        return ts, satellites

    except Exception as e:
        print(f"Erro crítico ao carregar TLEs: {e}")
        # Fallback: Se falhar o download, tenta usar o local mesmo sendo velho (se existir)
        if download_needed and os.path.exists(filepath):
            print("Tentando usar backup local antigo...")
            return ts, load.tle_file(filepath, reload=False)
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