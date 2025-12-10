"""
Módulo: data_handler.py
Descrição: Responsável pela interface de dados do simulador.
Este módulo gerencia o download, cache, parsing e conversão dos dados orbitais (TLEs).

Funcionalidades principais:
1. Gerenciamento de Cache: Evita downloads repetitivos do CelesTrak.
2. Filtragem: Converte dados para Pandas para permitir consultas (ex: "STARLINK").
3. Tradução Física: Converte elementos orbitais abstratos (TLE) em vetores cartesianos (r, v).
"""

from skyfield.api import load, EarthSatellite
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def satellites_to_dataframe(satellites):
    """
    Converte uma lista de objetos EarthSatellite (Skyfield) em um DataFrame do Pandas.
    
    Motivo: Listas nativas do Python são lentas para buscas de texto. 
    O Pandas permite filtrar milhares de satélites por nome em milissegundos.
    """
    data = []
    for sat in satellites:
        data.append({
            'name': sat.name,                 # Nome do Satélite (ex: STARLINK-1007)
            'catalog_number': sat.model.satnum, # ID NORAD único
            'epoch': sat.epoch.utc_datetime(),  # Data de referência do TLE
            'object': sat                     # O objeto Skyfield real (guardado para cálculos físicos)
        })
    
    return pd.DataFrame(data)

def load_tles_smart(url='https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle', 
                    filename='active_satellites.txt', 
                    max_days=1.0):
    """
    Carrega TLEs com sistema de Cache Inteligente.
    
    Lógica:
    1. Verifica se o arquivo já existe localmente na pasta 'data/'.
    2. Se existir, verifica a data de modificação.
    3. Se o arquivo for recente (menos de 'max_days'), carrega do disco (rápido).
    4. Se for antigo ou não existir, baixa do CelesTrak (lento, mas atualizado).
    
    Args:
        url (str): Endereço do CelesTrak.
        filename (str): Nome do arquivo local.
        max_days (float): Idade máxima do arquivo em dias antes de forçar atualização.
        
    Returns:
        tuple: (ts, satellites) -> Escala de tempo e lista de objetos.
    """
    
    # Define o caminho para a pasta 'data' na raiz do projeto
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    filepath = os.path.join(data_dir, filename)
    
    download_needed = True
    
    # 1. Verificação de existência e idade do arquivo
    if os.path.exists(filepath):
        file_mod_time = os.path.getmtime(filepath)
        file_age = datetime.now() - datetime.fromtimestamp(file_mod_time)
        
        # Se o arquivo é novo o suficiente, cancela o download
        if file_age < timedelta(days=max_days):
            print(f"Cache válido encontrado: '{filename}' (Idade: {file_age}). Usando arquivo local.")
            download_needed = False
        else:
            print(f"Cache expirado: '{filename}' (Idade: {file_age}). Baixando atualização...")
    else:
        print(f"Arquivo não encontrado localmente. Baixando de {url}...")

    # Carrega a escala de tempo astronômica (necessária para o Skyfield)
    ts = load.timescale()
    
    try:
        if download_needed:
            # reload=True força o Skyfield a baixar da URL e sobrescrever o arquivo
            satellites = load.tle_file(url, filename=filepath, reload=True)
        else:
            # reload=False força o uso do arquivo local, sem tentar conectar à internet
            satellites = load.tle_file(filepath, reload=False)
            
        print(f"Sucesso: {len(satellites)} satélites carregados.")
        return ts, satellites

    except Exception as e:
        print(f"Erro crítico ao carregar TLEs: {e}")
        # Mecanismo de Fallback (Segurança): 
        # Se a internet cair na apresentação, tenta usar o arquivo local mesmo que seja velho.
        if download_needed and os.path.exists(filepath):
            print("AVISO: Falha no download. Usando backup local antigo para prosseguir.")
            return ts, load.tle_file(filepath, reload=False)
        return None, None
    
def get_initial_state_and_time(satellite, ts, time_offset_seconds=0):
    """
    Extrai o Vetor de Estado Cartesiano (r, v) de um objeto TLE.
    
    Esta função atua como um 'Tradutor':
    - Entrada: Objeto TLE (elementos orbitais médios, abstratos).
    - Saída: Vetores numéricos [x, y, z, vx, vy, vz] (física pura).
    
    É aqui que o modelo SGP4 (do Skyfield) é chamado para gerar a condição inicial
    para o nosso integrador numérico RK4.
    """
    
    # 1. Define o tempo inicial (t0) como o 'Epoch' (momento exato) do TLE
    t0_epoch_skyfield = satellite.epoch

    # Aplica um deslocamento de tempo se solicitado (útil para sincronizar simulações)
    if time_offset_seconds != 0:
        t0 = ts.tt(jd=t0_epoch_skyfield.tt + time_offset_seconds / (24*3600))
    else:
        t0 = t0_epoch_skyfield

    # 2. Solicita ao Skyfield (SGP4) a posição/velocidade neste instante exato
    # O retorno é um objeto Geocentric (coordenadas centradas na Terra)
    geocentric_position = satellite.at(t0)

    # 3. Extração dos vetores puros
    # .position.km -> Array NumPy [x, y, z] em quilômetros
    r_vector = geocentric_position.position.km
    
    # .velocity.km_per_s -> Array NumPy [vx, vy, vz] em km/s
    v_vector = geocentric_position.velocity.km_per_s

    # 4. Unificação no Vetor de Estado (State Vector) de 6 elementos
    state_initial = np.hstack((r_vector, v_vector))

    return t0, state_initial