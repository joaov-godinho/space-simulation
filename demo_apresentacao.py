import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURAÇÃO DE AMBIENTE ---
# Adiciona o diretório atual ao path do Python.
# Isso é crucial para que o script consiga encontrar a pasta 'src'
# independentemente de onde o comando python for executado.
sys.path.append(os.getcwd())

# Importações dos módulos desenvolvidos no TCC
# load_tles_smart: Carrega dados com sistema de cache (evita downloads repetidos)
# satellites_to_dataframe: Converte objetos Skyfield para Pandas para filtragem rápida
# get_initial_state_and_time: Traduz TLEs para vetores cartesianos (r, v)
from src.data_handler import load_tles_smart, satellites_to_dataframe, get_initial_state_and_time

# get_derivatives: A função que contém a FÍSICA (Gravidade + J2)
from src.orbital_mechanics import get_derivatives 
from src.constants import RADIUS_EARTH

# --- CONFIGURAÇÕES DA DEMO (PARÂMETROS GLOBAIS) ---
FILTER_NAME = 'STARLINK'   # Nome da constelação a ser filtrada
NUM_SATELLITES = 50        # Limite de objetos para manter a fluidez visual
DELTA_T = 30               # Passo de tempo da simulação (dt) em segundos
SPEED_MULTIPLIER = 5       # Aceleração visual: quantos cálculos de física ocorrem por frame de vídeo.
                           # 5x significa que cada frame da animação avança 5 * 30s = 150s no tempo simulado.

# ==============================================================================
# 1. PREPARAÇÃO E AQUISIÇÃO DE DADOS
# ==============================================================================
print("Carregando TLEs e preparando simulação...")

# Carrega os dados orbitais. O parâmetro max_days=1.0 garante que usaremos
# um arquivo local se ele tiver menos de 24h, economizando banda e tempo.
ts, all_satellites = load_tles_smart(max_days=1.0)

# Verificação de segurança (Fail-safe)
if all_satellites is None:
    print("Erro crítico: Não foi possível carregar os satélites.")
    sys.exit()

# Converte a lista bruta para DataFrame para permitir consultas complexas
df = satellites_to_dataframe(all_satellites)

# Aplica o filtro de string para selecionar apenas os satélites desejados (ex: STARLINK)
demo_sats = df[df['name'].str.contains(FILTER_NAME, case=False)].head(NUM_SATELLITES)

# Extrai a lista de objetos 'EarthSatellite' do Skyfield para processamento
skyfield_objects = demo_sats['object'].tolist()

# ==============================================================================
# 2. INICIALIZAÇÃO DO VETOR DE ESTADO
# ==============================================================================
current_data = []
t_start_global = None

# Itera sobre os objetos selecionados para preparar o estado inicial (t0)
for i, sat in enumerate(skyfield_objects):
    # Obtém o tempo de época (t0) e o vetor de estado inicial [rx, ry, rz, vx, vy, vz]
    t0, state_init = get_initial_state_and_time(sat, ts)
    
    # Define o tempo global da simulação baseado no primeiro satélite da lista
    if i == 0:
        t_start_global = t0 
        
    # Armazena tudo em um dicionário para fácil acesso durante o loop
    current_data.append({
        'state': state_init,      # O vetor que será modificado pelo nosso integrador RK4
        'sat_obj': sat,           # O objeto original (usado para calcular a referência/verdade terrestre)
        't0': t0,                 # O tempo original do TLE (para sincronizar a validação)
        'name': sat.name
    })

print(f"Pronto! Simulando {len(current_data)} objetos.")

# ==============================================================================
# 3. CONFIGURAÇÃO DA VISUALIZAÇÃO (MATPLOTLIB)
# ==============================================================================
plt.style.use('dark_background') # Estilo visual "Sci-Fi" para melhor contraste na apresentação
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
fig.canvas.manager.set_window_title(f'Painel de Controle - {FILTER_NAME}')

# --- Construção da Terra (Wireframe) ---
# Cria uma esfera matemática representando a Terra para referência visual
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_earth = RADIUS_EARTH * np.outer(np.cos(u), np.sin(v))
y_earth = RADIUS_EARTH * np.outer(np.sin(u), np.sin(v))
z_earth = RADIUS_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
# Plota a Terra em ciano translúcido
ax.plot_wireframe(x_earth, y_earth, z_earth, color='cyan', alpha=0.1)

# --- Inicialização dos Objetos Gráficos ---
# 'scatter_plot': Representa a frota geral (pontos verdes)
scatter_plot = ax.scatter([], [], [], c='lime', marker='.', s=30, label='Frota RK4')
# 'target_scatter': Representa o satélite alvo da telemetria (ponto vermelho maior)
target_scatter = ax.scatter([], [], [], c='red', marker='o', s=100, label='Alvo (Telemetry)')

# --- HUD (Heads-Up Display) ---
# Elemento de texto 2D fixo na tela para mostrar dados em tempo real
hud_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='white', 
                     family='monospace', fontsize=10, verticalalignment='top')

# Configuração dos Limites e Labels dos Eixos
limit = RADIUS_EARTH + 2000
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title(f"Monitoramento em Tempo Real: {FILTER_NAME}", color='white')

# Limpeza visual (remove grades e planos de fundo padrão)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.axis('off') 

# Variável global para rastrear o tempo decorrido desde o início da simulação
elapsed_seconds = 0.0

# ==============================================================================
# 4. LOOP DE ATUALIZAÇÃO (CORE DA SIMULAÇÃO)
# ==============================================================================
def update(frame):
    global elapsed_seconds
    
    # Listas temporárias para as coordenadas X, Y, Z de todos os satélites neste frame
    xs, ys, zs = [], [], []
    
    # --- Atualização Física (Motor RK4) ---
    for i in range(len(current_data)):
        state = current_data[i]['state']
        
        # Sub-loop: Realiza múltiplos passos físicos para cada frame visual
        # Isso permite que a animação seja rápida sem perder a precisão do passo pequeno (Delta T)
        for _ in range(SPEED_MULTIPLIER):
            # Implementação manual do Runge-Kutta 4 (RK4) para performance
            # Calcula 4 inclinações (k1, k2, k3, k4) baseadas na função get_derivatives (Física)
            k1 = DELTA_T * get_derivatives(0, state)
            k2 = DELTA_T * get_derivatives(0, state + 0.5 * k1)
            k3 = DELTA_T * get_derivatives(0, state + 0.5 * k2)
            k4 = DELTA_T * get_derivatives(0, state + k3)
            
            # Atualiza o estado com a média ponderada das inclinações
            state = state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        # Salva o novo estado atualizado
        current_data[i]['state'] = state
        
        # Separa as coordenadas para o gráfico
        xs.append(state[0])
        ys.append(state[1])
        zs.append(state[2])

    # Incrementa o relógio da simulação
    elapsed_seconds += (DELTA_T * SPEED_MULTIPLIER)
    
    # --- Atualização Gráfica ---
    # Atualiza a posição de todos os satélites verdes
    scatter_plot._offsets3d = (xs, ys, zs)
    # Atualiza a posição do satélite alvo (índice 0) vermelho
    target_scatter._offsets3d = ([xs[0]], [ys[0]], [zs[0]])
    
    # --- CÁLCULO DE VALIDAÇÃO (RK4 vs SGP4) ---
    # Seleciona o primeiro satélite para análise detalhada
    target_sat = current_data[0]
    
    # 1. Posição atual segundo nosso modelo (RK4)
    rk4_pos = target_sat['state'][0:3]
    
    # 2. Posição atual segundo o modelo de referência (SGP4 via Skyfield)
    # Calculamos o tempo astronômico exato atual (Tempo Inicial + Segundos Decorridos)
    t_current_skyfield = ts.tt_jd(target_sat['t0'].tt + elapsed_seconds / 86400.0)
    # Pedimos ao Skyfield a posição "verdadeira"
    sgp4_pos = target_sat['sat_obj'].at(t_current_skyfield).position.km
    
    # 3. Cálculo do vetor de erro e sua magnitude (distância)
    error_vec = rk4_pos - sgp4_pos
    error_km = np.linalg.norm(error_vec)
    
    # --- Atualização do HUD (Display de Texto) ---
    log_msg = (
        f"SIMULATION TIME: +{elapsed_seconds/60:.1f} min\n" # Tempo decorrido
        f"--------------------------------\n"
        f"TARGET: {target_sat['name']}\n"
        f"POS RK4 : [{rk4_pos[0]:.0f}, {rk4_pos[1]:.0f}, {rk4_pos[2]:.0f}] km\n"
        f"POS SGP4: [{sgp4_pos[0]:.0f}, {sgp4_pos[1]:.0f}, {sgp4_pos[2]:.0f}] km\n"
        f"--------------------------------\n"
        f"DELTA (ERROR): {error_km:.4f} km\n" # O erro acumulado
        # Status simples: Se o erro for grande, indica deriva (falta de arrasto, etc)
        f"STATUS: {'NOMINAL' if error_km < 5.0 else 'DRIFTING'}" 
    )
    hud_text.set_text(log_msg)
    
    # Efeito cinematográfico: Gira a câmera lentamente
    ax.view_init(elev=20, azim=frame * 0.1)
    
    return scatter_plot, target_scatter, hud_text

# Inicia a animação
# interval=10 tenta manter ~100 FPS se o processamento permitir
ani = FuncAnimation(fig, update, frames=range(100000), interval=10, blit=False)

# Exibe a janela
plt.show()