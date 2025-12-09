import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

# --- CORREÇÃO AQUI: Importar load_tles_smart em vez de load_tles_from_url ---
from src.data_handler import load_tles_smart, satellites_to_dataframe, get_initial_state_and_time
from src.orbital_mechanics import get_derivatives 
from src.constants import RADIUS_EARTH

# --- CONFIGURAÇÕES DA DEMO ---
FILTER_NAME = 'STARLINK'
NUM_SATELLITES = 50       # Quantidade de satélites
DELTA_T = 30               # Passo de tempo físico (s)
SPEED_MULTIPLIER = 5       # Quantos passos físicos por frame visual

# 1. PREPARAÇÃO DOS DADOS
print("Carregando TLEs e preparando simulação...")

# --- CORREÇÃO AQUI: Usar a nova função com cache ---
ts, all_satellites = load_tles_smart(max_days=1.0)

# Verificação de segurança caso o download falhe
if all_satellites is None:
    print("Erro crítico: Não foi possível carregar os satélites.")
    sys.exit()

df = satellites_to_dataframe(all_satellites)

# Filtra
demo_sats = df[df['name'].str.contains(FILTER_NAME, case=False)].head(NUM_SATELLITES)
skyfield_objects = demo_sats['object'].tolist()

# 2. INICIALIZAÇÃO DO ESTADO
current_data = []
t_start_global = None

for i, sat in enumerate(skyfield_objects):
    t0, state_init = get_initial_state_and_time(sat, ts)
    
    if i == 0:
        t_start_global = t0 
        
    current_data.append({
        'state': state_init,      
        'sat_obj': sat,           
        't0': t0,                 
        'name': sat.name
    })

print(f"Pronto! Simulando {len(current_data)} objetos.")

# 3. CONFIGURAÇÃO DO GRÁFICO
plt.style.use('dark_background') 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
fig.canvas.manager.set_window_title(f'Painel de Controle - {FILTER_NAME}')

# Desenho da Terra 
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_earth = RADIUS_EARTH * np.outer(np.cos(u), np.sin(v))
y_earth = RADIUS_EARTH * np.outer(np.sin(u), np.sin(v))
z_earth = RADIUS_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_earth, y_earth, z_earth, color='cyan', alpha=0.1)

# Inicializa Scatter Plot
scatter_plot = ax.scatter([], [], [], c='lime', marker='.', s=30, label='Frota RK4')
target_scatter = ax.scatter([], [], [], c='red', marker='o', s=100, label='Alvo (Telemetry)')

# HUD
hud_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='white', 
                     family='monospace', fontsize=10, verticalalignment='top')

# Eixos
limit = RADIUS_EARTH + 2000
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title(f"Monitoramento em Tempo Real: {FILTER_NAME}", color='white')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.axis('off') 

elapsed_seconds = 0.0

# 4. FUNÇÃO DE ATUALIZAÇÃO
def update(frame):
    global elapsed_seconds
    
    xs, ys, zs = [], [], []
    
    for i in range(len(current_data)):
        state = current_data[i]['state']
        
        for _ in range(SPEED_MULTIPLIER):
            k1 = DELTA_T * get_derivatives(0, state)
            k2 = DELTA_T * get_derivatives(0, state + 0.5 * k1)
            k3 = DELTA_T * get_derivatives(0, state + 0.5 * k2)
            k4 = DELTA_T * get_derivatives(0, state + k3)
            state = state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        current_data[i]['state'] = state
        xs.append(state[0])
        ys.append(state[1])
        zs.append(state[2])

    elapsed_seconds += (DELTA_T * SPEED_MULTIPLIER)
    
    scatter_plot._offsets3d = (xs, ys, zs)
    target_scatter._offsets3d = ([xs[0]], [ys[0]], [zs[0]])
    
    # Cálculo de Erro para o Alvo
    target_sat = current_data[0]
    rk4_pos = target_sat['state'][0:3]
    
    t_current_skyfield = ts.tt_jd(target_sat['t0'].tt + elapsed_seconds / 86400.0)
    sgp4_pos = target_sat['sat_obj'].at(t_current_skyfield).position.km
    
    error_vec = rk4_pos - sgp4_pos
    error_km = np.linalg.norm(error_vec)
    
    log_msg = (
        f"SIMULATION TIME: +{elapsed_seconds/60:.1f} min\n"
        f"--------------------------------\n"
        f"TARGET: {target_sat['name']}\n"
        f"POS RK4 : [{rk4_pos[0]:.0f}, {rk4_pos[1]:.0f}, {rk4_pos[2]:.0f}] km\n"
        f"POS SGP4: [{sgp4_pos[0]:.0f}, {sgp4_pos[1]:.0f}, {sgp4_pos[2]:.0f}] km\n"
        f"--------------------------------\n"
        f"DELTA (ERROR): {error_km:.4f} km\n"
        f"STATUS: {'NOMINAL' if error_km < 5.0 else 'DRIFTING'}"
    )
    hud_text.set_text(log_msg)
    
    ax.view_init(elev=20, azim=frame * 0.1)
    
    return scatter_plot, target_scatter, hud_text

ani = FuncAnimation(fig, update, frames=range(100000), interval=10, blit=False)
plt.show()