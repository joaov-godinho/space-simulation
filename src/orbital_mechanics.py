"""
Módulo: orbital_mechanics.py
Descrição: Contém o núcleo físico-matemático do simulador.
Responsável por definir as equações de movimento (EDOs) e os algoritmos de
integração numérica.

Modelos Físicos:
1. Gravitação Universal (Problema de Dois Corpos).
2. Perturbação J2 (Oblatidade da Terra).

Método Numérico:
- Runge-Kutta de 4ª Ordem (RK4).
"""

import numpy as np
from .constants import GM_EARTH, RADIUS_EARTH, J2

# ==============================================================================
# 1. CÁLCULO DE FORÇAS (MODELO FÍSICO)
# ==============================================================================

def calculate_acceleration(r_vector):
    """
    Calcula o vetor aceleração total atuando sobre o satélite.
    Combina a gravidade central (Kepleriana) com a perturbação J2.
    
    Args:
        r_vector (np.array): Vetor posição [x, y, z] em km.
        
    Returns:
        np.array: Vetor aceleração total [ax, ay, az] em km/s^2.
    """
    # Magnitude do vetor posição (distância até o centro da Terra)
    r_norm = np.linalg.norm(r_vector)
    x, y, z = r_vector

    # --- TERMO 1: Aceleração Kepleriana (Dois Corpos) ---
    # a = - (GM / r^3) * vetor_r
    # Aponta sempre para o centro da Terra (0,0,0)
    a_kepler = -GM_EARTH / (r_norm ** 3) * r_vector

    # --- TERMO 2: Perturbação J2 (Achatamento da Terra) ---
    # O termo J2 corrige o fato de a Terra não ser esférica.
    # Ele introduz forças que causam a precessão da órbita.
    
    # Fator comum da equação do J2
    # k_j2 = 1.5 * J2 * GM * R_Terra^2 / r^5
    k_j2 = 1.5 * J2 * GM_EARTH * (RADIUS_EARTH ** 2) / (r_norm ** 5)

    # Termos auxiliares para simplificar as equações vetoriais
    z2 = z ** 2
    r2 = r_norm ** 2

    # Componentes vetoriais da perturbação J2
    # Estas fórmulas derivam do gradiente do potencial gravitacional com harmônicos zonais
    ax_j2 = k_j2 * x * (5 * z2 / r2 - 1)
    ay_j2 = k_j2 * y * (5 * z2 / r2 - 1)
    az_j2 = k_j2 * z * (5 * z2 / r2 - 3)

    a_j2 = np.array([ax_j2, ay_j2, az_j2])

    # Retorna a soma vetorial (Princípio da Superposição)
    # Aceleração Total = Gravidade Pura + Perturbação
    return a_kepler + a_j2


# ==============================================================================
# 2. DEFINIÇÃO DO SISTEMA DE EDOs
# ==============================================================================

def get_derivatives(t, state):
    """
    Função de derivadas do sistema (f(t, y)) para o integrador.
    Converte o estado atual em taxas de variação.
    
    Args:
        t (float): Tempo atual (não usado explicitamente pois a força é conservativa/autônoma).
        state (np.array): Vetor de estado completo [rx, ry, rz, vx, vy, vz].
        
    Returns:
        np.array: Derivadas [vx, vy, vz, ax, ay, az].
    """
    # Desempacota o estado em Posição e Velocidade
    r_vector = state[0:3]
    v_vector = state[3:6]

    # Calcula a aceleração baseada na posição atual
    a_vector = calculate_acceleration(r_vector)

    # Monta o vetor de derivadas (dy/dt)
    # A derivada da Posição é a Velocidade (v)
    # A derivada da Velocidade é a Aceleração (a)
    derivs = np.hstack((v_vector, a_vector))

    return derivs


# ==============================================================================
# 3. INTEGRADOR NUMÉRICO (RK4)
# ==============================================================================

def runge_kutta_4(state_initial, delta_t, num_steps):
    """
    Propagador Orbital utilizando o método Runge-Kutta de 4ª Ordem.
    
    Args:
        state_initial (np.array): Estado inicial [r, v].
        delta_t (float): Passo de tempo da integração (segundos).
        num_steps (int): Quantidade de passos a simular.
        
    Returns:
        tuple: (time_series, state_history)
    """
    num_variables = len(state_initial)
    
    # Pré-aloca matrizes para performance (evita redimensionamento em loop)
    state_history = np.zeros((num_steps + 1, num_variables))
    time_series = np.zeros(num_steps + 1)

    # Define condição inicial
    state_history[0, :] = state_initial
    state_current = state_initial.copy()

    # Loop principal da simulação
    for i in range(num_steps):
        t_current = time_series[i]

        # --- Passo 1: Inclinação no início do intervalo ---
        k1 = delta_t * get_derivatives(t_current, state_current)

        # --- Passo 2: Inclinação no ponto médio (usando k1) ---
        # Avança meio passo (0.5 * delta_t) usando a inclinação k1
        k2 = delta_t * get_derivatives(t_current + 0.5 * delta_t, state_current + 0.5 * k1)

        # --- Passo 3: Melhor estimativa no ponto médio (usando k2) ---
        k3 = delta_t * get_derivatives(t_current + 0.5 * delta_t, state_current + 0.5 * k2)

        # --- Passo 4: Inclinação no final do intervalo (usando k3) ---
        k4 = delta_t * get_derivatives(t_current + delta_t, state_current + k3)

        # --- Atualização Final (Média Ponderada) ---
        # k2 e k3 têm peso 2 porque representam o ponto médio (mais preciso)
        state_next = state_current + (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Atualiza variáveis para a próxima iteração
        state_current = state_next
        state_history[i + 1, :] = state_next
        time_series[i + 1] = t_current + delta_t
        
    return time_series, state_history