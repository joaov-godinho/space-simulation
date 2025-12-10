"""
Módulo: constants.py
Descrição: Centraliza as constantes físicas e geodésicas fundamentais utilizadas
no motor de propagação orbital.

Os valores adotados seguem padrões da astrodinâmica (como o modelo WGS84),
garantindo consistência com sistemas de referência reais.
"""

# --- PARÂMETROS GRAVITACIONAIS E GEODÉSICOS DA TERRA ---

# Parâmetro Gravitacional Padrão da Terra (mu = G * M)
# Unidade: km^3 / s^2
# Explicação: Representa o produto da Constante Gravitacional Universal (G) pela Massa da Terra (M).
# Em astrodinâmica, utiliza-se 'mu' diretamente pois este valor é conhecido com muito mais
# precisão experimental do que 'G' ou 'M' isoladamente.
GM_EARTH = 398600.4418

# Raio Equatorial da Terra
# Unidade: km
# Explicação: Distância do centro da Terra até a superfície no equador.
# Importância: A Terra não é uma esfera perfeita. Este valor é usado como referência
# para normalizar a distância nos cálculos de perturbação (J2).
RADIUS_EARTH = 6378.137

# Coeficiente de Perturbação J2 (Segundo Harmônico Zonal)
# Unidade: Adimensional
# Explicação: Fator que descreve o "achatamento" da Terra nos polos (oblateness).
# Efeito Físico: É a principal causa da precessão (rotação lenta) do plano da órbita
# e do argumento do perigeu em Órbita Baixa da Terra (LEO).
# Sem este valor, as órbitas simuladas não sofreriam deriva realista.
J2 = 1.08263e-3