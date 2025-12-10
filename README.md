# 🛰️ Simulador de Tráfego Espacial (Space Traffic Simulation)

> **Desenvolvimento e Validação de um Propagador Orbital Numérico para Simulação de Tráfego Espacial** > *Trabalho de Conclusão de Curso (TCC) - Ciência da Computação - Unoesc 2025*

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Sobre o Projeto

Este projeto consiste em um **propagador orbital numérico de alta fidelidade** desenvolvido em Python. O objetivo principal é simular trajetórias de satélites em Órbita Baixa da Terra (LEO) com precisão, utilizando métodos de integração numérica e considerando perturbações físicas reais.

O sistema foi arquitetado para **processamento em lote (batch processing)**, permitindo a simulação de cenários de tráfego espacial com múltiplos objetos (ex: constelações Starlink) simultaneamente.

### 🔭 Destaques Técnicos
* **Motor Físico:** Integrador Runge-Kutta de 4ª Ordem (RK4) implementado *ad hoc*.
* **Alta Fidelidade:** Correção de trajetória com o termo de perturbação gravitacional **$J_2$** (achatamento da Terra).
* **Dados Reais:** Integração automática com o **CelesTrak** para download de TLEs (Two-Line Elements).
* **Big Data:** Arquitetura baseada em **Pandas** para filtragem e processamento de milhares de objetos.
* **Validação:** Comparação estatística contra o modelo padrão da indústria (**SGP4/Skyfield**).

---

## ⚙️ Arquitetura e Tecnologias

O projeto segue uma arquitetura modular desacoplada:

| Módulo | Função | Tecnologias |
| :--- | :--- | :--- |
| **Núcleo Físico** | Cálculo vetorial de forças (Gravidade + J2) e Integração Numérica | `NumPy`, `SciPy` |
| **Interface de Dados** | Download, Cache Inteligente e Parsing de TLEs | `Skyfield`, `Requests` |
| **Orquestrador** | Gerenciamento de simulação em lote e validação cruzada | `Pandas` |
| **Visualização** | Renderização 3D estática e animações em tempo real | `Matplotlib` |

---

## 🚀 Como Executar

### 1. Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado. Clone o repositório e instale as dependências:

```bash
git clone [https://github.com/joaov-godinho/space-simulation.git](https://github.com/joaov-godinho/space-simulation.git)
cd space-simulation
pip install numpy matplotlib pandas skyfield astropy
