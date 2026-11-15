"""
Estudio Numérico del Péndulo Doble
Etapa B: Diagramas de fase y comparación de condiciones iniciales.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pendulo_doble import derivadas_pendulo


def simular_pendulo(estado_inicial, t_span, t_eval, parametros):
    """
    Función auxiliar para integrar una trayectoria dados el estado inicial,
    los tiempos y los parámetros físicos.
    """
    solucion = solve_ivp(
        fun=derivadas_pendulo,
        t_span=t_span,
        y0=estado_inicial,
        t_eval=t_eval,
        method='RK45',
        args=parametros,
        rtol=1e-9,
        atol=1e-9
    )
    return solucion


def main():
    # -------------------------------------------------------------
    # 1. Parámetros físicos del sistema (idénticos a la Etapa A)
    # -------------------------------------------------------------
    m1 = 1.0    # Masa 1 (kg)
    m2 = 1.0    # Masa 2 (kg)
    L1 = 1.0    # Longitud de la barra 1 (m)
    L2 = 1.0    # Longitud de la barra 2 (m)
    g  = 9.81   # Aceleración gravitatoria (m/s^2)
    parametros = (m1, m2, L1, L2, g)

    # Intervalo de integración
    t_inicio = 0.0
    t_fin = 30.0            # 30 segundos para ver bien el desarrollo de la órbita
    cantidad_puntos = 4000  # Buena resolución para curvas continuas
    t_span = (t_inicio, t_fin)
    t_eval = np.linspace(t_inicio, t_fin, cantidad_puntos)

    # -------------------------------------------------------------
    # 2. Definición de 3 condiciones iniciales distintas
    # -------------------------------------------------------------
    casos = [
        {
            "nombre": "Amplitud pequeña (Régimen cuasi-lineal)",
            "subtitulo": r"$\theta_1(0)=0.1$ rad, $\theta_2(0)=0.1$ rad",
            "y0": [0.1, 0.0, 0.1, 0.0],
            "color": "tab:green"
        },
        {
            "nombre": "Amplitud moderada (Régimen no lineal regular)",
            "subtitulo": r"$\theta_1(0)=0.7$ rad, $\theta_2(0)=0.0$ rad",
            "y0": [0.7, 0.0, 0.0, 0.0],
            "color": "tab:blue"
        },
        {
            "nombre": "Gran amplitud (Régimen caótico)",
            "subtitulo": r"$\theta_1(0)=\pi/2$ rad, $\theta_2(0)=\pi/2$ rad",
            "y0": [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
            "color": "tab:red"
        }
    ]

    # -------------------------------------------------------------
    # 3. Simulación y graficación
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)

    for i, caso in enumerate(casos):
        print(f"Simulando: {caso['nombre']}...")
        sol = simular_pendulo(caso["y0"], t_span, t_eval, parametros)

        theta1 = sol.y[0]
        omega1 = sol.y[1]
        theta2 = sol.y[2]
        omega2 = sol.y[3]

        # Fila 0: Espacio de fases del Péndulo 1 (theta1 vs omega1)
        ax_top = axes[0, i]
        ax_top.plot(theta1, omega1, color=caso["color"], linewidth=0.8, alpha=0.85)
        ax_top.plot(caso["y0"][0], caso["y0"][1], 'ko', markersize=4, label='Inicio')
        ax_top.set_title(f"{caso['nombre']}\n{caso['subtitulo']}", fontsize=11)
        ax_top.set_xlabel(r'$\theta_1$ (rad)', fontsize=10)
        ax_top.set_ylabel(r'$\omega_1$ (rad/s)', fontsize=10)
        ax_top.grid(True, linestyle='--', alpha=0.5)
        ax_top.legend(loc='upper right', fontsize=8)

        # Fila 1: Espacio de fases del Péndulo 2 (theta2 vs omega2)
        ax_bottom = axes[1, i]
        ax_bottom.plot(theta2, omega2, color=caso["color"], linewidth=0.8, alpha=0.85)
        ax_bottom.plot(caso["y0"][2], caso["y0"][3], 'ko', markersize=4, label='Inicio')
        ax_bottom.set_xlabel(r'$\theta_2$ (rad)', fontsize=10)
        ax_bottom.set_ylabel(r'$\omega_2$ (rad/s)', fontsize=10)
        ax_bottom.grid(True, linestyle='--', alpha=0.5)
        ax_bottom.legend(loc='upper right', fontsize=8)

    plt.suptitle("Etapa B: Proyecciones en el Espacio de Fases para Distintas Condiciones Iniciales", fontsize=14, y=1.00)
    plt.tight_layout()

    # Guardar gráfico
    nombre_grafico = "etapa_b_diagramas_fase.png"
    plt.savefig(nombre_grafico, dpi=200, bbox_inches='tight')
    print(f"Simulaciones completadas. Gráfico guardado como: {nombre_grafico}")
    plt.show()


if __name__ == '__main__':
    main()
