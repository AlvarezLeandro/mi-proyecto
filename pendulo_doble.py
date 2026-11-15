"""
Estudio Numérico del Péndulo Doble Ideal
Proyecto Universitario - Computación y Sistemas Dinámicos

Este programa implementa 5 análisis simplificados y directos:
1. Ángulos vs tiempo para 4 péndulos dobles (2 muy cercanos y 2 con diferencias mayores).
2. Energía mecánica total en función del tiempo para los 4 péndulos (verificando conservación).
3. Diagramas de fase (espacio de fases) para cada uno de los 4 péndulos.
4. Exponente de Lyapunov para la primera pareja (P1 y P2) y para la segunda pareja (P3 y P4).
5. Mapa de calor del exponente de Lyapunov con grilla fina (barrido 2D).
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# FUNCIONES FÍSICAS FUNDAMENTALES
# =============================================================================

def derivadas_pendulo(t, estado, m1, m2, L1, L2, g):
    """
    Calcula las derivadas del vector de estado del péndulo doble.
    estado = [theta1, omega1, theta2, omega2]
    retorna: [omega1, alpha1, omega2, alpha2]
    """
    theta1, omega1, theta2, omega2 = estado
    delta_theta = theta1 - theta2

    cos_d = np.cos(delta_theta)
    sin_d = np.sin(delta_theta)

    # Matriz de inercia M (2x2)
    M11 = (m1 + m2) * (L1 ** 2)
    M12 = m2 * L1 * L2 * cos_d
    M21 = M12
    M22 = m2 * (L2 ** 2)

    # Vector independiente b
    b1 = -m2 * L1 * L2 * (omega2 ** 2) * sin_d - (m1 + m2) * g * L1 * np.sin(theta1)
    b2 =  m2 * L1 * L2 * (omega1 ** 2) * sin_d - m2 * g * L2 * np.sin(theta2)

    # Resolución exacta por Regla de Cramer (rápida y exacta)
    det_M = M11 * M22 - M12 * M21
    alpha1 = (b1 * M22 - b2 * M12) / det_M
    alpha2 = (M11 * b2 - M21 * b1) / det_M

    return [omega1, alpha1, omega2, alpha2]


def calcular_energia(theta1, omega1, theta2, omega2, m1, m2, L1, L2, g):
    """
    Calcula la energía mecánica total E = T + V en Joules.
    Referencia V=0 en reposo inferior para asegurar valores positivos claros.
    """
    delta_theta = theta1 - theta2

    # Cinética
    T = (0.5 * (m1 + m2) * (L1 ** 2) * (omega1 ** 2)
         + 0.5 * m2 * (L2 ** 2) * (omega2 ** 2)
         + m2 * L1 * L2 * omega1 * omega2 * np.cos(delta_theta))

    # Potencial gravitatoria
    V = ((m1 + m2) * g * L1 * (1.0 - np.cos(theta1))
         + m2 * g * L2 * (1.0 - np.cos(theta2)))

    return T + V


def calcular_lyapunov_benettin(estado_inicial, parametros, d0=1e-6, tau=0.5, n_pasos=70):
    """
    Estima el exponente máximo de Lyapunov usando el método de Benettin.
    """
    s_ref = np.array(estado_inicial, dtype=float)
    s_pert = s_ref.copy()
    s_pert[0] += d0

    suma_log = 0.0
    tiempos = []
    lyap_acumulado = []

    for k in range(1, n_pasos + 1):
        sol_ref = solve_ivp(derivadas_pendulo, (0.0, tau), s_ref, args=parametros, method='RK45', rtol=1e-6, atol=1e-6)
        sol_pert = solve_ivp(derivadas_pendulo, (0.0, tau), s_pert, args=parametros, method='RK45', rtol=1e-6, atol=1e-6)

        s_ref = sol_ref.y[:, -1]
        s_pert = sol_pert.y[:, -1]

        delta = s_pert - s_ref
        d1 = np.linalg.norm(delta)

        suma_log += np.log(d1 / d0)
        t_actual = k * tau
        tiempos.append(t_actual)
        lyap_acumulado.append(suma_log / t_actual)

        # Renormalización a distancia d0
        s_pert = s_ref + d0 * (delta / d1)

    return np.array(tiempos), np.array(lyap_acumulado)


# =============================================================================
# ANÁLISIS 1: ÁNGULOS EN FUNCIÓN DEL TIEMPO (4 PÉNDULOS)
# =============================================================================

def figura_1_angulos(soluciones, pendulos):
    print("  [1/5] Generando Figura 1: Ángulos vs Tiempo para los 4 péndulos...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    t = soluciones[0].t

    for sol, p in zip(soluciones, pendulos):
        th1 = sol.y[0]
        th2 = sol.y[2]
        ax1.plot(t, th1, label=f"{p['nombre']} ({p['desc']})", color=p['color'], linestyle=p['estilo'], linewidth=1.4)
        ax2.plot(t, th2, label=f"{p['nombre']}", color=p['color'], linestyle=p['estilo'], linewidth=1.4)

    ax1.set_title('Figura 1: Evolución Temporal de los Ángulos (4 Péndulos Dobles)', fontsize=13)
    ax1.set_ylabel(r'Ángulo $\theta_1(t)$ (rad)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    ax2.set_xlabel('Tiempo $t$ (s)', fontsize=11)
    ax2.set_ylabel(r'Ángulo $\theta_2(t)$ (rad)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    nombre = 'figura_1_angulos_vs_tiempo.png'
    plt.savefig(nombre, dpi=200)
    print(f"        Guardado: {nombre}")
    plt.show(block=False)


# =============================================================================
# ANÁLISIS 2: ENERGÍA MECÁNICA EN FUNCIÓN DEL TIEMPO (4 PÉNDULOS)
# =============================================================================

def figura_2_energia(soluciones, pendulos, parametros):
    print("  [2/5] Generando Figura 2: Energía Mecánica vs Tiempo...")
    plt.figure(figsize=(10, 6))
    t = soluciones[0].t
    m1, m2, L1, L2, g = parametros

    for sol, p in zip(soluciones, pendulos):
        th1, w1 = sol.y[0], sol.y[1]
        th2, w2 = sol.y[2], sol.y[3]
        E = calcular_energia(th1, w1, th2, w2, m1, m2, L1, L2, g)
        E_media = np.mean(E)
        plt.plot(t, E, label=f"{p['nombre']} (E = {E_media:.2f} J)",
                 color=p['color'], linestyle=p['estilo'], linewidth=1.6)

    plt.title('Figura 2: Energía Mecánica Total en Función del Tiempo (Conservación)', fontsize=13)
    plt.xlabel('Tiempo $t$ (s)', fontsize=11)
    plt.ylabel('Energía Mecánica Total $E$ (Joules)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='center right')
    plt.tight_layout()

    nombre = 'figura_2_energia_vs_tiempo.png'
    plt.savefig(nombre, dpi=200)
    print(f"        Guardado: {nombre}")
    plt.show(block=False)


# =============================================================================
# ANÁLISIS 3: DIAGRAMAS DE FASE PARA CADA UNO DE LOS 4 PÉNDULOS
# =============================================================================

def figura_3_fase(soluciones, pendulos):
    print("  [3/5] Generando Figura 3: Diagramas de fase para los 4 péndulos...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i, (sol, p) in enumerate(zip(soluciones, pendulos)):
        th1, w1 = sol.y[0], sol.y[1]
        th2, w2 = sol.y[2], sol.y[3]

        # Fila superior: Eslabón 1 (theta1 vs omega1)
        ax_top = axes[0, i]
        ax_top.plot(th1, w1, color=p['color'], linewidth=0.7, alpha=0.85)
        ax_top.plot(th1[0], w1[0], 'ko', markersize=4)
        ax_top.set_title(f"{p['nombre']}\n{p['desc']}", fontsize=10)
        ax_top.set_xlabel(r'$\theta_1$ (rad)', fontsize=9)
        ax_top.set_ylabel(r'$\omega_1$ (rad/s)', fontsize=9)
        ax_top.grid(True, linestyle='--', alpha=0.4)

        # Fila inferior: Eslabón 2 (theta2 vs omega2)
        ax_bot = axes[1, i]
        ax_bot.plot(th2, w2, color=p['color'], linewidth=0.7, alpha=0.85)
        ax_bot.plot(th2[0], w2[0], 'ko', markersize=4)
        ax_bot.set_xlabel(r'$\theta_2$ (rad)', fontsize=9)
        ax_bot.set_ylabel(r'$\omega_2$ (rad/s)', fontsize=9)
        ax_bot.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Figura 3: Diagramas de Fase para los 4 Péndulos Dobles', fontsize=13, y=1.00)
    plt.tight_layout()

    nombre = 'figura_3_diagramas_fase.png'
    plt.savefig(nombre, dpi=200, bbox_inches='tight')
    print(f"        Guardado: {nombre}")
    plt.show(block=False)


# =============================================================================
# ANÁLISIS 4: EXPONENTES DE LYAPUNOV (PAREJA 1 Y PAREJA 2)
# =============================================================================

def figura_4_lyapunov(pendulos, parametros):
    print("  [4/5] Generando Figura 4: Exponentes de Lyapunov para ambas parejas...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pareja 1: Péndulo 1 y Péndulo 2 (Condiciones muy similares)
    print("        Calculando Lyapunov para Péndulo 1 y 2...")
    t1, lyap1 = calcular_lyapunov_benettin(pendulos[0]['y0'], parametros)
    t2, lyap2 = calcular_lyapunov_benettin(pendulos[1]['y0'], parametros)

    ax1.plot(t1, lyap1, color=pendulos[0]['color'], linewidth=1.6, label=f"P1 (Final: {lyap1[-1]:.2f} 1/s)")
    ax1.plot(t2, lyap2, '--', color=pendulos[1]['color'], linewidth=1.4, label=f"P2 (Final: {lyap2[-1]:.2f} 1/s)")
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax1.set_title('Pareja 1: Condiciones Muy Similares\n' + rf"$\Delta\theta_1(0) = 10^{{-4}}$ rad", fontsize=11)
    ax1.set_xlabel('Tiempo $t$ (s)', fontsize=10)
    ax1.set_ylabel(r'Exponente $\lambda(t)$ (s$^{-1}$)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # Pareja 2: Péndulo 3 y Péndulo 4 (Diferencias más grandes)
    print("        Calculando Lyapunov para Péndulo 3 y 4...")
    t3, lyap3 = calcular_lyapunov_benettin(pendulos[2]['y0'], parametros)
    t4, lyap4 = calcular_lyapunov_benettin(pendulos[3]['y0'], parametros)

    ax2.plot(t3, lyap3, color=pendulos[2]['color'], linewidth=1.6, label=f"P3 (Final: {lyap3[-1]:.2f} 1/s)")
    ax2.plot(t4, lyap4, '--', color=pendulos[3]['color'], linewidth=1.4, label=f"P4 (Final: {lyap4[-1]:.2f} 1/s)")
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax2.set_title('Pareja 2: Diferencias Más Grandes\n' + r"$\Delta\theta_1(0) = 0.3$ rad", fontsize=11)
    ax2.set_xlabel('Tiempo $t$ (s)', fontsize=10)
    ax2.set_ylabel(r'Exponente $\lambda(t)$ (s$^{-1}$)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle('Figura 4: Estimación del Exponente Máximo de Lyapunov (Método de Benettin)', fontsize=13, y=1.02)
    plt.tight_layout()

    nombre = 'figura_4_exponente_lyapunov.png'
    plt.savefig(nombre, dpi=200, bbox_inches='tight')
    print(f"        Guardado: {nombre}")
    plt.show(block=False)


# =============================================================================
# ANÁLISIS 5: MAPA DE CALOR FINO (BARRIDO 2D)
# =============================================================================

def figura_5_mapa_calor(parametros):
    print("  [5/5] Generando Figura 5: Mapa de Calor (Grilla Fina de 24x24 = 576 puntos)...")

    # Grilla fina de 24x24 (576 puntos calculados directamente)
    resolucion = 30 #24
    theta1_vals = np.linspace(-np.pi, np.pi, resolucion)
    omega1_vals = np.linspace(-4.0, 4.0, resolucion)

    mapa = np.zeros((resolucion, resolucion))

    # Parámetros rápidos de Benettin para barrido fino
    d0 = 1e-6
    tau = 0.5
    n_pasos = 30 #20

    t0 = time.time()
    for i, w1 in enumerate(omega1_vals):
        for j, th1 in enumerate(theta1_vals):
            # Referencia académica clásica: Stachowiak & Okada (2006)
            # theta2(0) = 0, omega2(0) = 0 fijos en reposo vertical
            estado_ini = [th1, w1, 0.0, 0.0]
            _, hist = calcular_lyapunov_benettin(estado_ini, parametros, d0=d0, tau=tau, n_pasos=n_pasos)
            # Valor crudo calculado (sin recortes artificiales)
            mapa[i, j] = hist[-1]

    print(f"        Barrido finalizado en {time.time() - t0:.1f} s.")

    plt.figure(figsize=(9, 7))
    imagen = plt.imshow(
        mapa,
        extent=[-np.pi, np.pi, -4.0, 4.0],
        origin='lower',
        cmap='viridis',
        aspect='auto',
        interpolation='nearest'  # Celdas discretas reales sin suavizado artificial
    )

    cbar = plt.colorbar(imagen)
    cbar.set_label(r'Exponente Máximo de Lyapunov $\lambda$ (s$^{-1}$)', fontsize=11)

    plt.title('Figura 5: Mapa de Calor del Exponente de Lyapunov\n'
              r'Barrido Fino de $\theta_1(0)$ vs $\omega_1(0)$ con $\theta_2(0)=0, \omega_2(0)=0$', fontsize=12)
    plt.xlabel(r'Ángulo inicial $\theta_1(0)$ (rad)', fontsize=11)
    plt.ylabel(r'Velocidad angular inicial $\omega_1(0)$ (rad/s)', fontsize=11)
    plt.tight_layout()

    nombre = 'figura_5_mapa_calor.png'
    plt.savefig(nombre, dpi=200)
    print(f"        Guardado: {nombre}")
    plt.show(block=False)


# =============================================================================
# BLOQUE PRINCIPAL
# =============================================================================

def main():
    m1, m2 = 1.0, 1.0
    L1, L2 = 1.0, 1.0
    g = 9.81
    parametros = (m1, m2, L1, L2, g)

    print("================================================================")
    print(" ESTUDIO NUMÉRICO DEL PÉNDULO DOBLE IDEAL (4 PÉNDULOS)")
    print("================================================================")

    # Definición de los 4 péndulos dobles:
    # - Pareja 1 (P1 y P2): condiciones casi idénticas (Delta theta1 = 0.0001 rad)
    # - Pareja 2 (P3 y P4): diferencias mayores (Delta theta1 = 0.3 rad)
    pendulos = [
        {
            "nombre": "Péndulo 1",
            "desc": r"$\theta_1=\pi/2, \theta_2=\pi/2$",
            "y0": [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
            "color": "tab:blue",
            "estilo": "-"
        },
        {
            "nombre": "Péndulo 2",
            "desc": r"$\theta_1=\pi/2 + 10^{-4}$",
            "y0": [np.pi / 2.0 + 1e-4, 0.0, np.pi / 2.0, 0.0],
            "color": "tab:cyan",
            "estilo": "--"
        },
        {
            "nombre": "Péndulo 3",
            "desc": r"$\theta_1=1.0, \theta_2=0.5$",
            "y0": [1.0, 0.0, 0.5, 0.0],
            "color": "tab:orange",
            "estilo": "-"
        },
        {
            "nombre": "Péndulo 4",
            "desc": r"$\theta_1=1.3, \theta_2=0.5$",
            "y0": [1.3, 0.0, 0.5, 0.0],
            "color": "tab:red",
            "estilo": "--"
        }
    ]

    # Simulación de las 4 trayectorias durante 40 segundos
    t_span = (0.0, 40.0)
    t_eval = np.linspace(0.0, 40.0, 4000)

    print("\nSimulando las trayectorias de los 4 péndulos durante 40 segundos...")
    soluciones = []
    for p in pendulos:
        sol = solve_ivp(
            derivadas_pendulo,
            t_span,
            p['y0'],
            t_eval=t_eval,
            method='RK45',
            args=parametros,
            rtol=1e-8,
            atol=1e-8
        )
        soluciones.append(sol)

    # 1. Ángulos vs tiempo para los 4 péndulos
    figura_1_angulos(soluciones, pendulos)

    # 2. Energía mecánica en función del tiempo para los 4 péndulos
    figura_2_energia(soluciones, pendulos, parametros)

    # 3. Diagramas de fase para cada uno de los 4 péndulos
    figura_3_fase(soluciones, pendulos)

    # 4. Exponente de Lyapunov (Pareja 1 y Pareja 2)
    figura_4_lyapunov(pendulos, parametros)

    # 5. Mapa de calor del exponente de Lyapunov con grilla fina
    figura_5_mapa_calor(parametros)

    print("\n================================================================")
    print(" ¡Todas las figuras han sido generadas y guardadas exitosamente!")
    print("================================================================")
    plt.show()


if __name__ == '__main__':
    main()
