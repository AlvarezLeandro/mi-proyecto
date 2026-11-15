"""
Estudio Numérico del Caos en el Péndulo Doble
Proyecto Universitario - Computación y Sistemas Dinámicos

Contenido completo:
- Etapa A: Integración numérica y evolución temporal de los ángulos.
- Etapa B: Proyecciones en el espacio de fases (regímenes regular y caótico).
- Etapa C: Verificación de conservación de la energía mecánica total.
- Etapa D: Sensibilidad extrema a las condiciones iniciales y saturación.
- Etapa E: Estimación del exponente máximo de Lyapunov (Método de Benettin).
- Etapa F: Mapa de calor de caos (Barrido bidimensional de condiciones iniciales).
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
    Calcula las derivadas temporales del vector de estado del péndulo doble.

    Vector de estado:
        estado = [theta1, omega1, theta2, omega2]

    Retorna:
        d_estado_dt = [omega1, alpha1, omega2, alpha2]
    """
    theta1, omega1, theta2, omega2 = estado
    delta_theta = theta1 - theta2

    cos_d = np.cos(delta_theta)
    sin_d = np.sin(delta_theta)

    # Matriz de inercia M(theta):
    # [ M11  M12 ] [ alpha1 ] = [ b1 ]
    # [ M21  M22 ] [ alpha2 ]   [ b2 ]
    M11 = (m1 + m2) * (L1 ** 2)
    M12 = m2 * L1 * L2 * cos_d
    M21 = M12
    M22 = m2 * (L2 ** 2)

    # Vector de términos independientes b(theta, omega)
    b1 = -m2 * L1 * L2 * (omega2 ** 2) * sin_d - (m1 + m2) * g * L1 * np.sin(theta1)
    b2 =  m2 * L1 * L2 * (omega1 ** 2) * sin_d - m2 * g * L2 * np.sin(theta2)

    # Resolución analítica exacta del sistema 2x2 (Regla de Cramer):
    # Evita el overhead de llamar a np.linalg.solve miles de veces por segundo
    det_M = M11 * M22 - M12 * M21
    alpha1 = (b1 * M22 - b2 * M12) / det_M
    alpha2 = (M11 * b2 - M21 * b1) / det_M

    return [omega1, alpha1, omega2, alpha2]


def calcular_energia(theta1, omega1, theta2, omega2, m1, m2, L1, L2, g):
    """
    Calcula la energía mecánica total E = T + V del péndulo doble.
    Se fija V=0 en el punto más bajo (theta1=0, theta2=0) para asegurar E >= 0.
    """
    delta_theta = theta1 - theta2

    # Energía cinética (T)
    T = (0.5 * (m1 + m2) * (L1 ** 2) * (omega1 ** 2)
         + 0.5 * m2 * (L2 ** 2) * (omega2 ** 2)
         + m2 * L1 * L2 * omega1 * omega2 * np.cos(delta_theta))

    # Energía potencial gravitatoria (V >= 0)
    V = ((m1 + m2) * g * L1 * (1.0 - np.cos(theta1))
         + m2 * g * L2 * (1.0 - np.cos(theta2)))

    return T + V


# =============================================================================
# ETAPA A: Integración numérica y evolución temporal
# =============================================================================

def ejecutar_etapa_a(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa A: Ángulos en función del tiempo ---")
    parametros = (m1, m2, L1, L2, g)

    # Condición inicial: soltar ambos brazos a 90 grados en reposo
    estado_inicial = [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
    t_span = (0.0, 20.0)
    t_eval = np.linspace(0.0, 20.0, 2000)

    solucion = solve_ivp(
        fun=derivadas_pendulo,
        t_span=t_span,
        y0=estado_inicial,
        t_eval=t_eval,
        method='RK45',
        args=parametros,
        rtol=1e-8,
        atol=1e-8
    )

    t = solucion.t
    theta1 = solucion.y[0]
    theta2 = solucion.y[2]

    plt.figure(figsize=(10, 5))
    plt.plot(t, theta1, label=r'$\theta_1(t)$ (primer péndulo)', color='tab:blue', linewidth=1.5)
    plt.plot(t, theta2, label=r'$\theta_2(t)$ (segundo péndulo)', color='tab:orange', linewidth=1.5)
    plt.title('Etapa A: Ángulos en función del tiempo (Péndulo Doble)', fontsize=13)
    plt.xlabel('Tiempo $t$ (s)', fontsize=11)
    plt.ylabel('Ángulo (rad)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    nombre_archivo = 'etapa_a_angulos_vs_tiempo.png'
    plt.savefig(nombre_archivo, dpi=200)
    print(f"Gráfico de Etapa A guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# ETAPA B: Diagramas de fase
# =============================================================================

def ejecutar_etapa_b(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa B: Proyecciones en el Espacio de Fases ---")
    parametros = (m1, m2, L1, L2, g)
    t_span = (0.0, 30.0)
    t_eval = np.linspace(0.0, 30.0, 4000)

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, caso in enumerate(casos):
        sol = solve_ivp(
            fun=derivadas_pendulo,
            t_span=t_span,
            y0=caso["y0"],
            t_eval=t_eval,
            method='RK45',
            args=parametros,
            rtol=1e-8,
            atol=1e-8
        )

        theta1, omega1 = sol.y[0], sol.y[1]
        theta2, omega2 = sol.y[2], sol.y[3]

        # Péndulo 1
        ax1 = axes[0, i]
        ax1.plot(theta1, omega1, color=caso["color"], linewidth=0.8, alpha=0.85)
        ax1.plot(caso["y0"][0], caso["y0"][1], 'ko', markersize=4, label='Inicio')
        ax1.set_title(f"{caso['nombre']}\n{caso['subtitulo']}", fontsize=10)
        ax1.set_xlabel(r'$\theta_1$ (rad)', fontsize=10)
        ax1.set_ylabel(r'$\omega_1$ (rad/s)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=8)

        # Péndulo 2
        ax2 = axes[1, i]
        ax2.plot(theta2, omega2, color=caso["color"], linewidth=0.8, alpha=0.85)
        ax2.plot(caso["y0"][2], caso["y0"][3], 'ko', markersize=4, label='Inicio')
        ax2.set_xlabel(r'$\theta_2$ (rad)', fontsize=10)
        ax2.set_ylabel(r'$\omega_2$ (rad/s)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)

    plt.suptitle("Etapa B: Proyecciones en el Espacio de Fases", fontsize=14, y=1.00)
    plt.tight_layout()

    nombre_archivo = "etapa_b_diagramas_fase.png"
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')
    print(f"Gráfico de Etapa B guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# ETAPA C: Verificación física y numérica (Conservación de energía)
# =============================================================================

def ejecutar_etapa_c(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa C: Verificación de Conservación de la Energía ---")
    parametros = (m1, m2, L1, L2, g)

    estado_inicial = [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
    t_span = (0.0, 30.0)
    t_eval = np.linspace(0.0, 30.0, 3000)

    configuraciones_tolerancia = [
        {"nombre": "Tolerancia gruesa (rtol=1e-4, atol=1e-6)", "rtol": 1e-4, "atol": 1e-6, "color": "tab:red"},
        {"nombre": "Tolerancia media (rtol=1e-8, atol=1e-8)",   "rtol": 1e-8, "atol": 1e-8, "color": "tab:orange"},
        {"nombre": "Tolerancia fina (rtol=1e-11, atol=1e-11)", "rtol": 1e-11, "atol": 1e-11, "color": "tab:blue"}
    ]

    plt.figure(figsize=(11, 6))

    for config in configuraciones_tolerancia:
        sol = solve_ivp(
            fun=derivadas_pendulo,
            t_span=t_span,
            y0=estado_inicial,
            t_eval=t_eval,
            method='RK45',
            args=parametros,
            rtol=config["rtol"],
            atol=config["atol"]
        )

        theta1, omega1 = sol.y[0], sol.y[1]
        theta2, omega2 = sol.y[2], sol.y[3]

        E = calcular_energia(theta1, omega1, theta2, omega2, m1, m2, L1, L2, g)
        E0 = E[0]

        error_relativo = np.abs(E - E0) / E0
        max_error = np.max(error_relativo)

        print(f"  {config['nombre']}: Max Error Relativo = {max_error:.2e}")

        plt.plot(sol.t, np.maximum(error_relativo, 1e-16),
                 label=f"{config['nombre']} (máx: {max_error:.1e})",
                 color=config["color"], linewidth=1.3)

    plt.yscale('log')
    plt.title('Etapa C: Error Relativo de la Energía Mecánica $|E(t) - E_0| / E_0$', fontsize=13)
    plt.xlabel('Tiempo $t$ (s)', fontsize=11)
    plt.ylabel('Error relativo $|E(t) - E_0| / E_0$', fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()

    nombre_archivo = "etapa_c_conservacion_energia.png"
    plt.savefig(nombre_archivo, dpi=200)
    print(f"Gráfico de Etapa C guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# ETAPA D: Sensibilidad a las condiciones iniciales
# =============================================================================

def ejecutar_etapa_d(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa D: Sensibilidad a Condiciones Iniciales ---")
    parametros = (m1, m2, L1, L2, g)

    # Dos estados iniciales idénticos salvo por una pequeña perturbación en theta1:
    delta_0 = 1e-5  # 0.00001 radianes de diferencia inicial
    estado_A = [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
    estado_B = [np.pi / 2.0 + delta_0, 0.0, np.pi / 2.0, 0.0]

    t_span = (0.0, 25.0)
    t_eval = np.linspace(0.0, 25.0, 2500)

    # Simular ambas trayectorias
    sol_A = solve_ivp(derivadas_pendulo, t_span, estado_A, t_eval=t_eval, method='RK45', args=parametros, rtol=1e-9, atol=1e-9)
    sol_B = solve_ivp(derivadas_pendulo, t_span, estado_B, t_eval=t_eval, method='RK45', args=parametros, rtol=1e-9, atol=1e-9)

    t = sol_A.t
    theta1_A, omega1_A = sol_A.y[0], sol_A.y[1]
    theta2_A, omega2_A = sol_A.y[2], sol_A.y[3]

    theta1_B, omega1_B = sol_B.y[0], sol_B.y[1]
    theta2_B, omega2_B = sol_B.y[2], sol_B.y[3]

    # Separación en el espacio de fases 4D: norma euclídea d(t) = ||s_A(t) - s_B(t)||
    d_fase = np.sqrt((theta1_A - theta1_B) ** 2 +
                     (omega1_A - omega1_B) ** 2 +
                     (theta2_A - theta2_B) ** 2 +
                     (omega2_A - omega2_B) ** 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Subgráfico 1: Comparación directa del ángulo theta1(t)
    ax1.plot(t, theta1_A, label=r'Trayectoria A: $\theta_{1A}(0) = \pi/2$', color='tab:blue', linewidth=1.5)
    ax1.plot(t, theta1_B, '--', label=rf'Trayectoria B: $\theta_{{1B}}(0) = \pi/2 + {delta_0}$', color='tab:red', linewidth=1.3)
    ax1.set_title('Etapa D: Sensibilidad Extrema a las Condiciones Iniciales', fontsize=13)
    ax1.set_ylabel(r'$\theta_1(t)$ (rad)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=10)

    # Subgráfico 2: Distancia en el espacio de fases en escala logarítmica
    ax2.plot(t, d_fase, color='tab:purple', linewidth=1.5, label='Distancia en el espacio de fases $d(t)$')
    ax2.set_yscale('log')
    ax2.set_xlabel('Tiempo $t$ (s)', fontsize=11)
    ax2.set_ylabel('Separación $d(t)$ (escala log)', fontsize=11)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=10)

    # Anotación explicativa de la saturación
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax2.text(1.0, 1.2, 'Zona de Saturación (tamaño finito del espacio de fases)', fontsize=9, color='gray')

    plt.tight_layout()
    nombre_archivo = "etapa_d_sensibilidad_trayectorias.png"
    plt.savefig(nombre_archivo, dpi=200)
    print(f"Gráfico de Etapa D guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# ETAPA E: Exponente máximo de Lyapunov (Método de Benettin)
# =============================================================================

def calcular_lyapunov_benettin(estado_inicial, parametros, d0=1e-6, tau=0.5, n_pasos=80):
    """
    Estima el exponente máximo de Lyapunov usando el método de Benettin
    con renormalizaciones periódicas periódicas para evitar saturación.

    Parámetros:
        estado_inicial: vector de estado [theta1, omega1, theta2, omega2]
        parametros: tupla (m1, m2, L1, L2, g)
        d0: magnitud fija de la perturbación inicial (pequeña)
        tau: intervalo temporal entre renormalizaciones sucesivas
        n_pasos: cantidad total de renormalizaciones (Tiempo total = n_pasos * tau)

    Retorna:
        tiempos: array con los instantes t_k = k * tau
        lyap_acumulado: array con el exponente estimado acumulado lambda(t_k)
    """
    s_ref = np.array(estado_inicial, dtype=float)

    # Perturbación inicial microscópica orientada en theta1: ||delta_0|| = d0
    s_pert = s_ref.copy()
    s_pert[0] += d0

    suma_log_crecimiento = 0.0
    tiempos = []
    lyap_acumulado = []

    for k in range(1, n_pasos + 1):
        # 1. Integrar ambas trayectorias durante el intervalo tau
        sol_ref = solve_ivp(derivadas_pendulo, (0.0, tau), s_ref, args=parametros, method='RK45', rtol=1e-7, atol=1e-7)
        sol_pert = solve_ivp(derivadas_pendulo, (0.0, tau), s_pert, args=parametros, method='RK45', rtol=1e-7, atol=1e-7)

        # Estados finales del intervalo
        s_ref_fin = sol_ref.y[:, -1]
        s_pert_fin = sol_pert.y[:, -1]

        # 2. Vector de separación y distancia alcanzada en 4D
        delta_vec = s_pert_fin - s_ref_fin
        d1 = np.linalg.norm(delta_vec)

        # 3. Acumular la tasa de divergencia logarítmica
        suma_log_crecimiento += np.log(d1 / d0)
        t_actual = k * tau
        tiempos.append(t_actual)
        lyap_acumulado.append(suma_log_crecimiento / t_actual)

        # 4. Renormalizar: recolocar la trayectoria perturbada a distancia d0
        # a lo largo de la dirección de máxima divergencia
        s_ref = s_ref_fin
        s_pert = s_ref_fin + d0 * (delta_vec / d1)

    return np.array(tiempos), np.array(lyap_acumulado)


def ejecutar_etapa_e(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa E: Exponente Máximo de Lyapunov (Benettin) ---")
    parametros = (m1, m2, L1, L2, g)

    # Comparar régimen caótico vs régimen regular
    caso_caotico = [np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
    caso_regular = [0.1, 0.0, 0.1, 0.0]

    d0 = 1e-6
    tau = 0.5
    n_pasos = 100  # Tiempo total de simulación = 50 s

    print("  Calculando exponente de Lyapunov para el caso caótico...")
    t_caos, lyap_caos = calcular_lyapunov_benettin(caso_caotico, parametros, d0=d0, tau=tau, n_pasos=n_pasos)

    print("  Calculando exponente de Lyapunov para el caso regular...")
    t_reg, lyap_reg = calcular_lyapunov_benettin(caso_regular, parametros, d0=d0, tau=tau, n_pasos=n_pasos)

    val_final_caos = lyap_caos[-1]
    val_final_reg = lyap_reg[-1]
    print(f"  Estimación final caso caótico: lambda = {val_final_caos:.3f} s^-1")
    print(f"  Estimación final caso regular: lambda = {val_final_reg:.3f} s^-1")

    plt.figure(figsize=(10, 6))
    plt.plot(t_caos, lyap_caos, color='tab:red', linewidth=1.8, label=rf'Régimen Caótico ($\lambda \to {val_final_caos:.2f}$ s$^{{-1}}$)')
    plt.plot(t_reg, lyap_reg, color='tab:green', linewidth=1.8, label=rf'Régimen Regular ($\lambda \to {val_final_reg:.2f}$ s$^{{-1}}$)')
    plt.axhline(0, color='black', linestyle=':', alpha=0.7)

    plt.title('Etapa E: Convergencia del Exponente Máximo de Lyapunov (Método de Benettin)', fontsize=13)
    plt.xlabel('Tiempo acumulado $t$ (s)', fontsize=11)
    plt.ylabel(r'Exponente de Lyapunov estimado $\lambda(t)$ (s$^{-1}$)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    nombre_archivo = "etapa_e_exponente_lyapunov.png"
    plt.savefig(nombre_archivo, dpi=200)
    print(f"Gráfico de Etapa E guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# ETAPA F: Mapa de calor del caos (Barrido bidimensional)
# =============================================================================

def ejecutar_etapa_f(m1, m2, L1, L2, g):
    print("\n--- Ejecutando Etapa F: Mapa de Calor del Caos (Barrido 2D) ---")
    parametros = (m1, m2, L1, L2, g)

    # Grilla de barrido: Variamos ángulo theta1(0) y velocidad angular omega1(0)
    resolucion = 16  # Grilla 16x16 (256 simulaciones rápidas y didácticas)
    theta1_vals = np.linspace(-np.pi, np.pi, resolucion)
    omega1_vals = np.linspace(-5.0, 5.0, resolucion)

    mapa_lyapunov = np.zeros((resolucion, resolucion))

    d0 = 1e-6
    tau = 0.4
    n_pasos = 20  # Estimación rápida para cada punto (T = 8 s)

    print(f"  Iniciando barrido de {resolucion}x{resolucion} = {resolucion*resolucion} condiciones iniciales...")
    t_inicio = time.time()

    for i, w1 in enumerate(omega1_vals):
        for j, th1 in enumerate(theta1_vals):
            # Fijamos theta2(0) = 0 y omega2(0) = 0
            estado_ini = [th1, w1, 0.0, 0.0]
            _, lyap_hist = calcular_lyapunov_benettin(estado_ini, parametros, d0=d0, tau=tau, n_pasos=n_pasos)
            # Acotamos a valores no negativos para distinguir caos (lambda > 0) de orden (lambda ~ 0)
            mapa_lyapunov[i, j] = max(0.0, lyap_hist[-1])

    tiempo_total = time.time() - t_inicio
    print(f"  Barrido completado en {tiempo_total:.1f} segundos.")

    plt.figure(figsize=(10, 8))
    # Visualizar con imshow
    imagen = plt.imshow(
        mapa_lyapunov,
        extent=[-np.pi, np.pi, -5.0, 5.0],
        origin='lower',
        cmap='inferno',
        aspect='auto',
        interpolation='bicubic'
    )

    cbar = plt.colorbar(imagen)
    cbar.set_label(r'Exponente Máximo de Lyapunov $\lambda$ (s$^{-1}$)', fontsize=11)

    plt.title('Etapa F: Mapa de Calor del Caos en el Péndulo Doble\n(Barrido de $\\theta_1(0)$ vs $\\omega_1(0)$ con $\\theta_2=0, \\omega_2=0$)', fontsize=13)
    plt.xlabel(r'Ángulo inicial $\theta_1(0)$ (rad)', fontsize=11)
    plt.ylabel(r'Velocidad angular inicial $\omega_1(0)$ (rad/s)', fontsize=11)
    plt.grid(color='white', linestyle=':', alpha=0.3)
    plt.tight_layout()

    nombre_archivo = "etapa_f_mapa_calor_caos.png"
    plt.savefig(nombre_archivo, dpi=200)
    print(f"Gráfico de Etapa F guardado como: {nombre_archivo}")
    plt.show(block=False)


# =============================================================================
# BLOQUE PRINCIPAL
# =============================================================================

def main():
    m1 = 1.0    # Masa 1 (kg)
    m2 = 1.0    # Masa 2 (kg)
    L1 = 1.0    # Longitud 1 (m)
    L2 = 1.0    # Longitud 2 (m)
    g  = 9.81   # Gravedad (m/s^2)

    print("================================================================")
    print(" ESTUDIO NUMÉRICO INTEGRAL DEL CAOS EN EL PÉNDULO DOBLE")
    print("================================================================")

    ejecutar_etapa_a(m1, m2, L1, L2, g)
    ejecutar_etapa_b(m1, m2, L1, L2, g)
    ejecutar_etapa_c(m1, m2, L1, L2, g)
    ejecutar_etapa_d(m1, m2, L1, L2, g)
    ejecutar_etapa_e(m1, m2, L1, L2, g)
    ejecutar_etapa_f(m1, m2, L1, L2, g)

    print("\n================================================================")
    print(" ¡Todas las etapas (A, B, C, D, E, F) fueron completadas!")
    print(" Gráficos generados en la carpeta del proyecto.")
    print("================================================================")

    # Bloqueante final solo si se corre en entorno interactivo con ventanas emergentes
    plt.show()


if __name__ == '__main__':
    main()
