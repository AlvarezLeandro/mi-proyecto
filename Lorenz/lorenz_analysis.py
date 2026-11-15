"""
===============================================================================
PROYECTO: SIMULACIÓN Y ANÁLISIS NUMÉRICO DEL SISTEMA CAÓTICO DE LORENZ
===============================================================================
Autor: Leandro Alvarez & Antigravity (Google DeepMind)
Fecha: Noviembre 2026
Descripción:
    Implementación completa y pedagógica para la simulación, caracterización
    y análisis numérico del sistema caótico de Lorenz (1963).

Estructura del programa:
    1. Librerías y configuración estética (tema científico oscuro)
    2. Parámetros del sistema y condiciones iniciales
    3. Definición de las ecuaciones diferenciales de Lorenz
    4. Simulación numérica principal (solve_ivp con RK45)
    5. Análisis 1: Evolución temporal de las variables (x, y, z)
    6. Análisis 2: Espacio de fases y atractor extraño 3D
    7. Análisis 3: Sensibilidad a las condiciones iniciales (Efecto Mariposa)
    8. Análisis 4: Cálculo del máximo exponente de Lyapunov (Benettin)
    9. Análisis 5: Influencia del parámetro rho (transición al caos)
   10. Análisis 6: Sección de Poincaré (corte z = 27 con z_punto > 0)
   11. Análisis 7: Diagrama de bifurcaciones (rho vs z_max)
   12. Análisis 8: Animación tridimensional de la trayectoria
   13. Verificación de los resultados numéricos
   14. Bloque de ejecución principal
===============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter


# =============================================================================
# 1. LIBRERÍAS Y CONFIGURACIÓN ESTÉTICA
# =============================================================================

# Carpeta de salida para almacenar automáticamente las figuras y animaciones
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
CARPETA_RESULTADOS = os.path.join(DIRECTORIO_ACTUAL, "resultados")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Configuración de estilo científico oscuro (Dark Theme sobrio y legible)
plt.rcParams.update({
    "figure.facecolor": "#0B1120",      # Azul marino muy oscuro
    "axes.facecolor": "#0F172A",        # Pizarra oscura
    "axes.edgecolor": "#475569",        # Borde gris azulado discreto
    "axes.labelcolor": "#E2E8F0",       # Etiquetas claras legibles
    "xtick.color": "#94A3B8",           # Ticks en gris claro
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",            # Cuadrícula tenue
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "text.color": "#F8FAFC",            # Texto blanco suave
    "legend.facecolor": "#1E293B",      # Fondo de leyenda
    "legend.edgecolor": "#475569",
    "font.size": 10,
    "figure.titlesize": 13,
    "axes.titlesize": 11,
})

# Paleta de colores científicos de alto contraste
COLOR_X = "#38BDF8"       # Celeste / Cyan
COLOR_Y = "#F59E0B"       # Ámbar / Naranja suave
COLOR_Z = "#34D399"       # Esmeralda / Verde menta
COLOR_TRAZADO = "#60A5FA"   # Azul claro para trayectoria 3D
COLOR_ALT = "#F43F5E"       # Rosa / Carmín para contrastes


# =============================================================================
# 2. DEFINICIÓN DEL SISTEMA DE LORENZ Y PARÁMETROS
# =============================================================================

# Parámetros clásicos del sistema de Lorenz (1963)
SIGMA_DEFAULT = 10.0      # Número de Prandtl (viscosidad / difusividad térmica)
RHO_DEFAULT = 28.0        # Número de Rayleigh normalizado (diferencia de temperatura)
BETA_DEFAULT = 8.0 / 3.0  # Razón de aspecto geométrica de los cilindros convectivos

# Condiciones iniciales estándar
X0_DEFAULT = 1.0
Y0_DEFAULT = 1.0
Z0_DEFAULT = 1.0

# Intervalo temporal estándar de simulación
T_INICIO = 0.0
T_FIN = 40.0
DT_SALIDA = 0.01          # Paso de evaluación para gráficos suaves
CANTIDAD_PUNTOS = int((T_FIN - T_INICIO) / DT_SALIDA)

# Tolerancias numéricas del integrador adaptativo
RTOL_DEFAULT = 1e-8
ATOL_DEFAULT = 1e-8


def lorenz(t, estado, sigma, rho, beta):
    """
    Ecuaciones diferenciales del sistema de Lorenz:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parámetros:
        t: Tiempo actual (requerido por solve_ivp).
        estado: Lista o vector con [x, y, z].
        sigma, rho, beta: Parámetros físicos adimensionales del sistema.

    Retorna:
        [dxdt, dydt, dzdt]: Derivadas temporales de cada coordenada.
    """
    x, y, z = estado
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# =============================================================================
# 3. SIMULACIÓN NUMÉRICA PRINCIPAL
# =============================================================================

def simular_lorenz(estado_inicial, t_span, t_eval, sigma=SIGMA_DEFAULT,
                    rho=RHO_DEFAULT, beta=BETA_DEFAULT, rtol=RTOL_DEFAULT,
                    atol=ATOL_DEFAULT):
    """
    Integra numéricamente el sistema de Lorenz usando Runge-Kutta 4(5) adaptativo.
    Reutilizada por los distintos análisis para evitar cálculos duplicados.
    """
    solucion = solve_ivp(
        fun=lorenz,
        t_span=t_span,
        y0=estado_inicial,
        t_eval=t_eval,
        method="RK45",
        args=(sigma, rho, beta),
        rtol=rtol,
        atol=atol
    )
    return solucion


# =============================================================================
# 4. ANÁLISIS 5.1: EVOLUCIÓN TEMPORAL DE LAS VARIABLES
# =============================================================================

def analisis_1_evolucion_temporal(t, x, y, z):
    """
    Genera gráficos de la evolución temporal de cada variable x(t), y(t) y z(t).
    Muestra la naturaleza aperiódica e irregular de las oscilaciones.
    """
    print("\n--- Ejecutando Análisis 5.1: Evolución Temporal de las Variables ---")

    variables = [
        ("x(t) - Tasa de rotación convectiva", x, COLOR_X, "01_evolucion_x.png"),
        ("y(t) - Gradiente térmico horizontal", y, COLOR_Y, "01_evolucion_y.png"),
        ("z(t) - Distorsión del perfil vertical de temperatura", z, COLOR_Z, "01_evolucion_z.png")
    ]

    for nombre, datos, color, nombre_archivo in variables:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, datos, color=color, linewidth=1.1, label=nombre.split(" - ")[0])
        ax.set_title(f"Sistema de Lorenz: {nombre}", pad=10)
        ax.set_xlabel("Tiempo adimensional $t$ [u.a.]")
        ax.set_ylabel("Amplitud [u.a.]")
        ax.grid(True)
        ax.legend(loc="upper right")
        plt.tight_layout()

        ruta = os.path.join(CARPETA_RESULTADOS, nombre_archivo)
        plt.savefig(ruta, dpi=200)
        plt.show(block=False)
        print(f"  Guardado: {nombre_archivo}")

    # Gráfico conjunto para visión global sincronizada
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax1.plot(t, x, color=COLOR_X, linewidth=1.0, label="$x(t)$")
    ax1.set_ylabel("$x$ [u.a.]")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.plot(t, y, color=COLOR_Y, linewidth=1.0, label="$y(t)$")
    ax2.set_ylabel("$y$ [u.a.]")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    ax3.plot(t, z, color=COLOR_Z, linewidth=1.0, label="$z(t)$")
    ax3.set_ylabel("$z$ [u.a.]")
    ax3.set_xlabel("Tiempo adimensional $t$ [u.a.]")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    plt.suptitle("Evolución Temporal Conjunta del Sistema de Lorenz", y=0.98)
    plt.tight_layout()
    ruta_conjunta = os.path.join(CARPETA_RESULTADOS, "01_evolucion_temporal_conjunta.png")
    plt.savefig(ruta_conjunta, dpi=200)
    plt.show(block=False)
    print("  Guardado: 01_evolucion_temporal_conjunta.png")


# =============================================================================
# 5. ANÁLISIS 5.2: ESPACIO DE FASES Y ATRACTOR EXTRAÑO (3D)
# =============================================================================

def analisis_2_espacio_fases_3d(t, x, y, z, t_corte_transitorio=5.0):
    """
    Representa la trayectoria en el espacio de fases tridimensional (x, y, z).
    Descarta el transitorio inicial para visualizar la estructura del atractor.
    """
    print("\n--- Ejecutando Análisis 5.2: Espacio de Fases y Atractor Extraño 3D ---")

    # Descartar transitorio inicial
    mascara = t >= t_corte_transitorio
    x_atr = x[mascara]
    y_atr = y[mascara]
    z_atr = z[mascara]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#0F172A")

    # Trazado continuo del atractor con degradado visual sutil
    ax.plot(x_atr, y_atr, z_atr, color=COLOR_TRAZADO, linewidth=0.65, alpha=0.85,
            label="Atractor de Lorenz")

    # Marcar los dos puntos fijos inestables de convección C+ y C-
    # Para rho > 1: x* = y* = +/- sqrt(beta*(rho - 1)), z* = rho - 1
    x_c = np.sqrt(BETA_DEFAULT * (RHO_DEFAULT - 1.0))
    z_c = RHO_DEFAULT - 1.0
    ax.scatter([x_c, -x_c], [x_c, -x_c], [z_c, z_c], color=COLOR_ALT, s=35,
               label=r"Puntos fijos $C^+$ y $C^-$", zorder=5)

    ax.set_title(rf"Atractor Extraño de Lorenz ($\sigma={SIGMA_DEFAULT}, \rho={RHO_DEFAULT}, \beta={BETA_DEFAULT:.2f}$)" + "\n"
                 f"Transitorio descartado ($t > {t_corte_transitorio}$ u.a.)", pad=15)
    ax.set_xlabel("x [u.a.]", labelpad=8)
    ax.set_ylabel("y [u.a.]", labelpad=8)
    ax.set_zlabel("z [u.a.]", labelpad=8)

    ax.view_init(elev=22, azim=-62)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_RESULTADOS, "02_atractor_lorenz_3d.png")
    plt.savefig(ruta, dpi=200)
    plt.show(block=False)
    print("  Guardado: 02_atractor_lorenz_3d.png")


# =============================================================================
# 6. ANÁLISIS 5.3: SENSIBILIDAD A LAS CONDICIONES INICIALES (EFECTO MARIPOSA)
# =============================================================================

def analisis_3_sensibilidad(sigma=SIGMA_DEFAULT, rho=RHO_DEFAULT, beta=BETA_DEFAULT):
    """
    Compara dos trayectorias que comienzan en condiciones iniciales casi idénticas
    (separadas por 10^-8) y analiza la separación euclídea d(t) y su saturación.
    """
    print("\n--- Ejecutando Análisis 5.3: Sensibilidad a las Condiciones Iniciales ---")

    estado_1 = [1.0, 1.0, 1.0]
    delta_0 = 1e-8
    estado_2 = [1.0 + delta_0, 1.0, 1.0]

    t_span = (0.0, 35.0)
    t_eval = np.linspace(0.0, 35.0, 3500)

    sol1 = simular_lorenz(estado_1, t_span, t_eval, sigma, rho, beta)
    sol2 = simular_lorenz(estado_2, t_span, t_eval, sigma, rho, beta)

    t = sol1.t
    x1, y1, z1 = sol1.y
    x2, y2, z2 = sol2.y

    # Distancia euclídea en el espacio de fases tridimensional
    d_t = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    # 1. Comparación 3D de ambas trayectorias
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#0F172A")
    ax.plot(x1, y1, z1, color=COLOR_X, linewidth=0.7, alpha=0.8,
            label=r"Trayectoria 1: $\mathbf{X}_1(0) = (1, 1, 1)$")
    ax.plot(x2, y2, z2, color=COLOR_ALT, linewidth=0.7, alpha=0.8, linestyle="--",
            label=rf"Trayectoria 2: $\mathbf{{X}}_2(0) = (1 + 10^{{-8}}, 1, 1)$")
    ax.set_title("Sensibilidad Extrema a Condiciones Iniciales (Espacio 3D)", pad=15)
    ax.set_xlabel("x [u.a.]")
    ax.set_ylabel("y [u.a.]")
    ax.set_zlabel("z [u.a.]")
    ax.view_init(elev=20, azim=-55)
    ax.legend(loc="upper left")
    plt.tight_layout()
    ruta_3d = os.path.join(CARPETA_RESULTADOS, "03_sensibilidad_trayectorias_3d.png")
    plt.savefig(ruta_3d, dpi=200)
    plt.show(block=False)
    print("  Guardado: 03_sensibilidad_trayectorias_3d.png")

    # 2. Separación d(t) en escala lineal y logarítmica
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_lin.plot(t, d_t, color=COLOR_Y, linewidth=1.2, label=r"Distancia euclídea $d(t)$")
    ax_lin.set_title("Evolución de la Separación entre Trayectorias", pad=8)
    ax_lin.set_ylabel("Distancia $d(t)$ [u.a.]")
    ax_lin.grid(True)
    ax_lin.legend(loc="upper left")

    ax_log.plot(t, np.maximum(d_t, 1e-16), color="#A855F7", linewidth=1.2,
                label=r"$\log_{10} d(t)$ (Crecimiento exponencial y saturación)")
    ax_log.set_yscale("log")
    ax_log.axhline(y=35.0, color="#64748B", linestyle=":", label="Cota de saturación del atractor")
    ax_log.set_xlabel("Tiempo $t$ [u.a.]")
    ax_log.set_ylabel("Distancia $d(t)$ (escala log)")
    ax_log.grid(True, which="both")
    ax_log.legend(loc="lower right")

    plt.suptitle("Análisis Cuantitativo de la Divergencia de Trayectorias", y=0.98)
    plt.tight_layout()
    ruta_dist = os.path.join(CARPETA_RESULTADOS, "03_sensibilidad_distancia.png")
    plt.savefig(ruta_dist, dpi=200)
    plt.show(block=False)
    print("  Guardado: 03_sensibilidad_distancia.png")


# =============================================================================
# 7. ANÁLISIS 5.4: MÁXIMO EXPONENTE DE LYAPUNOV (MÉTODO DE BENETTIN)
# =============================================================================

def analisis_4_lyapunov(sigma=SIGMA_DEFAULT, rho=RHO_DEFAULT, beta=BETA_DEFAULT,
                        d0=1e-7, tau=0.1, n_pasos=350):
    """
    Calcula el máximo exponente de Lyapunov mediante el algoritmo de Benettin
    con renormalizaciones periódicas periódicas para evitar la saturación finita:
        lambda_max = (1 / T) * sum(ln(d_i / d0))
    """
    print("\n--- Ejecutando Análisis 5.4: Máximo Exponente de Lyapunov (Benettin) ---")

    # 1. Descartar transitorio: evolucionar 15 u.a. hasta entrar al atractor
    sol_transitorio = simular_lorenz([1.0, 1.0, 1.0], (0.0, 15.0),
                                     np.linspace(0.0, 15.0, 1500), sigma, rho, beta)
    estado_base = sol_transitorio.y[:, -1]

    # 2. Inicializar estado de referencia y estado perturbado a distancia d0
    s_ref = estado_base.copy()
    s_pert = s_ref.copy()
    s_pert[0] += d0  # Perturbación inicial microscópica en x

    suma_log = 0.0
    tiempos = []
    lyap_acumulado = []

    for k in range(1, n_pasos + 1):
        # Evolucionar ambas trayectorias durante el intervalo tau
        sol_r = solve_ivp(lorenz, (0.0, tau), s_ref, args=(sigma, rho, beta),
                          method="RK45", rtol=1e-8, atol=1e-8)
        sol_p = solve_ivp(lorenz, (0.0, tau), s_pert, args=(sigma, rho, beta),
                          method="RK45", rtol=1e-8, atol=1e-8)

        s_r_fin = sol_r.y[:, -1]
        s_p_fin = sol_p.y[:, -1]

        # Vector de separación y distancia alcanzada en el espacio tridimensional
        delta_vec = s_p_fin - s_r_fin
        d1 = np.linalg.norm(delta_vec)

        # Acumular tasa de expansión logarítmica
        suma_log += np.log(d1 / d0)
        t_actual = k * tau
        tiempos.append(t_actual)
        lyap_acumulado.append(suma_log / t_actual)

        # Renormalización: reubicar el estado perturbado a distancia d0 en la dirección de divergencia
        s_ref = s_r_fin
        s_pert = s_r_fin + d0 * (delta_vec / d1)

    lambda_estimado = lyap_acumulado[-1]
    print(f"  Estimación final de lambda_max: {lambda_estimado:.4f} [1/u.a.]")
    print(f"  Referencia en literatura (Lorenz clásico): ~0.9056 [1/u.a.]")

    # Gráfico de convergencia del exponente acumulado
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos, lyap_acumulado, color=COLOR_Z, linewidth=1.5,
            label=rf"Estimación acumulada $\lambda(t) \to {lambda_estimado:.3f}$")
    ax.axhline(y=0.9056, color=COLOR_ALT, linestyle="--", linewidth=1.2,
               label=r"Referencia teórica ($\lambda \approx 0.906$)")
    ax.axhline(y=0.0, color="#64748B", linestyle=":", label=r"Frontera de caos ($\lambda = 0$)")

    ax.set_title(r"Convergencia del Máximo Exponente de Lyapunov $\lambda_{\max}$ (Benettin)", pad=10)
    ax.set_xlabel("Tiempo acumulado de renormalización $T$ [u.a.]")
    ax.set_ylabel(r"Exponente de Lyapunov estimado $\lambda$ [1/u.a.]")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_RESULTADOS, "04_exponente_lyapunov.png")
    plt.savefig(ruta, dpi=200)
    plt.show(block=False)
    print("  Guardado: 04_exponente_lyapunov.png")

    return lambda_estimado


# =============================================================================
# 8. ANÁLISIS 5.5: INFLUENCIA DEL PARÁMETRO RHO
# =============================================================================

def analisis_5_influencia_rho(sigma=SIGMA_DEFAULT, beta=BETA_DEFAULT):
    """
    Estudia cómo cambia la dinámica al variar rho (Rayleigh).
    Muestra la transición desde puntos fijos estables hasta el caos turbulento.
    """
    print("\n--- Ejecutando Análisis 5.5: Influencia del Parámetro rho ---")

    # Valores representativos documentados en la literatura
    valores_rho = [14.0, 21.0, 28.0, 100.0]
    subtitulos = [
        r"$\rho = 14$ (Punto fijo estable $C^+$ y $C^-$)",
        r"$\rho = 21$ (Pre-caos / Caos transitorio)",
        r"$\rho = 28$ (Atractor caótico clásico)",
        r"$\rho = 100$ (Caos extendido y ciclos límite)"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), subplot_kw={"projection": "3d"})
    colores_rho = [COLOR_X, COLOR_Y, COLOR_Z, COLOR_ALT]

    t_span = (0.0, 30.0)
    t_eval = np.linspace(5.0, 30.0, 2500)  # Descartar transitorio t < 5

    for i, (r_val, ax, color, sub) in enumerate(zip(valores_rho, axes.flatten(), colores_rho, subtitulos)):
        sol = simular_lorenz([1.0, 1.0, 1.0], t_span, t_eval, sigma, r_val, beta)
        ax.set_facecolor("#0F172A")
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color=color, linewidth=0.7, alpha=0.85)
        ax.set_title(sub, fontsize=10, pad=8)
        ax.set_xlabel("x", labelpad=5)
        ax.set_ylabel("y", labelpad=5)
        ax.set_zlabel("z", labelpad=5)
        ax.view_init(elev=20, azim=-60)
        ax.grid(True, linestyle=":", alpha=0.25)

    plt.suptitle(r"Evolución de los Atractores al Modificar $\rho$ ($\sigma=10, \beta=8/3$)", y=0.98)
    plt.tight_layout()
    ruta_comp = os.path.join(CARPETA_RESULTADOS, "05_influencia_rho_atractores.png")
    plt.savefig(ruta_comp, dpi=200)
    plt.show(block=False)
    print("  Guardado: 05_influencia_rho_atractores.png")

    # Estimación de Lyapunov para una serie de valores de rho
    rhos_barrido = [10.0, 14.0, 18.0, 22.0, 25.0, 28.0, 35.0, 45.0, 60.0]
    lyaps = []
    print("  Calculando exponentes de Lyapunov para el barrido en rho...")
    for r in rhos_barrido:
        # Pre-evolución
        s_ini = solve_ivp(lorenz, (0.0, 10.0), [1.0, 1.0, 1.0], args=(sigma, r, beta),
                          method="RK45", rtol=1e-6, atol=1e-6).y[:, -1]
        # Estimación con Benettin
        d0, tau, N = 1e-6, 0.1, 100
        sp = s_ini.copy(); sp[0] += d0
        s_r = s_ini.copy()
        s_log = 0.0
        for _ in range(N):
            sr_next = solve_ivp(lorenz, (0.0, tau), s_r, args=(sigma, r, beta), method="RK45", rtol=1e-6, atol=1e-6).y[:, -1]
            sp_next = solve_ivp(lorenz, (0.0, tau), sp, args=(sigma, r, beta), method="RK45", rtol=1e-6, atol=1e-6).y[:, -1]
            diff = sp_next - sr_next
            d1 = np.linalg.norm(diff)
            s_log += np.log(d1 / d0)
            s_r = sr_next
            sp = sr_next + d0 * (diff / d1)
        lyaps.append(s_log / (N * tau))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rhos_barrido, lyaps, marker="o", color=COLOR_Y, linewidth=1.5, markersize=5)
    ax.axhline(0.0, color="#64748B", linestyle=":", label=r"Frontera de caos ($\lambda = 0$)")
    ax.axvline(24.74, color=COLOR_ALT, linestyle="--", alpha=0.8,
               label=r"Bifurcación de Hopf subcrítica ($\rho_H \approx 24.74$)")
    ax.set_title(r"Máximo Exponente de Lyapunov en Función de $\rho$", pad=10)
    ax.set_xlabel(r"Parámetro de control $\rho$ [u.a.]")
    ax.set_ylabel(r"$\lambda_{\max}$ [1/u.a.]")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    ruta_lyap_rho = os.path.join(CARPETA_RESULTADOS, "05_lyapunov_vs_rho.png")
    plt.savefig(ruta_lyap_rho, dpi=200)
    plt.show(block=False)
    print("  Guardado: 05_lyapunov_vs_rho.png")


# =============================================================================
# 9. ANÁLISIS 5.6: SECCIONES DE POINCARÉ
# =============================================================================

def analisis_6_seccion_poincare(sigma=SIGMA_DEFAULT, rho=RHO_DEFAULT, beta=BETA_DEFAULT,
                                 z_plano=27.0):
    """
    Construye la sección de Poincaré en el plano z = 27 con cruces ascendentes (dz/dt > 0).
    Utiliza interpolación lineal exacta entre pasos para máxima precisión geométrica.
    """
    print("\n--- Ejecutando Análisis 5.6: Sección de Poincaré (Plano z = 27, dz/dt > 0) ---")

    # Simular una trayectoria larga para acumular una cantidad representativa de cortes
    t_span = (0.0, 120.0)
    t_eval = np.linspace(10.0, 120.0, 15000)  # Descartar transitorio t < 10

    sol = simular_lorenz([1.0, 1.0, 1.0], t_span, t_eval, sigma, rho, beta)
    x, y, z = sol.y

    cruces_x = []
    cruces_y = []

    for k in range(len(z) - 1):
        z_k = z[k]
        z_kp1 = z[k + 1]

        # Condición de cruce del plano z = z_plano de abajo hacia arriba (dz/dt > 0)
        if z_k < z_plano and z_kp1 >= z_plano:
            # Interpolación lineal precisa: z(alpha) = z_k + alpha * (z_{k+1} - z_k) = z_plano
            fraccion = (z_plano - z_k) / (z_kp1 - z_k)
            x_interp = x[k] + fraccion * (x[k + 1] - x[k])
            y_interp = y[k] + fraccion * (y[k + 1] - y[k])

            # Verificar que dz/dt sea positivo: dz/dt = x*y - beta*z
            dzdt = x_interp * y_interp - beta * z_plano
            if dzdt > 0:
                cruces_x.append(x_interp)
                cruces_y.append(y_interp)

    print(f"  Puntos de intersección detectados: {len(cruces_x)}")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(cruces_x, cruces_y, s=10, color=COLOR_X, alpha=0.75, edgecolors="none",
               label=f"Cruces ($N = {len(cruces_x)}$)")
    ax.set_title(f"Sección de Poincaré del Atractor de Lorenz\n"
                 f"Plano $z = {z_plano}$, con $\\dot{{z}} > 0$", pad=10)
    ax.set_xlabel("Coordenada $x$ [u.a.]")
    ax.set_ylabel("Coordenada $y$ [u.a.]")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_RESULTADOS, "06_seccion_poincare.png")
    plt.savefig(ruta, dpi=200)
    plt.show(block=False)
    print("  Guardado: 06_seccion_poincare.png")

    return len(cruces_x)


# =============================================================================
# 10. ANÁLISIS 5.7: DIAGRAMA DE BIFURCACIONES
# =============================================================================

def analisis_7_diagrama_bifurcaciones(sigma=SIGMA_DEFAULT, beta=BETA_DEFAULT,
                                       rho_min=10.0, rho_max=38.0, n_rhos=60):
    """
    Genera el diagrama de bifurcaciones detectando los máximos locales de z(t)
    en función del parámetro de control rho.
    """
    print(f"\n--- Ejecutando Análisis 5.7: Diagrama de Bifurcaciones (rho in [{rho_min}, {rho_max}]) ---")

    rhos = np.linspace(rho_min, rho_max, n_rhos)
    todos_rhos = []
    todos_z_max = []

    t_inicio_barrido = time.time()

    for r in rhos:
        # Simular y descartar transitorio inicial (t in [15, 32])
        sol = solve_ivp(
            lorenz, (0.0, 32.0), [1.0, 1.0, 1.0], args=(sigma, r, beta),
            method="RK45", rtol=1e-6, atol=1e-6,
            t_eval=np.linspace(15.0, 32.0, 2000)
        )
        z = sol.y[2]

        # Detección de picos locales: z[k-1] < z[k] > z[k+1]
        mascara_picos = (z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])
        picos_z = z[1:-1][mascara_picos]

        for pz in picos_z:
            todos_rhos.append(r)
            todos_z_max.append(pz)

    print(f"  Barrido completado en {time.time() - t_inicio_barrido:.1f} s. Picos detectados: {len(todos_z_max)}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(todos_rhos, todos_z_max, s=4, color=COLOR_Y, alpha=0.55, edgecolors="none")
    ax.axvline(x=24.74, color=COLOR_ALT, linestyle="--", linewidth=1.2,
               label=r"Hopf subcrítica ($\rho_H \approx 24.74$)")

    ax.set_title("Diagrama de Bifurcaciones del Sistema de Lorenz: Máximos Locales de $z(t)$", pad=10)
    ax.set_xlabel(r"Parámetro de control $\rho$ [u.a.]")
    ax.set_ylabel(r"Máximos locales de $z$ ($z_{\max}$) [u.a.]")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.tight_layout()

    ruta = os.path.join(CARPETA_RESULTADOS, "07_diagrama_bifurcaciones.png")
    plt.savefig(ruta, dpi=200)
    plt.show(block=False)
    print("  Guardado: 07_diagrama_bifurcaciones.png")


# =============================================================================
# 11. ANÁLISIS 5.8: ANIMACIÓN DE LA TRAYECTORIA TRIDIMENSIONAL
# =============================================================================

def analisis_8_animacion(t, x, y, z, nombre_gif="08_animacion_lorenz.gif"):
    """
    Crea una animación tridimensional de la trayectoria progresiva del atractor
    usando FuncAnimation y guarda el resultado como GIF con Pillow.
    """
    print("\n--- Ejecutando Análisis 5.8: Animación de la Trayectoria 3D ---")

    # Muestrear a 1000 puntos para optimizar la generación fluida de ~100 frames
    paso_muestreo = max(1, len(t) // 1000)
    x_sub = x[::paso_muestreo]
    y_sub = y[::paso_muestreo]
    z_sub = z[::paso_muestreo]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#0F172A")

    # Límites fijos para evitar cambios de escala durante la animación
    ax.set_xlim(x_sub.min() - 2, x_sub.max() + 2)
    ax.set_ylim(y_sub.min() - 2, y_sub.max() + 2)
    ax.set_zlim(z_sub.min() - 2, z_sub.max() + 2)

    ax.set_title("Evolución Dinámica en el Atractor de Lorenz (3D)", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    ax.grid(True, linestyle=":", alpha=0.3)

    linea, = ax.plot([], [], [], color=COLOR_TRAZADO, linewidth=0.8, alpha=0.85)
    punto_cabeza, = ax.plot([], [], [], marker="o", color=COLOR_ALT, markersize=5)

    n_frames = 100
    salto = len(x_sub) // n_frames

    def actualizar(frame):
        fin = min(len(x_sub), (frame + 1) * salto)
        linea.set_data(x_sub[:fin], y_sub[:fin])
        linea.set_3d_properties(z_sub[:fin])
        if fin > 0:
            punto_cabeza.set_data([x_sub[fin - 1]], [y_sub[fin - 1]])
            punto_cabeza.set_3d_properties([z_sub[fin - 1]])
        return linea, punto_cabeza

    t0_anim = time.time()
    animacion = FuncAnimation(fig, actualizar, frames=n_frames, interval=100, blit=False, repeat=True)

    ruta_gif = os.path.join(CARPETA_RESULTADOS, nombre_gif)
    try:
        escritor = PillowWriter(fps=10)
        animacion.save(ruta_gif, writer=escritor)
        print(f"  Animación guardada con éxito en {time.time() - t0_anim:.1f} s: {nombre_gif}")
    except Exception as e:
        print(f"  Aviso: No se pudo guardar el GIF ({e}), pero la animación interactiva está lista.")

    plt.show(block=False)
    return animacion


# =============================================================================
# 12. VERIFICACIÓN DE LOS RESULTADOS NUMÉRICOS
# =============================================================================

def analisis_9_verificacion(solucion_principal, lambda_estimado, n_cruces_poincare,
                            sigma=SIGMA_DEFAULT, rho=RHO_DEFAULT, beta=BETA_DEFAULT):
    """
    Ejecuta comprobaciones numéricas sistemáticas para certificar la confiabilidad
    del integrador, las soluciones y las propiedades dinámicas.
    """
    print("\n================================================================")
    print(" VERIFICACIÓN DE LOS RESULTADOS NUMÉRICOS")
    print("================================================================")

    comprobaciones = []

    # 1. Integración numérica exitosa
    if solucion_principal.success:
        comprobaciones.append(("Integración RK45 finalizada con éxito", "SUPERADA",
                                f"{len(solucion_principal.t)} puntos calculados."))
    else:
        comprobaciones.append(("Integración RK45", "FALLIDA", solucion_principal.message))

    # 2. Ausencia de valores NaN o infinitos
    datos = solucion_principal.y
    tiene_nan = np.isnan(datos).any()
    tiene_inf = np.isinf(datos).any()
    if not tiene_nan and not tiene_inf:
        comprobaciones.append(("Estabilidad de datos (sin NaN ni Inf)", "SUPERADA",
                                f"Valores acotados: x in [{datos[0].min():.1f}, {datos[0].max():.1f}], "
                                f"z in [{datos[2].min():.1f}, {datos[2].max():.1f}]"))
    else:
        comprobaciones.append(("Estabilidad de datos", "REQUIERE ATENCIÓN", "Se encontraron valores NaN o Inf."))

    # 3. Exponente de Lyapunov positivo (firma del caos determinista)
    if lambda_estimado > 0.0:
        comprobaciones.append(("Exponente de Lyapunov positivo (caos)", "SUPERADA",
                                f"lambda = {lambda_estimado:.4f} > 0 [1/u.a.]"))
    else:
        comprobaciones.append(("Exponente de Lyapunov positivo", "FALLIDA",
                                f"lambda = {lambda_estimado:.4f} <= 0"))

    # 4. Comparación con valor de referencia en literatura (~0.9056)
    ref_lyap = 0.9056
    error_rel_lyap = abs(lambda_estimado - ref_lyap) / ref_lyap
    if error_rel_lyap < 0.10:  # Menor al 10% de error en simulación de tiempo finito
        comprobaciones.append(("Concordancia con literatura (lambda ~ 0.906)", "SUPERADA",
                                f"Error relativo: {error_rel_lyap * 100:.2f}% (Estimado: {lambda_estimado:.3f}, Ref: {ref_lyap})"))
    else:
        comprobaciones.append(("Concordancia con literatura", "REQUIERE ATENCIÓN",
                                f"Desviación mayor al 10%: {error_rel_lyap * 100:.2f}%"))

    # 5. Prueba de sensibilidad a tolerancias numéricas (rtol=1e-7 vs rtol=1e-10)
    sol_tol_gruesa = simular_lorenz([1.0, 1.0, 1.0], (0.0, 10.0), np.linspace(0.0, 10.0, 1000),
                                     rtol=1e-7, atol=1e-7)
    sol_tol_fina = simular_lorenz([1.0, 1.0, 1.0], (0.0, 10.0), np.linspace(0.0, 10.0, 1000),
                                   rtol=1e-10, atol=1e-10)
    dif_max_tol = np.max(np.abs(sol_tol_gruesa.y - sol_tol_fina.y))
    if dif_max_tol < 1e-3:
        comprobaciones.append(("Consistencia frente a tolerancias", "SUPERADA",
                                f"Diferencia máxima en t=10s: {dif_max_tol:.2e} [u.a.]"))
    else:
        comprobaciones.append(("Consistencia frente a tolerancias", "REQUIERE ATENCIÓN",
                                f"Diferencia observada: {dif_max_tol:.2e}"))

    # 6. Detección consistente de cruces de Poincaré
    if n_cruces_poincare > 20:
        comprobaciones.append(("Sección de Poincaré (cruces z = 27, dz/dt > 0)", "SUPERADA",
                                f"{n_cruces_poincare} cruces ascendentes calculados."))
    else:
        comprobaciones.append(("Sección de Poincaré", "REQUIERE ATENCIÓN",
                                f"Cruces insuficientes: {n_cruces_poincare}"))

    # Mostrar en consola de forma estructurada
    for item, estado, detalle in comprobaciones:
        icono = "✅" if estado == "SUPERADA" else ("⚠️" if estado == "REQUIERE ATENCIÓN" else "❌")
        print(f"  {icono} [{estado}] {item}: {detalle}")

    # Generar tarjeta gráfica de verificación
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    texto_resumen = "RESUMEN DE VERIFICACIÓN NUMÉRICA - SISTEMA DE LORENZ\n\n"
    for item, estado, detalle in comprobaciones:
        texto_resumen += f"• [{estado}] {item}\n   -> {detalle}\n\n"

    ax.text(0.03, 0.95, texto_resumen, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", fontfamily="monospace", color="#F8FAFC")
    plt.tight_layout()
    ruta_verif = os.path.join(CARPETA_RESULTADOS, "09_resumen_verificacion.png")
    plt.savefig(ruta_verif, dpi=200)
    plt.show(block=False)
    print("  Guardado: 09_resumen_verificacion.png")


# =============================================================================
# 13. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("================================================================")
    print(" SIMULADOR Y ANALIZADOR NUMÉRICO DEL SISTEMA CAÓTICO DE LORENZ")
    print("================================================================")
    t_inicio_total = time.time()

    # 1. Simulación de referencia
    print("\nEjecutando simulación de referencia...")
    t_span = (T_INICIO, T_FIN)
    t_eval = np.linspace(T_INICIO, T_FIN, CANTIDAD_PUNTOS)
    solucion = simular_lorenz([X0_DEFAULT, Y0_DEFAULT, Z0_DEFAULT], t_span, t_eval)

    t = solucion.t
    x, y, z = solucion.y

    # 2. Análisis 5.1: Evolución temporal
    analisis_1_evolucion_temporal(t, x, y, z)

    # 3. Análisis 5.2: Atractor extraño en 3D
    analisis_2_espacio_fases_3d(t, x, y, z)

    # 4. Análisis 5.3: Sensibilidad a condiciones iniciales
    analisis_3_sensibilidad()

    # 5. Análisis 5.4: Máximo exponente de Lyapunov
    lambda_estimado = analisis_4_lyapunov()

    # 6. Análisis 5.5: Influencia de rho
    analisis_5_influencia_rho()

    # 7. Análisis 5.6: Secciones de Poincaré
    n_cruces = analisis_6_seccion_poincare()

    # 8. Análisis 5.7: Diagrama de bifurcaciones
    analisis_7_diagrama_bifurcaciones()

    # 9. Análisis 5.8: Animación de la trayectoria (mantenemos referencia viva)
    anim_ref = analisis_8_animacion(t, x, y, z)

    # 10. Verificación de los resultados numéricos
    analisis_9_verificacion(solucion, lambda_estimado, n_cruces)

    tiempo_total = time.time() - t_inicio_total
    print("\n================================================================")
    print(f" ¡Todos los análisis fueron completados con éxito en {tiempo_total:.1f} s!")
    print(f" Archivos generados en: {CARPETA_RESULTADOS}")
    print(" Mostrando todas las figuras y animación en ventanas emergentes...")
    print(" (Cierre las ventanas para finalizar la ejecución)")
    print("================================================================")

    # Mostrar interactivamente todas las ventanas emergentes en pantalla
    plt.show()


if __name__ == "__main__":
    main()
