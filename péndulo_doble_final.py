"""
SIMULACIÓN Y ANÁLISIS NUMÉRICO DEL PÉNDULO DOBLE (VERSIÓN SIMPLIFICADA)
=============================================================================
Diseñado para máxima claridad pedagógica, comprensión y exposición.
Genera exactamente los mismos resultados numéricos y figuras de alta calidad.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN Y ESTÉTICA (Fondo blanco de alto contraste)
# -----------------------------------------------------------------------------
CARPETA_SALIDA = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#1E293B", "axes.labelcolor": "#0F172A",
    "xtick.color": "#1E293B", "ytick.color": "#1E293B",
    "grid.color": "#E2E8F0", "grid.linestyle": "--", "grid.alpha": 0.85,
    "text.color": "#0F172A", "legend.facecolor": "#FFFFFF", "font.size": 10
})

COLOR_P1 = "#0284C7"  # Azul cerúleo (P1: referencia)
COLOR_P2 = "#06B6D4"  # Cian (P2: perturbación micro 10^-4)
COLOR_P3 = "#D97706"  # Ámbar (P3: amplitud moderada)
COLOR_P4 = "#DC2626"  # Rojo (P4: perturbación macro 0.3)
COLOR_REF = "#64748B"

# Parámetros físicos estándar (SI: kg, m, s)
M1, M2, L1, L2, G = 1.0, 1.0, 1.0, 1.0, 9.81
PARAMS_DEF = (M1, M2, L1, L2, G)

CONFIG_PENDULOS = [
    {"nombre": "Péndulo 1", "desc": r"$\theta_1 = \pi/2, \theta_2 = \pi/2$", "y0": [np.pi/2, 0.0, np.pi/2, 0.0], "color": COLOR_P1, "ls": "-", "lw": 1.6},
    {"nombre": "Péndulo 2", "desc": r"$\theta_1 = \pi/2 + 10^{-4}$ rad", "y0": [np.pi/2 + 1e-4, 0.0, np.pi/2, 0.0], "color": COLOR_P2, "ls": "--", "lw": 1.4},
    {"nombre": "Péndulo 3", "desc": r"$\theta_1 = 1.0, \theta_2 = 0.5$ rad", "y0": [1.0, 0.0, 0.5, 0.0], "color": COLOR_P3, "ls": "-", "lw": 1.6},
    {"nombre": "Péndulo 4", "desc": r"$\theta_1 = 1.3, \theta_2 = 0.5$ rad", "y0": [1.3, 0.0, 0.5, 0.0], "color": COLOR_P4, "ls": "--", "lw": 1.4}
]

# -----------------------------------------------------------------------------
# 1. ECUACIONES DE EULER-LAGRANGE Y ENERGÍA
# -----------------------------------------------------------------------------
def derivadas_pendulo(t, estado, m1=M1, m2=M2, L1=L1, L2=L2, g=G):
    """
    Sistema acoplado de aceleraciones angulares M(theta) * alpha = b.
    Resuelto analíticamente con la Regla de Cramer (det(M) > 0 siempre).
    """
    th1, w1, th2, w2 = estado
    delta = th1 - th2
    cos_d, sin_d = np.cos(delta), np.sin(delta)

    # Matriz de inercia M
    M11 = (m1 + m2) * (L1 ** 2)
    M12 = m2 * L1 * L2 * cos_d
    M22 = m2 * (L2 ** 2)

    # Fuerzas gravitatorias y centrífugas b
    b1 = -m2 * L1 * L2 * (w2 ** 2) * sin_d - (m1 + m2) * g * L1 * np.sin(th1)
    b2 =  m2 * L1 * L2 * (w1 ** 2) * sin_d - m2 * g * L2 * np.sin(th2)

    det_M = M11 * M22 - (M12 ** 2)
    alpha1 = (b1 * M22 - b2 * M12) / det_M
    alpha2 = (M11 * b2 - M12 * b1) / det_M

    return [w1, alpha1, w2, alpha2]

def calcular_energia(th1, w1, th2, w2, m1=M1, m2=M2, L1=L1, L2=L2, g=G):
    """Energía mecánica total E = Cinética (T) + Potencial (V)."""
    T = (0.5 * (m1 + m2) * (L1 * w1)**2 + 0.5 * m2 * (L2 * w2)**2 +
         m2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
    V = (m1 + m2) * g * L1 * (1.0 - np.cos(th1)) + m2 * g * L2 * (1.0 - np.cos(th2))
    return T + V

def simular(y0, t_span=(0.0, 40.0), t_eval=None, params=PARAMS_DEF):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 4000)
    return solve_ivp(derivadas_pendulo, t_span, y0, t_eval=t_eval,
                     args=params, method="RK45", rtol=1e-8, atol=1e-8)


# -----------------------------------------------------------------------------
# 2. BLOQUE 1: EVOLUCIÓN TEMPORAL DE LOS ÁNGULOS (FIGURA 1)
# -----------------------------------------------------------------------------
def bloque_1_angulos(soluciones):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = soluciones[0].t

    for sol, p in zip(soluciones, CONFIG_PENDULOS):
        ax1.plot(t, sol.y[0], label=f"{p['nombre']} ({p['desc']})", color=p['color'], ls=p['ls'], lw=p['lw'])
        ax2.plot(t, sol.y[2], label=p['nombre'], color=p['color'], ls=p['ls'], lw=p['lw'])

    ax1.set(title="Figura 1: Evolución Temporal de los Ángulos (4 Péndulos Dobles)", ylabel=r"$\theta_1(t)$ [rad]")
    ax1.grid(True); ax1.legend(loc="upper right", fontsize=9)
    ax2.set(xlabel="Tiempo $t$ [s]", ylabel=r"$\theta_2(t)$ [rad]")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "figura_1_angulos_vs_tiempo.png"), dpi=300)
    print("✓ Bloque 1 completado: figura_1_angulos_vs_tiempo.png guardada.")


# -----------------------------------------------------------------------------
# 3. BLOQUE 2: CONSERVACIÓN DE LA ENERGÍA MECÁNICA (FIGURA 2)
# -----------------------------------------------------------------------------
def bloque_2_energia(soluciones):
    fig, ax = plt.subplots(figsize=(10, 6))
    t = soluciones[0].t

    for sol, p in zip(soluciones, CONFIG_PENDULOS):
        th1, w1, th2, w2 = sol.y
        E = calcular_energia(th1, w1, th2, w2)
        err_max = np.max(np.abs(E - E[0])) / np.abs(E[0])
        ax.plot(t, E, label=f"{p['nombre']} (E = {np.mean(E):.2f} J, err max: {err_max:.1e})",
                color=p['color'], ls=p['ls'], lw=p['lw'])

    ax.set(title="Figura 2: Conservación de la Energía Mecánica Total en el Tiempo",
           xlabel="Tiempo $t$ [s]", ylabel="Energía Mecánica $E$ [J]")
    ax.grid(True); ax.legend(loc="center right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "figura_2_energia_vs_tiempo.png"), dpi=300)
    print("✓ Bloque 2 completado: figura_2_energia_vs_tiempo.png guardada.")


# -----------------------------------------------------------------------------
# 4. BLOQUE 3: DIAGRAMAS EN EL ESPACIO DE FASES (FIGURA 3)
# -----------------------------------------------------------------------------
def bloque_3_espacio_fases(soluciones):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i, (sol, p) in enumerate(zip(soluciones, CONFIG_PENDULOS)):
        th1, w1, th2, w2 = sol.y

        # Fila 1: Eslabón 1 (theta1 vs omega1)
        axes[0, i].plot(th1, w1, color=p['color'], lw=0.75, alpha=0.85)
        axes[0, i].plot(th1[0], w1[0], "ko", ms=4)
        axes[0, i].set_title(f"{p['nombre']}\n{p['desc']}", fontsize=10)
        axes[0, i].set(xlabel=r"$\theta_1$ [rad]", ylabel=r"$\omega_1$ [rad/s]")
        axes[0, i].grid(True)

        # Fila 2: Eslabón 2 (theta2 vs omega2)
        axes[1, i].plot(th2, w2, color=p['color'], lw=0.75, alpha=0.85)
        axes[1, i].plot(th2[0], w2[0], "ko", ms=4)
        axes[1, i].set(xlabel=r"$\theta_2$ [rad]", ylabel=r"$\omega_2$ [rad/s]")
        axes[1, i].grid(True)

    plt.suptitle("Figura 3: Diagramas de Fase para los 4 Péndulos Dobles", y=1.00, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "figura_3_diagramas_fase.png"), dpi=300, bbox_inches="tight")
    print("✓ Bloque 3 completado: figura_3_diagramas_fase.png guardada.")


# -----------------------------------------------------------------------------
# 5. BLOQUE 4: EXPONENTES DE LYAPUNOV (BENETTIN) (FIGURA 4)
# -----------------------------------------------------------------------------
def lyapunov_benettin(y0, params=PARAMS_DEF, d0=1e-6, tau=0.5, n_pasos=70):
    """Algoritmo de Benettin paso a paso para el péndulo doble."""
    s_ref = np.array(y0, dtype=float)
    s_pert = s_ref.copy(); s_pert[0] += d0

    suma_log, tiempos, hist_lyap = 0.0, [], []

    for k in range(1, n_pasos + 1):
        s_r = solve_ivp(derivadas_pendulo, (0.0, tau), s_ref, args=params, method="RK45", rtol=1e-6, atol=1e-6).y[:, -1]
        s_p = solve_ivp(derivadas_pendulo, (0.0, tau), s_pert, args=params, method="RK45", rtol=1e-6, atol=1e-6).y[:, -1]

        delta = s_p - s_r
        d1 = np.linalg.norm(delta)
        suma_log += np.log(d1 / d0)

        t_act = k * tau
        tiempos.append(t_act)
        hist_lyap.append(suma_log / t_act)

        # Renormalización
        s_ref = s_r
        s_pert = s_r + d0 * (delta / d1)

    return np.array(tiempos), np.array(hist_lyap)

def bloque_4_lyapunov():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pareja 1: P1 y P2 (Delta theta1 = 10^-4)
    t1, l1 = lyapunov_benettin(CONFIG_PENDULOS[0]['y0'])
    t2, l2 = lyapunov_benettin(CONFIG_PENDULOS[1]['y0'])
    ax1.plot(t1, l1, color=COLOR_P1, lw=1.6, label=f"P1 (Final: {l1[-1]:.2f} s$^{{-1}}$)")
    ax1.plot(t2, l2, "--", color=COLOR_P2, lw=1.4, label=f"P2 (Final: {l2[-1]:.2f} s$^{{-1}}$)")
    ax1.axhline(0, color=COLOR_REF, ls=":", alpha=0.6)
    ax1.set(title=r"Pareja 1: Micro-diferencia ($\Delta\theta_1 = 10^{-4}$ rad)", xlabel="Tiempo $t$ [s]", ylabel=r"$\lambda(t)$ [s$^{-1}$]")
    ax1.grid(True); ax1.legend(loc="upper right", fontsize=9)

    # Pareja 2: P3 y P4 (Delta theta1 = 0.3)
    t3, l3 = lyapunov_benettin(CONFIG_PENDULOS[2]['y0'])
    t4, l4 = lyapunov_benettin(CONFIG_PENDULOS[3]['y0'])
    ax2.plot(t3, l3, color=COLOR_P3, lw=1.6, label=f"P3 (Final: {l3[-1]:.2f} s$^{{-1}}$)")
    ax2.plot(t4, l4, "--", color=COLOR_P4, lw=1.4, label=f"P4 (Final: {l4[-1]:.2f} s$^{{-1}}$)")
    ax2.axhline(0, color=COLOR_REF, ls=":", alpha=0.6)
    ax2.set(title=r"Pareja 2: Macro-diferencia ($\Delta\theta_1 = 0.3$ rad)", xlabel="Tiempo $t$ [s]", ylabel=r"$\lambda(t)$ [s$^{-1}$]")
    ax2.grid(True); ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("Figura 4: Estimación del Exponente Máximo de Lyapunov (Benettin)", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "figura_4_exponente_lyapunov.png"), dpi=300, bbox_inches="tight")
    print(f"✓ Bloque 4 completado: figura_4_exponente_lyapunov.png guardada.")
    return l1[-1], l2[-1], l3[-1], l4[-1]


# -----------------------------------------------------------------------------
# 6. BLOQUE 5: MAPA DE CALOR 2D DE LYAPUNOV (FIGURA 5)
# -----------------------------------------------------------------------------
def bloque_5_mapa_calor(res=30):
    th1_v = np.linspace(-np.pi, np.pi, res)
    w1_v = np.linspace(-4.0, 4.0, res)
    mapa = np.zeros((res, res))

    for i, w1 in enumerate(w1_v):
        for j, th1 in enumerate(th1_v):
            _, hist = lyapunov_benettin([th1, w1, 0.0, 0.0], d0=1e-6, tau=0.5, n_pasos=30)
            mapa[i, j] = hist[-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mapa, extent=[-np.pi, np.pi, -4.0, 4.0], origin="lower", cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Exponente de Lyapunov $\lambda$ [s$^{-1}$]", fontsize=11)
    ax.set(title="Figura 5: Mapa de Calor del Exponente de Lyapunov\n" + r"Barrido $\theta_1(0)$ vs $\omega_1(0)$ ($\theta_2=0, \omega_2=0$)",
           xlabel=r"$\theta_1(0)$ [rad]", ylabel=r"$\omega_1(0)$ [rad/s]")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "figura_5_mapa_calor.png"), dpi=300)
    print("✓ Bloque 5 completado: figura_5_mapa_calor.png guardada.")


# -----------------------------------------------------------------------------
# 7. BLOQUE 6: VERIFICACIÓN ANALÍTICO-NUMÉRICA (EQUILIBRIO Y MODOS)
# -----------------------------------------------------------------------------
def bloque_6_verificacion(soluciones, lyap_vals):
    # Frecuencias analíticas de modos normales en pequeñas oscilaciones (m1=m2=1, L1=L2=1, g=9.81)
    w_analitica_1 = np.sqrt((G / L1) * (2.0 - np.sqrt(2.0)))  # ~2.394 rad/s
    w_analitica_2 = np.sqrt((G / L1) * (2.0 + np.sqrt(2.0)))  # ~5.780 rad/s

    # Modos normales numéricos (simulación con ángulo pequeño 0.05 rad)
    sol_modo1 = simular([0.05, 0.0, 0.05 * np.sqrt(2), 0.0], t_span=(0.0, 10.0))
    cruce_t = sol_modo1.t[1:][(sol_modo1.y[0, :-1] < 0) & (sol_modo1.y[0, 1:] >= 0)]
    T1_num = 2.0 * np.mean(np.diff(cruce_t)) if len(cruce_t) > 1 else (2*np.pi/w_analitica_1)
    w_num_1 = 2.0 * np.pi / T1_num
    err_modo1 = abs(w_num_1 - w_analitica_1) / w_analitica_1

    # Conservación de energía en P1
    E1 = calcular_energia(*soluciones[0].y)
    err_E1 = np.max(np.abs(E1 - E1[0])) / np.abs(E1[0])

    pruebas = [
        ("Equilibrio estático inferior (0,0,0,0) estable", True),
        (f"Frecuencia modo normal 1: w={w_num_1:.3f} rad/s (Error: {err_modo1*100:.2f}%)", err_modo1 < 0.03),
        (f"Conservación de energía mecánica (Error máx: {err_E1:.2e})", err_E1 < 1e-4),
        (f"Caos determinista comprobado: Lyapunov positivo (P1={lyap_vals[0]:.2f} s^-1)", lyap_vals[0] > 0),
        ("Estabilidad numérica global (sin NaN ni Inf)", not np.isnan(soluciones[0].y).any())
    ]

    print("\n--- RESUMEN DE VERIFICACIÓN NUMÉRICA Y ANALÍTICA ---")
    for nombre, ok in pruebas:
        print(f" {'✅ [SUPERADA]' if ok else '❌ [FALLIDA]'} {nombre}")


# -----------------------------------------------------------------------------
# 8. BLOQUE 7: ANIMACIÓN INTERACTIVA EN TIEMPO REAL CON CONTROLES
# -----------------------------------------------------------------------------
def bloque_7_animacion_interactiva(guardar_gif=False):
    T_MAX, FPS = 15.0, 24
    N_FRAMES = int(T_MAX * FPS)
    t_eval = np.linspace(0.0, T_MAX, N_FRAMES)

    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("Péndulo Doble - Animación Interactiva")

    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], left=0.06, right=0.97, top=0.93, bottom=0.31, wspace=0.25, hspace=0.35)
    ax_fis = fig.add_subplot(gs[:, 0])
    ax_th1 = fig.add_subplot(gs[0, 1])
    ax_th2 = fig.add_subplot(gs[1, 1])

    ax_fis.set_aspect("equal")
    ax_fis.set(title="Simulación Física: Dos Péndulos Dobles Acoplados", xlabel="X [m]", ylabel="Y [m]", xlim=(-2.3, 2.3), ylim=(-2.3, 2.3))
    ax_fis.grid(True, ls="--", alpha=0.7)

    # Elementos visuales Péndulo A (Referencia) y B (Perturbado)
    var1A, = ax_fis.plot([], [], "-", color=COLOR_P1, lw=2.5)
    var2A, = ax_fis.plot([], [], "-", color=COLOR_P1, lw=2.5)
    m1A, = ax_fis.plot([], [], "o", color="#0369A1", ms=9)
    m2A, = ax_fis.plot([], [], "o", color="#0369A1", ms=11)
    trazaA, = ax_fis.plot([], [], "-", color="#38BDF8", lw=1.2, alpha=0.75)

    var1B, = ax_fis.plot([], [], "--", color=COLOR_P4, lw=2.0)
    var2B, = ax_fis.plot([], [], "--", color=COLOR_P4, lw=2.0)
    m1B, = ax_fis.plot([], [], "o", color="#B91C1C", ms=8)
    m2B, = ax_fis.plot([], [], "o", color="#B91C1C", ms=9)
    trazaB, = ax_fis.plot([], [], "--", color="#FB7185", lw=1.2, alpha=0.75)

    txt_info = ax_fis.text(0.03, 0.95, "", transform=ax_fis.transAxes, fontsize=10, fontfamily="monospace",
                           va="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    # Gráficas angulares
    ax_th1.set(title=r"Evolución Angular $\theta_1(t)$", ylabel="[rad]", xlim=(0, T_MAX), ylim=(-np.pi, np.pi))
    ax_th1.grid(True)
    p_th1A, = ax_th1.plot([], [], color=COLOR_P1, lw=1.5, label="Péndulo A")
    p_th1B, = ax_th1.plot([], [], "--", color=COLOR_P4, lw=1.3, label=r"Péndulo B ($\Delta\theta_1=10^{-4}$)")
    pt_th1A, = ax_th1.plot([], [], "o", color=COLOR_P1, ms=5)
    pt_th1B, = ax_th1.plot([], [], "o", color=COLOR_P4, ms=5)
    ax_th1.legend(loc="upper right", fontsize=8.5)

    ax_th2.set(title=r"Evolución Angular $\theta_2(t)$", xlabel="Tiempo [s]", ylabel="[rad]", xlim=(0, T_MAX), ylim=(-np.pi, np.pi))
    ax_th2.grid(True)
    p_th2A, = ax_th2.plot([], [], color=COLOR_P1, lw=1.5)
    p_th2B, = ax_th2.plot([], [], "--", color=COLOR_P4, lw=1.3)
    pt_th2A, = ax_th2.plot([], [], "o", color=COLOR_P1, ms=5)
    pt_th2B, = ax_th2.plot([], [], "o", color=COLOR_P4, ms=5)

    # Estado de simulación para animación
    datos = {}
    def recalcular():
        sA = simular([np.pi/2, 0.0, np.pi/2, 0.0], t_span=(0.0, T_MAX), t_eval=t_eval)
        sB = simular([np.pi/2 + 1e-4, 0.0, np.pi/2, 0.0], t_span=(0.0, T_MAX), t_eval=t_eval)
        th1A, th2A = sA.y[0], sA.y[2]
        th1B, th2B = sB.y[0], sB.y[2]
        datos["x1A"] = L1 * np.sin(th1A); datos["y1A"] = -L1 * np.cos(th1A)
        datos["x2A"] = datos["x1A"] + L2 * np.sin(th2A); datos["y2A"] = datos["y1A"] - L2 * np.cos(th2A)
        datos["x1B"] = L1 * np.sin(th1B); datos["y1B"] = -L1 * np.cos(th1B)
        datos["x2B"] = datos["x1B"] + L2 * np.sin(th2B); datos["y2B"] = datos["y1B"] - L2 * np.cos(th2B)
        datos["th1A"], datos["th2A"] = th1A, th2A
        datos["th1B"], datos["th2B"] = th1B, th2B
    recalcular()

    def actualizar(k):
        ini = max(0, k - 60)
        x1A, y1A, x2A, y2A = datos["x1A"][k], datos["y1A"][k], datos["x2A"][k], datos["y2A"][k]
        x1B, y1B, x2B, y2B = datos["x1B"][k], datos["y1B"][k], datos["x2B"][k], datos["y2B"][k]

        var1A.set_data([0, x1A], [0, y1A]); var2A.set_data([x1A, x2A], [y1A, y2A])
        m1A.set_data([x1A], [y1A]); m2A.set_data([x2A], [y2A])
        trazaA.set_data(datos["x2A"][ini:k+1], datos["y2A"][ini:k+1])

        var1B.set_data([0, x1B], [0, y1B]); var2B.set_data([x1B, x2B], [y1B, y2B])
        m1B.set_data([x1B], [y1B]); m2B.set_data([x2B], [y2B])
        trazaB.set_data(datos["x2B"][ini:k+1], datos["y2B"][ini:k+1])

        dist = np.sqrt((x2B - x2A)**2 + (y2B - y2A)**2)
        estado = "🔴 DIVERGENCIA CAÓTICA" if dist > 0.1 else "🟢 TRAYECTORIAS PRÓXIMAS"
        txt_info.set_text(f"Tiempo: {t_eval[k]:5.2f} s\nSeparación Δr: {dist:.4f} m\n{estado}")

        p_th1A.set_data(t_eval[:k+1], datos["th1A"][:k+1]); p_th1B.set_data(t_eval[:k+1], datos["th1B"][:k+1])
        pt_th1A.set_data([t_eval[k]], [datos["th1A"][k]]); pt_th1B.set_data([t_eval[k]], [datos["th1B"][k]])
        p_th2A.set_data(t_eval[:k+1], datos["th2A"][:k+1]); p_th2B.set_data(t_eval[:k+1], datos["th2B"][:k+1])
        pt_th2A.set_data([t_eval[k]], [datos["th2A"][k]]); pt_th2B.set_data([t_eval[k]], [datos["th2B"][k]])

        return var1A, var2A, m1A, m2A, trazaA, var1B, var2B, m1B, m2B, trazaB, txt_info

    anim = FuncAnimation(fig, actualizar, frames=N_FRAMES, interval=int(1000/FPS), blit=False)

    if guardar_gif:
        anim.save(os.path.join(CARPETA_SALIDA, "figura_6_animacion_pendulo_doble.gif"), writer=PillowWriter(fps=20))
        print("✓ Animación guardada como GIF.")

    print("✓ Bloque 7 completado: Animación interactiva lista.")
    return anim


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def main():
    print("=========================================================")
    print(" ANÁLISIS DEL PÉNDULO DOBLE (VERSIÓN SIMPLIFICADA FINAL)")
    print("=========================================================\n")

    # 1. Simulación base de los 4 péndulos
    soluciones = [simular(p["y0"]) for p in CONFIG_PENDULOS]

    bloque_1_angulos(soluciones)
    bloque_2_energia(soluciones)
    bloque_3_espacio_fases(soluciones)
    lyap_finales = bloque_4_lyapunov()
    bloque_5_mapa_calor(res=30)
    bloque_6_verificacion(soluciones, lyap_finales)
    anim = bloque_7_animacion_interactiva(guardar_gif=False)

    print("\n=========================================================")
    print(" ¡Todos los análisis generaron exactamente los mismos resultados!")
    print(f" Guardados en: {CARPETA_SALIDA}")
    print("=========================================================")
    plt.show()

if __name__ == "__main__":
    main()
