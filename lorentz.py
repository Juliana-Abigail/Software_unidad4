"""
lorentz.py - Movimiento de una partícula cargada en campos E y B (Unidad 4).

Fuerza de Lorentz:   F = q (E + v x B)      ->      m dv/dt = q (E + v x B)

Se integra con el método de Boris, el estándar en física de plasmas:
  1. medio paso de aceleración por E
  2. rotación exacta de v alrededor de B  (no cambia |v|, conserva la energía)
  3. otro medio paso de E
  4. se avanza la posición con la velocidad media del paso (regla del trapecio)
Unidades SI: m, s, kg, C, V/m, T.
"""
import numpy as np

E_CARGA = 1.602176634e-19   # C
M_ELECTRON = 9.1093837e-31  # kg
M_PROTON = 1.67262192e-27   # kg
M_ALFA = 6.6446573e-27      # kg


class ErrorLorentz(ValueError):
    """Parámetros no válidos."""


def simular(q, m, r0, v0, E, B, t_total, pasos_por_periodo=200, max_pasos=20000):
    """Devuelve (t, r, v) con r y v de forma (N, 3).

    q : carga en coulombs        m : masa en kg
    r0, v0, E, B : vectores de 3 componentes
    """
    if m <= 0:
        raise ErrorLorentz("La masa debe ser mayor que 0.")
    if q == 0:
        raise ErrorLorentz("La carga no puede ser 0.")
    if t_total <= 0:
        raise ErrorLorentz("El tiempo total debe ser mayor que 0.")
    r0, v0, E, B = (np.asarray(a, dtype=float) for a in (r0, v0, E, B))
    Bmod = np.linalg.norm(B)
    # Paso de tiempo: una fracción del periodo ciclotrónico (o del tiempo total si B = 0)
    dt = (2 * np.pi * m / (abs(q) * Bmod)) / pasos_por_periodo if Bmod > 0 else t_total / 1000
    n = int(min(max(np.ceil(t_total / dt), 10), max_pasos))
    dt = t_total / n

    r = np.empty((n + 1, 3)); v = np.empty((n + 1, 3))
    r[0], v[0] = r0, v0
    qmdt2 = q * dt / (2 * m)
    t_vec = qmdt2 * B                      # vector de rotación de Boris
    s_vec = 2 * t_vec / (1 + t_vec @ t_vec)
    for k in range(n):
        v_menos = v[k] + qmdt2 * E
        v_prima = v_menos + np.cross(v_menos, t_vec)
        v_mas = v_menos + np.cross(v_prima, s_vec)
        v[k + 1] = v_mas + qmdt2 * E
        r[k + 1] = r[k] + 0.5 * (v[k] + v[k + 1]) * dt   # regla del trapecio
    return np.linspace(0, t_total, n + 1), r, v


# ---------------------------------------------------------- resultados analíticos
def analisis(q, m, v0, E, B):
    """Cantidades teóricas del movimiento (diccionario)."""
    v0, E, B = (np.asarray(a, dtype=float) for a in (v0, E, B))
    Bmod = np.linalg.norm(B)
    res = {"Ec0": 0.5 * m * float(v0 @ v0)}
    if Bmod == 0:
        res["acel"] = q * E / m
        return res
    b = B / Bmod
    v_par = float(v0 @ b)
    v_perp = float(np.linalg.norm(v0 - v_par * b))
    wc = abs(q) * Bmod / m
    res.update({
        "v_par": v_par, "v_perp": v_perp,
        "w": wc, "T": 2 * np.pi / wc, "f": wc / (2 * np.pi),
        "radio": v_perp / wc,
        "paso": abs(v_par) * 2 * np.pi / wc,
        "v_deriva": np.cross(E, B) / Bmod ** 2,
    })
    return res


def formato_longitud(x):
    a = abs(x)
    if a >= 1:
        return f"{x:.3g} m"
    if a >= 1e-3:
        return f"{x * 1e3:.3g} mm"
    if a >= 1e-6:
        return f"{x * 1e6:.3g} µm"
    return f"{x * 1e9:.3g} nm"


# ---------------------------------------------------------- ejemplos listos
# nombre: (q en e, m kg, v0, E, B, t_total)
T_E = 2 * np.pi * M_ELECTRON / (E_CARGA * 1e-3)
T_P = 2 * np.pi * M_PROTON / (E_CARGA * 1e-2)
EJEMPLOS = {
    "Electrón en B uniforme (círculo)": (-1, M_ELECTRON, (1e6, 0, 0), (0, 0, 0), (0, 0, 1e-3), 3 * T_E),
    "Protón en B uniforme (círculo)": (1, M_PROTON, (1e5, 0, 0), (0, 0, 0), (0, 0, 1e-2), 3 * T_P),
    "Electrón con velocidad inclinada (hélice)": (-1, M_ELECTRON, (1e6, 0, 5e5), (0, 0, 0), (0, 0, 1e-3), 4 * T_E),
    "Selector de velocidades (E y B cruzados)": (1, M_PROTON, (1e5, 0, 0), (0, 1e3, 0), (0, 0, 1e-2), 2e-6),
    "Deriva E×B (cicloide)": (1, M_PROTON, (0, 0, 0), (0, 1e3, 0), (0, 0, 1e-2), 3 * T_P),
    "Solo campo eléctrico (parábola)": (-1, M_ELECTRON, (1e6, 0, 0), (0, 1e3, 0), (0, 0, 0), 2e-8),
}
