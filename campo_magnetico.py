"""
campo_magnetico.py - Cálculo del campo magnético B (Unidad 4).

No usa interfaz gráfica, por eso se puede probar solo.

Fundamentos:
  * Hilo recto infinito (Ampère):      B = mu0 * I / (2*pi*r)
  * Ley de Biot-Savart (espira, solenoide, imán):
        dB = mu0/(4*pi) * I * dl x r / |r|^3
    La integral se calcula sumando muchos segmentos pequeños de corriente.
  * Imán de barra: se modela como un solenoide corto (modelo de Ampère:
    la magnetización equivale a corrientes superficiales).
  * Fuerza entre dos hilos paralelos:  F/L = mu0 * I1 * I2 / (2*pi*d)

Convención del dibujo (plano de la pantalla = plano x-y, eje z sale hacia ti):
  * Hilos rectos: corriente en el eje z.  I > 0 sale de la pantalla (punto).
  * Espira / solenoide / imán: el eje es el eje x.  I > 0 produce B en +x.
Todas las longitudes están en metros, B en teslas, I en amperios.
"""
import numpy as np

MU0 = 4e-7 * np.pi  # permeabilidad del vacío (T·m/A)

TIPOS = ["Imán de barra", "Solenoide", "Espira circular", "Hilo recto", "Dos hilos paralelos"]


class ErrorCampo(ValueError):
    """Parámetros no válidos."""


# ---------------------------------------------------------------- hilos rectos
def campo_hilo(X, Y, I, x0=0.0, y0=0.0, eps=1e-9):
    """Campo de un hilo infinito en (x0, y0) con corriente I a lo largo de +z."""
    dx, dy = X - x0, Y - y0
    r2 = dx ** 2 + dy ** 2 + eps ** 2
    k = MU0 * I / (2 * np.pi * r2)
    return -k * dy, k * dx


# ------------------------------------------------------ Biot-Savart (espiras)
def _espira_segmentos(xc, radio, n=90):
    """Segmentos (puntos medios y dl) de una espira circular de eje x centrada en xc."""
    th = (np.arange(n) + 0.5) * 2 * np.pi / n
    dth = 2 * np.pi / n
    pos = np.stack([np.full(n, xc), radio * np.cos(th), radio * np.sin(th)], axis=1)
    dl = np.stack([np.zeros(n), -radio * np.sin(th) * dth, radio * np.cos(th) * dth], axis=1)
    return pos, dl


def biot_savart(puntos, pos, dl, I, eps=1e-9, bloque=256):
    """B en 'puntos' (N,3) creado por segmentos (pos (M,3), dl (M,3)) con corriente I.

    Se procesa por bloques de puntos para no gastar demasiada memoria.
    """
    salida = np.empty((len(puntos), 3))
    for a in range(0, len(puntos), bloque):
        r = puntos[a:a + bloque, None, :] - pos[None, :, :]          # (n, M, 3)
        d3 = (np.sum(r ** 2, axis=2) + eps ** 2) ** 1.5              # |r|^3
        dB = np.cross(dl[None, :, :], r) / d3[:, :, None]            # dl x r / r^3
        salida[a:a + bloque] = dB.sum(axis=1)
    return MU0 * I / (4 * np.pi) * salida


def campo_solenoide(X, Y, I, largo, radio, vueltas, xc=0.0, z=0.0):
    """B (Bx, By) en el plano z = 'z' para un solenoide de 'vueltas' espiras repartidas en 'largo'."""
    if vueltas < 1:
        raise ErrorCampo("El número de vueltas debe ser al menos 1.")
    # Cada espira ocupa una franja de ancho largo/vueltas y se coloca en su centro,
    # así la densidad de espiras es exactamente n = vueltas/largo.
    xs = [xc] if vueltas == 1 else xc - largo / 2 + (np.arange(vueltas) + 0.5) * largo / vueltas
    pos, dl = zip(*[_espira_segmentos(x, radio) for x in xs])
    pos, dl = np.concatenate(pos), np.concatenate(dl)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    B = biot_savart(pts, pos, dl, I)
    return B[:, 0].reshape(X.shape), B[:, 1].reshape(X.shape)


# ----------------------------------------------------------- interfaz común
def calcular_campo(tipo, p, X, Y):
    """Devuelve (Bx, By). 'p' es un diccionario con los parámetros (SI)."""
    I = p["I"]
    if tipo == "Hilo recto":
        return campo_hilo(X, Y, I)
    if tipo == "Dos hilos paralelos":
        d = p["d"] / 2
        I2 = -I if p.get("opuestas") else I
        b1 = campo_hilo(X, Y, I, -d, 0.0)
        b2 = campo_hilo(X, Y, I2, d, 0.0)
        return b1[0] + b2[0], b1[1] + b2[1]
    if tipo == "Espira circular":
        return campo_solenoide(X, Y, I, 0.0, p["radio"], 1)
    if tipo == "Solenoide":
        return campo_solenoide(X, Y, I, p["largo"], p["radio"], int(p["vueltas"]))
    if tipo == "Imán de barra":
        # El imán equivale a un solenoide corto con 25 espiras de corriente.
        return campo_solenoide(X, Y, I, p["largo"], p["radio"], 25)
    raise ErrorCampo(f"Tipo desconocido: {tipo}")


def campo_en_punto(tipo, p, x, y):
    bx, by = calcular_campo(tipo, p, np.array([[x]]), np.array([[y]]))
    return float(bx[0, 0]), float(by[0, 0])


def fuerza_entre_hilos(I1, I2, d):
    """Fuerza por metro (N/m). Positiva = atracción (corrientes del mismo sentido)."""
    if d <= 0:
        raise ErrorCampo("La separación entre hilos debe ser mayor que 0.")
    return MU0 * I1 * I2 / (2 * np.pi * d)


def formato_campo(b):
    """Texto de B con la unidad más cómoda (T, mT, µT, nT)."""
    a = abs(b)
    if a >= 1:
        return f"{b:.3g} T"
    if a >= 1e-3:
        return f"{b * 1e3:.3g} mT"
    if a >= 1e-6:
        return f"{b * 1e6:.3g} µT"
    return f"{b * 1e9:.3g} nT"
