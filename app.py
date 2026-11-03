"""
app.py - Simulador de la Unidad 4: Fuerza y Campo Magnético.
Interfaz con Tkinter + Matplotlib (sin HTML ni CSS).

Pestañas:
  1. Campo magnético  (imán, solenoide, espira, hilo, dos hilos)
  2. Fuerza de Lorentz (partícula en campos E y B)

Ejecutar:  python app.py
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import campo_magnetico as cm
import lorentz as lz

FUENTE_TITULO = ("TkDefaultFont", 10, "bold")
COLOR_OK = "#1e8449"
COLOR_ERROR = "#c0392b"


def numero(texto, nombre):
    """Convierte un texto en número (acepta coma decimal y notación 1e6)."""
    try:
        return float(str(texto).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"«{texto}» no es un número válido en {nombre}.")


# =====================================================================
# Pestaña 1: campo magnético
# =====================================================================
# (clave, etiqueta, mínimo, máximo, valor inicial, factor para pasar a SI)
PARAMETROS = {
    "Imán de barra": [("I", "Intensidad del imán (A equiv.)", -20, 20, 8, 1),
                      ("largo", "Largo (cm)", 4, 20, 10, 0.01),
                      ("radio", "Radio (cm)", 1, 5, 2, 0.01)],
    "Solenoide": [("I", "Corriente I (A)", -10, 10, 3, 1),
                  ("largo", "Largo (cm)", 5, 30, 15, 0.01),
                  ("radio", "Radio (cm)", 1, 5, 2, 0.01),
                  ("vueltas", "Número de vueltas", 2, 40, 12, 1)],
    "Espira circular": [("I", "Corriente I (A)", -20, 20, 5, 1),
                        ("radio", "Radio (cm)", 1, 10, 4, 0.01)],
    "Hilo recto": [("I", "Corriente I (A)", -30, 30, 10, 1)],
    "Dos hilos paralelos": [("I", "Corriente I (A)", -30, 30, 10, 1),
                            ("d", "Separación d (cm)", 2, 15, 6, 0.01)],
}
AYUDA = {
    "Imán de barra": "Imán modelado como un solenoide corto (modelo de Ampère). Polo N (rojo) a la derecha si I > 0.",
    "Solenoide": "Biot–Savart numérico: B ≈ μ0·n·I en el interior si el solenoide es largo.",
    "Espira circular": "Biot–Savart numérico. En el centro: B = μ0·I / (2a).",
    "Hilo recto": "Hilo infinito perpendicular a la pantalla: B = μ0·I / (2π·r). I > 0 sale de la pantalla (•).",
    "Dos hilos paralelos": "Superposición de dos hilos. Misma dirección → se atraen; opuestas → se repelen.",
}


class PestanaCampo(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6)
        self.vars = {}          # clave -> DoubleVar
        self.etiquetas = {}
        self.marca = None       # último punto medido con el ratón
        self._trabajo = None

        izq = ttk.Frame(self)
        izq.pack(side="left", fill="y", padx=(0, 8))

        caja = ttk.LabelFrame(izq, text="1. ¿Qué fuente de campo?", padding=6)
        caja.pack(fill="x")
        self.tipo = tk.StringVar(value=cm.TIPOS[0])
        for t in cm.TIPOS:
            ttk.Radiobutton(caja, text=t, value=t, variable=self.tipo,
                            command=self.cambiar_tipo).pack(anchor="w")

        self.caja_param = ttk.LabelFrame(izq, text="2. Ajusta los valores", padding=6)
        self.caja_param.pack(fill="x", pady=6)

        self.opuestas = tk.BooleanVar(value=False)
        self.chk = ttk.Checkbutton(izq, text="Corrientes en sentidos opuestos",
                                   variable=self.opuestas, command=self.programar)
        self.var_ayuda = tk.StringVar()
        ttk.Label(izq, textvariable=self.var_ayuda, wraplength=270, foreground="#555").pack(anchor="w", pady=4)
        ttk.Label(izq, text="Resultados", font=FUENTE_TITULO).pack(anchor="w", pady=(8, 0))
        self.var_info = tk.StringVar()
        ttk.Label(izq, textvariable=self.var_info, wraplength=270, foreground=COLOR_OK,
                  justify="left").pack(anchor="w")
        ttk.Label(izq, text="Haz clic sobre el gráfico para medir B en ese punto.",
                  wraplength=270, foreground="#555").pack(anchor="w", pady=(10, 0))
        self.var_clic = tk.StringVar()
        ttk.Label(izq, textvariable=self.var_clic, wraplength=270, foreground="#1f4e79",
                  justify="left").pack(anchor="w")

        der = ttk.Frame(self)
        der.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(7.5, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.al_hacer_clic)

        self.cambiar_tipo()

    # ------------------------------------------------------ controles
    def cambiar_tipo(self):
        for w in self.caja_param.winfo_children():
            w.destroy()
        self.vars, self.etiquetas, self.marca = {}, {}, None
        self.var_clic.set("")
        tipo = self.tipo.get()
        for clave, etiqueta, mn, mx, ini, _ in PARAMETROS[tipo]:
            var = tk.DoubleVar(value=ini)
            self.vars[clave] = var
            lbl = ttk.Label(self.caja_param)
            lbl.pack(anchor="w")
            self.etiquetas[clave] = (lbl, etiqueta)
            ttk.Scale(self.caja_param, from_=mn, to=mx, variable=var, length=250,
                      command=lambda _v: self.programar()).pack(anchor="w", pady=(0, 6))
        if tipo == "Dos hilos paralelos":
            self.chk.pack(anchor="w")
        else:
            self.chk.pack_forget()
        self.var_ayuda.set(AYUDA[tipo])
        self.programar()

    def programar(self):
        """Espera unos milisegundos para no redibujar en cada micro-movimiento del deslizador."""
        if self._trabajo is not None:
            self.after_cancel(self._trabajo)
        self._trabajo = self.after(60, self.dibujar)

    def parametros(self):
        p = {}
        for clave, etiqueta, mn, mx, ini, factor in PARAMETROS[self.tipo.get()]:
            v = self.vars[clave].get()
            if clave in ("vueltas",) or (clave == "I"):
                v = round(v) if clave == "vueltas" else round(v, 1)
            else:
                v = round(v, 1)
            p[clave] = v * factor
            texto = f"{etiqueta} = {v:g}"
            self.etiquetas[clave][0].config(text=texto)
        p["opuestas"] = self.opuestas.get()
        return p

    # ------------------------------------------------------ dibujo
    def ventana(self, tipo, p):
        if tipo == "Hilo recto":
            wx = 0.10
        elif tipo == "Dos hilos paralelos":
            wx = max(0.10, 1.4 * p["d"] + 0.04)
        elif tipo == "Espira circular":
            wx = 3.2 * p["radio"]
        else:
            wx = 0.5 * p["largo"] + 3.2 * p["radio"] + 0.01
        return wx, 0.72 * wx

    def dibujar(self):
        self._trabajo = None
        tipo = self.tipo.get()
        p = self.parametros()
        wx, wy = self.ventana(tipo, p)
        X, Y = np.meshgrid(np.linspace(-wx, wx, 44), np.linspace(-wy, wy, 32))
        try:
            Bx, By = cm.calcular_campo(tipo, p, X, Y)
        except cm.ErrorCampo as e:
            self.var_info.set(str(e))
            return
        modulo = np.hypot(Bx, By)
        color = np.log10(np.maximum(modulo, 1e-12))
        # Evita que la singularidad junto a un hilo "aplaste" la escala de colores
        vmin, vmax = np.percentile(color, 5), np.percentile(color, 97)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.ax = ax
        fl = ax.streamplot(X, Y, Bx, By, color=color, cmap="plasma", density=1.5,
                           linewidth=1.3, arrowsize=1.1, norm=matplotlib.colors.Normalize(vmin, vmax))
        cb = self.fig.colorbar(fl.lines, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("log10 de |B| (T)  →  más intenso", fontsize=8)
        self.dibujar_fuente(ax, tipo, p, wx)
        ax.set_xlim(-wx, wx); ax.set_ylim(-wy, wy)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"Líneas de campo magnético: {tipo}")
        if self.marca:
            ax.plot(*self.marca, "c+", ms=14, mew=2)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.var_info.set(self.resumen(tipo, p))

    def dibujar_fuente(self, ax, tipo, p, wx):
        I = p["I"]
        def punto(x, y, saliendo, tam=13):
            ax.plot(x, y, "o", ms=tam, mfc="white", mec="k", zorder=5)
            ax.plot(x, y, ".", ms=tam / 3, color="k", zorder=6) if saliendo else \
                ax.plot(x, y, "x", ms=tam / 2, color="k", mew=1.5, zorder=6)
        if tipo == "Hilo recto":
            punto(0, 0, I > 0)
        elif tipo == "Dos hilos paralelos":
            d = p["d"] / 2
            punto(-d, 0, I > 0)
            punto(d, 0, (-I if p["opuestas"] else I) > 0)
        elif tipo == "Imán de barra":
            L, a = p["largo"], p["radio"]
            n_sur, n_nor = ("S", "N") if I >= 0 else ("N", "S")
            c_sur, c_nor = ("#2e86c1", "#c0392b") if I >= 0 else ("#c0392b", "#2e86c1")
            ax.add_patch(Rectangle((-L / 2, -a), L / 2, 2 * a, fc=c_sur, alpha=0.45, ec="k", zorder=4))
            ax.add_patch(Rectangle((0, -a), L / 2, 2 * a, fc=c_nor, alpha=0.45, ec="k", zorder=4))
            ax.text(-L / 4, 0, n_sur, ha="center", va="center", fontsize=16, weight="bold", zorder=7)
            ax.text(L / 4, 0, n_nor, ha="center", va="center", fontsize=16, weight="bold", zorder=7)
        else:   # espira o solenoide: cortes de las espiras (arriba sale •, abajo entra ×)
            a = p["radio"]
            if tipo == "Espira circular":
                xs, tam = [0.0], 12
            else:
                n = int(p["vueltas"])
                xs, tam = -p["largo"] / 2 + (np.arange(n) + 0.5) * p["largo"] / n, 8
            for x in xs:
                punto(x, a, I > 0, tam)
                punto(x, -a, I < 0, tam)

    def resumen(self, tipo, p):
        I = p["I"]
        if tipo == "Hilo recto":
            r = 0.05
            return (f"B a {r * 100:g} cm del hilo:\n  {cm.formato_campo(cm.MU0 * I / (2 * np.pi * r))}\n"
                    "(B = μ0·I / 2πr)")
        if tipo == "Dos hilos paralelos":
            I2 = -I if p["opuestas"] else I
            F = cm.fuerza_entre_hilos(I, I2, p["d"])
            tipo_f = "atracción" if F > 0 else "repulsión"
            return (f"Fuerza por metro entre los hilos:\n  F/L = μ0·I1·I2 / 2πd\n  = {abs(F) * 1e6:.4g} µN/m ({tipo_f})")
        bx, by = cm.campo_en_punto(tipo, p, 0.0, 0.0)
        txt = f"B en el centro: {cm.formato_campo(bx)}"
        if tipo == "Espira circular":
            txt += f"\nTeoría μ0·I/2a = {cm.formato_campo(cm.MU0 * I / (2 * p['radio']))}"
        elif tipo == "Solenoide":
            n = p["vueltas"] / p["largo"]
            txt += f"\nSolenoide largo μ0·n·I = {cm.formato_campo(cm.MU0 * n * I)}"
        return txt

    def al_hacer_clic(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        tipo, p = self.tipo.get(), self.parametros()
        bx, by = cm.campo_en_punto(tipo, p, ev.xdata, ev.ydata)
        mod = float(np.hypot(bx, by))
        ang = float(np.degrees(np.arctan2(by, bx)))
        self.var_clic.set(f"Punto ({ev.xdata * 100:.1f} cm, {ev.ydata * 100:.1f} cm)\n"
                          f"|B| = {cm.formato_campo(mod)}\nDirección: {ang:.0f}° respecto a +x")
        self.marca = (ev.xdata, ev.ydata)
        self.dibujar()


# =====================================================================
# Pestaña 2: fuerza de Lorentz
# =====================================================================
PARTICULAS = {"Electrón": (-1, lz.M_ELECTRON), "Protón": (1, lz.M_PROTON),
              "Partícula alfa": (2, lz.M_ALFA), "Otra (escribe q y m)": None}


class PestanaLorentz(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6)
        self.trayectoria = None
        self._anim = None

        izq = ttk.Frame(self)
        izq.pack(side="left", fill="y", padx=(0, 8))

        caja = ttk.LabelFrame(izq, text="1. Ejemplo", padding=6)
        caja.pack(fill="x")
        self.combo = ttk.Combobox(caja, values=list(lz.EJEMPLOS), state="readonly", width=38)
        self.combo.pack()
        self.combo.bind("<<ComboboxSelected>>", self.cargar_ejemplo)

        caja = ttk.LabelFrame(izq, text="2. Partícula", padding=6)
        caja.pack(fill="x", pady=4)
        self.part = ttk.Combobox(caja, values=list(PARTICULAS), state="readonly", width=22)
        self.part.grid(row=0, column=0, columnspan=2, sticky="w")
        self.part.bind("<<ComboboxSelected>>", self.elegir_particula)
        self.v_q, self.v_m = tk.StringVar(), tk.StringVar()
        ttk.Label(caja, text="Carga q (en múltiplos de e)").grid(row=1, column=0, sticky="w")
        ttk.Entry(caja, textvariable=self.v_q, width=11).grid(row=1, column=1, pady=1)
        ttk.Label(caja, text="Masa m (kg)").grid(row=2, column=0, sticky="w")
        ttk.Entry(caja, textvariable=self.v_m, width=11).grid(row=2, column=1, pady=1)

        self.vec = {}
        for clave, titulo in (("v", "3. Velocidad inicial v (m/s)"), ("E", "4. Campo eléctrico E (V/m)"),
                              ("B", "5. Campo magnético B (T)")):
            caja = ttk.LabelFrame(izq, text=titulo, padding=6)
            caja.pack(fill="x", pady=2)
            self.vec[clave] = []
            for k, eje in enumerate("xyz"):
                ttk.Label(caja, text=eje).grid(row=0, column=2 * k)
                var = tk.StringVar()
                ttk.Entry(caja, textvariable=var, width=9).grid(row=0, column=2 * k + 1, padx=(2, 8))
                self.vec[clave].append(var)

        caja = ttk.LabelFrame(izq, text="6. Tiempo total (s)", padding=6)
        caja.pack(fill="x", pady=2)
        self.v_t = tk.StringVar()
        ttk.Entry(caja, textvariable=self.v_t, width=12).pack(anchor="w")

        fb = ttk.Frame(izq); fb.pack(fill="x", pady=6)
        ttk.Button(fb, text="Simular", command=self.simular).pack(side="left", padx=(0, 4))
        ttk.Button(fb, text="Animar", command=self.animar).pack(side="left", padx=(0, 4))
        ttk.Button(fb, text="Detener", command=self.detener).pack(side="left")
        self.var_msg = tk.StringVar()
        self.lbl_msg = ttk.Label(izq, textvariable=self.var_msg, wraplength=300, foreground=COLOR_ERROR)
        self.lbl_msg.pack(anchor="w")

        der = ttk.Frame(self)
        der.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 5.2), dpi=100)
        self.var_info = tk.StringVar()
        # El cuadro de resultados se empaqueta primero (abajo) para que el gráfico no lo tape
        ttk.Label(der, textvariable=self.var_info, font=("TkFixedFont", 9), justify="left",
                  foreground="#1f4e79").pack(side="bottom", anchor="w", pady=4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.combo.current(0)
        self.cargar_ejemplo()

    # ------------------------------------------------------ entradas
    def elegir_particula(self, _e=None):
        datos = PARTICULAS[self.part.get()]
        if datos:
            self.v_q.set(f"{datos[0]:g}"); self.v_m.set(f"{datos[1]:.4g}")

    def cargar_ejemplo(self, _e=None):
        q, m, v0, E, B, t = lz.EJEMPLOS[self.combo.get()]
        self.v_q.set(f"{q:g}"); self.v_m.set(f"{m:.4g}")
        nombre = next((n for n, d in PARTICULAS.items() if d and d[0] == q and abs(d[1] - m) / m < 1e-3),
                      "Otra (escribe q y m)")
        self.part.set(nombre)
        for clave, valores in (("v", v0), ("E", E), ("B", B)):
            for var, val in zip(self.vec[clave], valores):
                var.set(f"{val:.4g}")
        self.v_t.set(f"{t:.4g}")
        self.simular()

    def leer(self):
        q = numero(self.v_q.get(), "la carga") * lz.E_CARGA
        m = numero(self.v_m.get(), "la masa")
        v0 = [numero(v.get(), "la velocidad") for v in self.vec["v"]]
        E = [numero(v.get(), "el campo E") for v in self.vec["E"]]
        B = [numero(v.get(), "el campo B") for v in self.vec["B"]]
        t = numero(self.v_t.get(), "el tiempo total")
        return q, m, v0, E, B, t

    # ------------------------------------------------------ simulación
    def simular(self):
        self.detener()
        try:
            q, m, v0, E, B, t = self.leer()
            tt, r, v = lz.simular(q, m, (0, 0, 0), v0, E, B, t)
        except ValueError as e:      # incluye ErrorLorentz
            self.var_msg.set("Error: " + str(e))
            return
        self.var_msg.set("")
        self.trayectoria = (tt, r, v)
        self.datos = (q, m, v0, E, B)
        # Unidad de longitud cómoda
        extension = max(np.ptp(r, axis=0).max(), 1e-12)
        self.unidad, self.factor = next((u, f) for u, f in (("m", 1), ("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9))
                                        if extension >= f or u == "nm")
        self.dibujar(len(r))
        self.var_info.set(self.informe(q, m, v0, E, B, r, v))

    def dibujar(self, hasta):
        tt, r, v = self.trayectoria
        R = r / self.factor
        u = self.unidad
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        a3 = self.fig.add_subplot(gs[:, 0], projection="3d")
        axy = self.fig.add_subplot(gs[0, 1]); axz = self.fig.add_subplot(gs[1, 1])
        centro = (R.max(axis=0) + R.min(axis=0)) / 2
        mitad = max(np.ptp(R, axis=0).max() / 2, 1e-12) * 1.1
        a3.set_xlim(centro[0] - mitad, centro[0] + mitad)
        a3.set_ylim(centro[1] - mitad, centro[1] + mitad)
        a3.set_zlim(centro[2] - mitad, centro[2] + mitad)
        a3.set_box_aspect((1, 1, 1))
        a3.set_xlabel(f"x ({u})"); a3.set_ylabel(f"y ({u})"); a3.set_zlabel(f"z ({u})")
        a3.set_title("Trayectoria 3D")
        self.l3, = a3.plot(R[:hasta, 0], R[:hasta, 1], R[:hasta, 2], color="#2471a3", lw=1.8)
        a3.plot([R[0, 0]], [R[0, 1]], [R[0, 2]], "o", color="green", label="inicio")
        self.p3, = a3.plot([R[hasta - 1, 0]], [R[hasta - 1, 1]], [R[hasta - 1, 2]], "o", color="red", label="fin")
        a3.legend(loc="upper left", fontsize=8)
        self.lineas2d = []
        for ax, i, j, tit in ((axy, 0, 1, "Vista XY"), (axz, 0, 2, "Vista XZ")):
            l, = ax.plot(R[:hasta, i], R[:hasta, j], color="#2471a3", lw=1.5)
            ax.plot(R[0, i], R[0, j], "o", color="green")
            p, = ax.plot(R[hasta - 1, i], R[hasta - 1, j], "o", color="red")
            ax.set_xlabel(f"{'xyz'[i]} ({u})"); ax.set_ylabel(f"{'xyz'[j]} ({u})")
            ax.set_title(tit, fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")
            self.lineas2d.append((l, p, i, j))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def animar(self):
        if not self.trayectoria:
            return
        self.detener()
        n = len(self.trayectoria[1])
        paso = max(1, n // 120)
        self._k = 0
        def cuadro():
            self._k = min(self._k + paso, n)
            R = self.trayectoria[1] / self.factor
            k = max(self._k, 2)
            self.l3.set_data_3d(R[:k, 0], R[:k, 1], R[:k, 2])
            self.p3.set_data_3d([R[k - 1, 0]], [R[k - 1, 1]], [R[k - 1, 2]])
            for l, p, i, j in self.lineas2d:
                l.set_data(R[:k, i], R[:k, j]); p.set_data([R[k - 1, i]], [R[k - 1, j]])
            self.canvas.draw_idle()
            self._anim = self.after(30, cuadro) if self._k < n else None
        cuadro()

    def detener(self):
        if self._anim is not None:
            self.after_cancel(self._anim)
            self._anim = None

    def informe(self, q, m, v0, E, B, r, v):
        a = lz.analisis(q, m, v0, E, B)
        lineas = []
        if "T" in a:
            lineas.append(f"Radio de giro r = m·v⊥/(|q|B) = {lz.formato_longitud(a['radio'])}    "
                          f"Periodo T = 2πm/(|q|B) = {a['T']:.3g} s    f = {a['f']:.3g} Hz")
            if abs(a["v_par"]) > 1e-9 * max(1.0, np.linalg.norm(v0)):
                lineas.append(f"Paso de la hélice = {lz.formato_longitud(a['paso'])}   (v∥ = {a['v_par']:.3g} m/s)")
            vd = a["v_deriva"]
            if np.linalg.norm(vd) > 0:
                lineas.append(f"Velocidad de deriva E×B/B² = ({vd[0]:.3g}, {vd[1]:.3g}, {vd[2]:.3g}) m/s")
        else:
            ac = a["acel"]
            lineas.append(f"Sin campo B: aceleración a = qE/m = ({ac[0]:.3g}, {ac[1]:.3g}, {ac[2]:.3g}) m/s²")
        ec0 = 0.5 * m * float(v[0] @ v[0]); ec1 = 0.5 * m * float(v[-1] @ v[-1])
        lineas.append(f"Energía cinética: inicial {ec0:.4g} J → final {ec1:.4g} J"
                      + ("   (el campo magnético no hace trabajo)" if not np.any(E) else ""))
        return "\n".join(lineas)


# =====================================================================
# Ventana principal
# =====================================================================
class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador de Campo Magnético y Fuerza de Lorentz - Unidad 4")
        self.geometry("1150x720")
        libro = ttk.Notebook(self)
        libro.pack(fill="both", expand=True)
        self.pestanas = (PestanaCampo(libro), PestanaLorentz(libro))
        libro.add(self.pestanas[0], text="Campo magnético")
        libro.add(self.pestanas[1], text="Fuerza de Lorentz")


def main():
    Aplicacion().mainloop()


if __name__ == "__main__":
    main()
