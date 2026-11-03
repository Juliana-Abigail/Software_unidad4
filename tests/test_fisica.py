"""Pruebas automáticas. Ejecutar:  python -m unittest discover -s tests -v"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import campo_magnetico as cm
import lorentz as lz

MU0 = cm.MU0


class PruebasCampo(unittest.TestCase):
    def test_hilo_formula(self):
        bx, by = cm.campo_hilo(np.array([[0.05]]), np.array([[0.0]]), 10)
        self.assertAlmostEqual(np.hypot(bx, by)[0, 0], MU0 * 10 / (2 * np.pi * 0.05), places=12)

    def test_hilo_regla_mano_derecha(self):
        # I hacia +z: en (+x) el campo apunta a +y; en (+y) apunta a -x
        bx, by = cm.campo_hilo(np.array([[0.1]]), np.array([[0.0]]), 5)
        self.assertGreater(by[0, 0], 0); self.assertAlmostEqual(bx[0, 0], 0)
        bx, by = cm.campo_hilo(np.array([[0.0]]), np.array([[0.1]]), 5)
        self.assertLess(bx[0, 0], 0)

    def test_ley_de_ampere(self):
        """La circulación de B alrededor de un hilo debe valer mu0 * I."""
        I, R = 7.0, 0.03
        th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        X, Y = R * np.cos(th), R * np.sin(th)
        bx, by = cm.campo_hilo(X, Y, I)
        dl_x, dl_y = -R * np.sin(th) * (2 * np.pi / 2000), R * np.cos(th) * (2 * np.pi / 2000)
        self.assertAlmostEqual(np.sum(bx * dl_x + by * dl_y) / (MU0 * I), 1.0, places=6)

    def test_espira_eje(self):
        """Biot-Savart numérico vs B = mu0 I a² / (2 (a²+x²)^(3/2)) sobre el eje."""
        I, a = 3.0, 0.05
        for x in (0.0, 0.03, 0.1):
            bx, _ = cm.campo_en_punto("Espira circular", {"I": I, "radio": a}, x, 0.0)
            teorico = MU0 * I * a ** 2 / (2 * (a ** 2 + x ** 2) ** 1.5)
            self.assertAlmostEqual(bx / teorico, 1.0, places=3)

    def test_solenoide_largo(self):
        """En el centro de un solenoide largo, B ≈ mu0 n I."""
        p = {"I": 2.0, "largo": 1.0, "radio": 0.02, "vueltas": 100}
        bx, _ = cm.campo_en_punto("Solenoide", p, 0.0, 0.0)
        self.assertAlmostEqual(bx / (MU0 * 100 * 2.0 / 1.0), 1.0, delta=0.01)

    def test_iman_polo_norte(self):
        """Con I>0 el campo dentro del imán apunta a +x (polo norte a la derecha)."""
        bx, _ = cm.campo_en_punto("Imán de barra", {"I": 5, "largo": 0.08, "radio": 0.015}, 0.0, 0.0)
        self.assertGreater(bx, 0)

    def test_dos_hilos_superposicion(self):
        p = {"I": 4.0, "d": 0.1, "opuestas": False}
        bx, by = cm.campo_en_punto("Dos hilos paralelos", p, 0.0, 0.0)   # punto medio
        self.assertAlmostEqual(np.hypot(bx, by), 0.0, places=12)          # se anulan (mismo sentido)
        p["opuestas"] = True
        _, by = cm.campo_en_punto("Dos hilos paralelos", p, 0.0, 0.0)     # se suman
        self.assertAlmostEqual(abs(by), 2 * MU0 * 4 / (2 * np.pi * 0.05), places=10)

    def test_fuerza_entre_hilos(self):
        # Definición del amperio: 1 A y 1 A a 1 m -> 2e-7 N/m
        self.assertAlmostEqual(cm.fuerza_entre_hilos(1, 1, 1), 2e-7, places=15)
        with self.assertRaises(cm.ErrorCampo):
            cm.fuerza_entre_hilos(1, 1, 0)

    def test_errores(self):
        with self.assertRaises(cm.ErrorCampo):
            cm.calcular_campo("Solenoide", {"I": 1, "largo": .1, "radio": .01, "vueltas": 0},
                              np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaises(cm.ErrorCampo):
            cm.calcular_campo("Otro", {"I": 1}, np.zeros((2, 2)), np.zeros((2, 2)))


class PruebasLorentz(unittest.TestCase):
    def test_radio_y_periodo_ciclotron(self):
        q, m = -lz.E_CARGA, lz.M_ELECTRON
        v, B = 1e6, 1e-3
        a = lz.analisis(q, m, (v, 0, 0), (0, 0, 0), (0, 0, B))
        self.assertAlmostEqual(a["radio"], m * v / (abs(q) * B), places=12)
        self.assertAlmostEqual(a["T"], 2 * np.pi * m / (abs(q) * B), places=15)

    def test_trayectoria_circular(self):
        """Numérico vs teoría: el radio de la órbita y el cierre tras un periodo."""
        q, m, v, B = -lz.E_CARGA, lz.M_ELECTRON, 1e6, 1e-3
        T = 2 * np.pi * m / (abs(q) * B); R = m * v / (abs(q) * B)
        t, r, vel = lz.simular(q, m, (0, 0, 0), (v, 0, 0), (0, 0, 0), (0, 0, B), T)
        self.assertAlmostEqual((r[:, 1].max() - r[:, 1].min()) / 2 / R, 1.0, places=3)
        self.assertLess(np.linalg.norm(r[-1] - r[0]) / R, 1e-3)           # vuelve al inicio

    def test_conserva_energia_sin_E(self):
        q, m = lz.E_CARGA, lz.M_PROTON
        t, r, v = lz.simular(q, m, (0, 0, 0), (1e5, 2e4, 3e4), (0, 0, 0), (0.01, 0.02, 0.05), 1e-5)
        v2 = np.sum(v ** 2, axis=1)
        self.assertLess(np.ptp(v2) / v2[0], 1e-9)

    def test_helice_paso(self):
        q, m, B = -lz.E_CARGA, lz.M_ELECTRON, 1e-3
        a = lz.analisis(q, m, (1e6, 0, 5e5), (0, 0, 0), (0, 0, B))
        t, r, v = lz.simular(q, m, (0, 0, 0), (1e6, 0, 5e5), (0, 0, 0), (0, 0, B), a["T"])
        self.assertAlmostEqual(r[-1, 2] / a["paso"], 1.0, places=3)

    def test_selector_de_velocidades(self):
        """Con E = v B la partícula sigue en línea recta."""
        q, m, v, B = lz.E_CARGA, lz.M_PROTON, 1e5, 1e-2
        t, r, vel = lz.simular(q, m, (0, 0, 0), (v, 0, 0), (0, v * B, 0), (0, 0, B), 2e-6)
        self.assertLess(abs(r[:, 1]).max() / abs(r[-1, 0]), 1e-6)
        self.assertAlmostEqual(r[-1, 0], v * 2e-6, places=6)

    def test_deriva_ExB(self):
        q, m, E, B = lz.E_CARGA, lz.M_PROTON, 1e3, 1e-2
        T = 2 * np.pi * m / (q * B)
        t, r, v = lz.simular(q, m, (0, 0, 0), (0, 0, 0), (0, E, 0), (0, 0, B), 10 * T)
        self.assertAlmostEqual(r[-1, 0] / (10 * T) / (E / B), 1.0, places=3)  # v_deriva = E/B

    def test_solo_E_parabola(self):
        q, m, E = -lz.E_CARGA, lz.M_ELECTRON, 1e3
        t, r, v = lz.simular(q, m, (0, 0, 0), (1e6, 0, 0), (0, E, 0), (0, 0, 0), 2e-8)
        a = q * E / m
        self.assertAlmostEqual(r[-1, 1] / (0.5 * a * (2e-8) ** 2), 1.0, places=3)

    def test_errores(self):
        for kw in ({"m": 0}, {"q": 0}, {"t": 0}):
            with self.assertRaises(lz.ErrorLorentz):
                lz.simular(kw.get("q", 1e-19), kw.get("m", 1e-27), (0, 0, 0), (1, 0, 0),
                           (0, 0, 0), (0, 0, 1), kw.get("t", 1e-6))

    def test_ejemplos_corren(self):
        for q, m, v0, E, B, t in lz.EJEMPLOS.values():
            lz.simular(q * lz.E_CARGA, m, (0, 0, 0), v0, E, B, t)


if __name__ == "__main__":
    unittest.main()
