import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
Программа вычисляет распределение освещённости на горизонтальной плоскости (z = 0)
от ламбертовского точечного излучателя, находящегося выше плоскости.

Диапазоны (рекомендуемые):
- area_w, area_h: 100..10000 мм
- grid_w, Hres: 200..800 пикселей
- src_x, src_y: ±10000 мм
- src_z: 100..10000 мм
- I0: 0.01..10000 Вт/ср
"""

def compute_illuminance(area_w, area_h, grid_w, grid_h, src_x, src_y, src_z, I0, xc, yc, R):
    """
    Основная вычислительная функция. Создаёт равномерную сетку,
    считает освещённость от точечного ламбертовского источника и
    формирует нормированное изображение.

    Параметры:
        area_w, area_h       размеры области (мм)
        grid_w, grid_h       разрешение по X, Y
        src_x, src_y, src_z  положение источника (мм)
        intensity0           I0
        xc, yc, R            область интереса (нормировка)

    Возвращает словарь:
        {
            "X", "Y"       сетка координат,
            "E"            матрица освещённости,
            "P"            нормированная 8-бит картинка,
            "mask"         булева маска объекта интереса,
            "gw", "gh"     фактическое разрешение сетки
        }
    """

    area_w = float(area_w)
    area_h = float(area_h)
    grid_w = int(grid_w)
    Hres = int(grid_h)
    src_x = float(src_x)
    src_y = float(src_y)
    src_z = float(src_z)
    I0 = float(I0)
    xc = float(xc)
    yc = float(yc)
    R = float(R)

    if area_w <= 0 or area_h <= 0: raise ValueError("area_w и area_h должны быть > 0")
    if grid_w <= 1 or Hres <= 1: raise ValueError("grid_w и Hres должны быть > 1")
    if src_z <= 0: raise ValueError("src_z должен быть > 0")

    pixel_size_x = area_w / grid_w
    pixel_size_y = area_h / Hres

    if abs(pixel_size_x - pixel_size_y) > 0.01:
        pixel_size = max(pixel_size_x, pixel_size_y)
        grid_w = int(area_w / pixel_size)
        Hres = int(area_h / pixel_size)
        if grid_w < 1: grid_w = 1
        if Hres < 1: Hres = 1

    if grid_w <= 1 or Hres <= 1: raise ValueError("После корректировки grid_w и Hres должны быть > 1")

    x = np.linspace(-area_w/2, area_w/2, grid_w)
    y = np.linspace(-area_h/2, area_h/2, Hres)
    X, Y = np.meshgrid(x, y)
    dx = X - src_x
    dy = Y - src_y
    dz = 0.0 - src_z
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)
    cos_theta = (-dz) / r
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    eps = 1e-12
    E = I0 * (src_z**2) / np.maximum(r2*r2, eps)
    E[cos_theta <= 0] = 0.0

    mask = (X - xc)**2 + (Y - yc)**2 <= R**2 if R > 0 else np.ones_like(E, dtype=bool)
    E[~mask] = 0.0
    Emax = E[mask].max() if mask.any() else 0.0
    scaled = (np.rint(255.0 * (E / Emax)).astype(np.uint8) if Emax > 0 else np.zeros_like(E, dtype=np.uint8))
    return X, Y, E, scaled, mask, grid_w, Hres

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Освещённость от ламбертовского источника")
        self.geometry("720x640")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        entries = {}
        defaults = {
            "area_w (мм)": 2000, "area_h (мм)": 2000, "grid_w (пкс)": 400, "Hres (пкс)": 400,
            "src_x (мм)": 400, "src_y (мм)": 200, "src_z (мм)": 800, "I0 (Вт/ср)": 250,
            "xc (мм)": 0, "yc (мм)": 0, "R (мм)": 900,
            "Имя файла PNG": "illuminance.png"
        }

        grid = ttk.Frame(frm)
        grid.pack(fill="x", pady=(0,10))

        r = 0
        for label, val in defaults.items():
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=(0,8), pady=4)
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(grid, textvariable=var, width=20)
            ent.grid(row=r, column=1, sticky="w")
            entries[label] = var
            r += 1

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Обзор...", command=lambda:self._pick_file(entries["Имя файла PNG"])).pack(side="left", padx=(0,8))
        ttk.Button(btns, text="Рассчитать и показать", command=lambda:self._run(entries)).pack(side="left")
        ttk.Button(btns, text="Показать сечения (гориз. и верт.)", command=lambda:self._show_sections(entries)).pack(side="left", padx=8)
        ttk.Button(btns, text="Сохранить только PNG", command=lambda:self._save(entries)).pack(side="left", padx=8)
        ttk.Button(btns, text="Сохранить сечения PNG", command=lambda:self._save_sections(entries)).pack(side="left", padx=8)

        info = ttk.Label(frm, foreground="#555",
                         text="Совет: используйте одинаковые grid_w и Hres для квадратных пикселей. "
                              "Сечение строится по линии y=0 через центр области.")
        info.pack(fill="x")

    def _pick_file(self, var):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"), ("All","*.*")])
        if path:
            var.set(path)

    def _parse(self, entries):
        try:
            area_w = float(entries["area_w (мм)"].get()); area_h = float(entries["area_h (мм)"].get())
            grid_w = int(entries["grid_w (пкс)"].get()); Hres = int(entries["Hres (пкс)"].get())
            src_x = float(entries["src_x (мм)"].get()); src_y = float(entries["src_y (мм)"].get()); src_z = float(entries["src_z (мм)"].get())
            I0 = float(entries["I0 (Вт/ср)"].get())
            xc = float(entries["xc (мм)"].get()); yc = float(entries["yc (мм)"].get()); R = float(entries["R (мм)"].get())
            fname = entries["Имя файла PNG"].get()
            return (area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R,fname)
        except Exception as e:
            raise ValueError(f"Ошибка ввода: {e}")

    def _run(self, entries):
        try:
            area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R,fname = self._parse(entries)
            X, Y, E, scaled, mask, Wres_corr, Hres_corr = compute_illuminance(area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R)
            if Wres_corr != grid_w or Hres_corr != Hres:
                entries["grid_w (пкс)"].set(str(Wres_corr))
                entries["Hres (пкс)"].set(str(Hres_corr))
            # Показ изображения
            plt.figure()
            plt.imshow(scaled, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='equal')
            plt.title("Нормированная освещённость (0–255)")
            plt.xlabel("x, мм")
            plt.ylabel("y, мм")
            plt.show()
            # Сечение по y=0
            y_idx = np.abs(Y[:,0] - 0.0).argmin()
            Emax = E[mask].max() if mask.any() else (E.max() if E.size else 1.0)
            norm_row = (E[y_idx, :] / Emax) if Emax > 0 else np.zeros_like(E[y_idx, :])
            plt.figure()
            plt.plot(X[y_idx, :], norm_row)
            plt.title("Сечение по центру области (y=0)")
            plt.xlabel("x, мм")
            plt.ylabel("E (норм., 0–1)")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


    def _show_sections(self, entries):
        try:
            area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R,fname = self._parse(entries)
            X, Y, E, scaled, mask, Wres_corr, Hres_corr = compute_illuminance(area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R)
            if Wres_corr != grid_w or Hres_corr != Hres:
                entries["grid_w (пкс)"].set(str(Wres_corr))
                entries["Hres (пкс)"].set(str(Hres_corr))
            # нормировочный максимум по области интереса
            Emax = E[mask].max() if mask.any() else (E.max() if E.size else 1.0)
            # Горизонтальное сечение y=yc
            y_idx = (abs(Y[:,0] - yc)).argmin()
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(X[y_idx,:], (E[y_idx,:]/Emax) if Emax>0 else E[y_idx,:]*0)
            plt.title(f"Горизонтальное сечение через центр области: y = {yc} мм")
            plt.xlabel("x, мм"); plt.ylabel("E (норм., 0–1)"); plt.grid(True)
            plt.show()
            # Вертикальное сечение x=xc
            x_idx = (abs(X[0,:] - xc)).argmin()
            plt.figure()
            plt.plot(Y[:,x_idx], (E[:,x_idx]/Emax) if Emax>0 else E[:,x_idx]*0)
            plt.title(f"Вертикальное сечение через центр области: x = {xc} мм")
            plt.xlabel("y, мм"); plt.ylabel("E (норм., 0–1)"); plt.grid(True)
            plt.show()
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Ошибка", str(e))

    def _save(self, entries):

        try:
            area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R,fname = self._parse(entries)
            _, _, _, scaled, _, Wres_corr, Hres_corr = compute_illuminance(area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R)
            if Wres_corr != grid_w or Hres_corr != Hres:
                entries["grid_w (пкс)"].set(str(Wres_corr))
                entries["Hres (пкс)"].set(str(Hres_corr))
            Image.fromarray(scaled, mode="L").save(fname)
            messagebox.showinfo("Готово", f"PNG сохранён: {fname}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _save_sections(self, entries):
        try:
            area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R,fname = self._parse(entries)
            X, Y, E, scaled, mask, Wres_corr, Hres_corr = compute_illuminance(area_w,area_h,grid_w,Hres,src_x,src_y,src_z,I0,xc,yc,R)
            if Wres_corr != grid_w or Hres_corr != Hres:
                entries["grid_w (пкс)"].set(str(Wres_corr))
                entries["Hres (пкс)"].set(str(Hres_corr))
            Emax = E[mask].max() if mask.any() else (E.max() if E.size else 1.0)
            import os
            base, ext = os.path.splitext(fname)
            hor = base + "_horz.png"
            ver = base + "_vert.png"
            import matplotlib.pyplot as plt
            # Горизонтальное
            y_idx = (abs(Y[:,0] - yc)).argmin()
            plt.figure()
            plt.plot(X[y_idx,:], (E[y_idx,:]/Emax) if Emax>0 else E[y_idx,:]*0)
            plt.title(f"Горизонтальное сечение через центр области: y = {yc} мм")
            plt.xlabel("x, мм"); plt.ylabel("E (норм., 0–1)"); plt.grid(True)
            plt.savefig(hor, bbox_inches="tight")
            plt.close()
            # Вертикальное
            x_idx = (abs(X[0,:] - xc)).argmin()
            plt.figure()
            plt.plot(Y[:,x_idx], (E[:,x_idx]/Emax) if Emax>0 else E[:,x_idx]*0)
            plt.title(f"Вертикальное сечение через центр области: x = {xc} мм")
            plt.xlabel("y, мм"); plt.ylabel("E (норм., 0–1)"); plt.grid(True)
            plt.savefig(ver, bbox_inches="tight")
            plt.close()
            from tkinter import messagebox
            messagebox.showinfo("Готово", f"Сечения сохранены:\n{hor}\n{ver}")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    App().mainloop()