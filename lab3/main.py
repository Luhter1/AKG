import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

"""
Программа вычисляет распределение освещённости на горизонтальной плоскости (z = 0)
от ламбертовского точечного излучателя, находящегося выше плоскости.

Диапазоны (рекомендуемые):
- area_w, area_h: 100..10000 мм
- grid_w, grid_h: 200..800 пикселей
- src_x, src_y: ±10000 мм
- src_z: 100..10000 мм
- I0: 0.01..10000 Вт/ср
"""

def calc_light_map(
        area_w, area_h, grid_w, grid_h,
        src_x, src_y, src_z, intensity0,
        roi_x, roi_y, roi_r):
    """
    Основная вычислительная функция. Создаёт равномерную сетку,
    считает освещённость от точечного ламбертовского источника и
    формирует нормированное изображение.

    Параметры:
        area_w, area_h      размеры области (мм)
        grid_w, grid_h      разрешение по X, Y
        src_x, src_y, src_z положение источника (мм)
        intensity0          I0 (Вт/ср)
        roi_x, roi_y, roi_r область интереса (нормировка)

    Возвращает словарь:
        {
            "X", "Y"      сетка координат,
            "E"           матрица освещённости,
            "P"           нормированная 8-бит картинка,
            "mask"        булева маска объекта интереса,
            "gw", "gh"    фактическое разрешение сетки
        }
    """

    area_w = float(area_w)
    area_h = float(area_h)
    grid_w = int(grid_w)
    grid_h = int(grid_h)
    src_x = float(src_x)
    src_y = float(src_y)
    src_z = float(src_z)
    intensity0 = float(intensity0)
    roi_x = float(roi_x)
    roi_y = float(roi_y)
    roi_r = float(roi_r)

    # Базовые проверки
    if area_w <= 0 or area_h <= 0:
        raise ValueError("Размеры области должны быть положительными.")
    if grid_w <= 1 or grid_h <= 1:
        raise ValueError("Разрешение по осям должно быть > 1.")
    if src_z <= 0:
        raise ValueError("Высота источника должна быть > 0.")

    # Проверка однообразности пикселей
    step_x = area_w / grid_w
    step_y = area_h / grid_h

    # Если пиксели заметно различаются – корректируем
    if abs(step_x - step_y) > 0.01:
        cell = max(step_x, step_y)
        grid_w = max(2, int(area_w / cell))
        grid_h = max(2, int(area_h / cell))

    # Формирование координатной сетки
    x_vals = np.linspace(-area_w / 2, area_w / 2, grid_w)
    y_vals = np.linspace(-area_h / 2, area_h / 2, grid_h)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Разница координат от источника до точек
    dx = X - src_x
    dy = Y - src_y
    dz = -src_z    # плоскость z=0 → источник выше

    # Расстояние до источника
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)

    # Косинус угла между вектором и осью -Z
    cos_theta = (-dz) / r
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    # Физическая формула освещённости
    # Защита от деления на ноль
    eps = 1e-12
    E = intensity0 * (src_z ** 2) / np.maximum(r2 * r2, eps)

    # Тени (cosθ <= 0)
    E[cos_theta <= 0] = 0.0

    # Маска области интереса
    if roi_r > 0:
        mask = (X - roi_x) ** 2 + (Y - roi_y) ** 2 <= roi_r ** 2
    else:
        mask = np.ones_like(E, dtype=bool)

    # Нормировка
    Emax = E[mask].max() if np.any(mask) else 0.0

    if Emax > 0:
        P = (255 * E / Emax).clip(0, 255).astype(np.uint8)
    else:
        P = np.zeros_like(E, dtype=np.uint8)

    return {
        "X": X,
        "Y": Y,
        "E": E,
        "P": P,
        "mask": mask,
        "gw": grid_w,
        "gh": grid_h
    }

class LightGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Моделирование освещённости от точечного источника")
        self.geometry("780x680")
        self.resizable(True, True)

        # Словарь полей ввода
        self.fields = {}

        # Создание интерфейса
        self.build_interface()

    def build_interface(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        grid = ttk.Frame(frm)
        grid.pack(anchor="w")

        # Значения по умолчанию
        defaults = {
            "Ширина W (мм)": 2000,
            "Высота H (мм)": 2000,
            "Сетка Wres": 400,
            "Сетка Hres": 400,
            "Источник X": 300,
            "Источник Y": 200,
            "Источник Z": 700,
            "Интенсивность I0": 200,
            "ROI центр X": 0,
            "ROI центр Y": 0,
            "ROI радиус": 800,
            "PNG файл": "result.png"
        }

        # Создание строк ввода
        r = 0
        for label, val in defaults.items():
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=5, pady=4)
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(grid, textvariable=var, width=22)
            ent.grid(row=r, column=1, sticky="w")
            self.fields[label] = var
            r += 1

        # Кнопки
        buttons = ttk.Frame(frm)
        buttons.pack(pady=10, anchor="w")

        ttk.Button(buttons, text="Выбрать файл...",
                   command=lambda: self.pickfile("PNG файл")).pack(side="left", padx=6)
        ttk.Button(buttons, text="Показать карту",
                   command=self.show_map).pack(side="left", padx=6)
        ttk.Button(buttons, text="Сечения",
                   command=self.show_sections).pack(side="left", padx=6)
        ttk.Button(buttons, text="Сохранить PNG",
                   command=self.save_png).pack(side="left", padx=6)
        ttk.Button(buttons, text="Сохранить сечения",
                   command=self.save_sections).pack(side="left", padx=6)

    def pickfile(self, key):
        """Выбор файла для сохранения"""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG файл", "*.png"), ("Все файлы", "*.*")]
        )
        if path:
            self.fields["PNG файл"].set(path)

    def read_inputs(self):
        """Читает данные из полей ввода и возвращает кортеж"""
        try:
            W = float(self.fields["Ширина W (мм)"].get())
            H = float(self.fields["Высота H (мм)"].get())
            Wres = int(self.fields["Сетка Wres"].get())
            Hres = int(self.fields["Сетка Hres"].get())
            lx = float(self.fields["Источник X"].get())
            ly = float(self.fields["Источник Y"].get())
            lz = float(self.fields["Источник Z"].get())
            I0 = float(self.fields["Интенсивность I0"].get())
            cx = float(self.fields["ROI центр X"].get())
            cy = float(self.fields["ROI центр Y"].get())
            cr = float(self.fields["ROI радиус"].get())
            fname = self.fields["PNG файл"].get()
            return W, H, Wres, Hres, lx, ly, lz, I0, cx, cy, cr, fname
        except Exception as e:
            raise ValueError(f"Ошибка ввода данных: {e}")

    def compute(self):
        """Выполняет расчёт и возвращает результат"""
        params = self.read_inputs()
        res = calc_light_map(*params[:-1])
        # обновление разрешения при автокоррекции
        self.fields["Сетка Wres"].set(str(res["gw"]))
        self.fields["Сетка Hres"].set(str(res["gh"]))
        return res

    def show_map(self):
        """Отображает изображение нормированной освещённости"""
        try:
            data = self.compute()
            X, Y, P = data["X"], data["Y"], data["P"]

            plt.figure()
            plt.imshow(P, origin="lower",
                       extent=[X.min(), X.max(), Y.min(), Y.max()],
                       aspect="equal")
            plt.colorbar(label="Отн. уровни (0–255)")
            plt.title("Карта нормированной освещённости")
            plt.xlabel("x, мм")
            plt.ylabel("y, мм")
            plt.show()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_sections(self):
        """Строит графики горизонтального и вертикального срезов"""
        try:
            data = self.compute()
            X, Y, E, mask = data["X"], data["Y"], data["E"], data["mask"]
            W, H = X.shape[1], Y.shape[0]

            # Нормировка
            Emax = E[mask].max() if np.any(mask) else 1.0
            E_norm = E / Emax

            # Горизонтальная линия (y=ближайшее к ROI центру)
            cy = float(self.fields["ROI центр Y"].get())
            y_idx = np.argmin(np.abs(Y[:, 0] - cy))

            plt.figure()
            plt.plot(X[y_idx, :], E_norm[y_idx, :])
            plt.title(f"Горизонтальный срез (y={cy})")
            plt.xlabel("x, мм")
            plt.ylabel("E/Emax")
            plt.grid(True)
            plt.show()

            # Вертикальная линия (x=ROI центр)
            cx = float(self.fields["ROI центр X"].get())
            x_idx = np.argmin(np.abs(X[0, :] - cx))

            plt.figure()
            plt.plot(Y[:, x_idx], E_norm[:, x_idx])
            plt.title(f"Вертикальный срез (x={cx})")
            plt.xlabel("y, мм")
            plt.ylabel("E/Emax")
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_png(self):
        """Сохраняет нормированную карту освещённости"""
        try:
            data = self.compute()
            out = self.fields["PNG файл"].get()
            Image.fromarray(data["P"], mode="L").save(out)
            messagebox.showinfo("Готово", f"Сохранено: {out}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_sections(self):
        """Сохраняет графики срезов в PNG"""
        try:
            data = self.compute()
            X, Y, E, mask = data["X"], data["Y"], data["E"], data["mask"]
            fname = self.fields["PNG файл"].get()

            # Базовое имя без расширения
            base, _ = os.path.splitext(fname)

            # Нормировка
            Emax = E[mask].max() if np.any(mask) else 1.0
            E_norm = E / Emax

            # Горизонтальный
            cy = float(self.fields["ROI центр Y"].get())
            idx_y = np.argmin(np.abs(Y[:, 0] - cy))
            hor_file = base + "_horz.png"

            plt.figure()
            plt.plot(X[idx_y, :], E_norm[idx_y, :])
            plt.title(f"Горизонтальный срез (y={cy})")
            plt.xlabel("x, мм")
            plt.ylabel("E/Emax")
            plt.grid(True)
            plt.savefig(hor_file, bbox_inches="tight")
            plt.close()

            # Вертикальный
            cx = float(self.fields["ROI центр X"].get())
            idx_x = np.argmin(np.abs(X[0, :] - cx))
            ver_file = base + "_vert.png"

            plt.figure()
            plt.plot(Y[:, idx_x], E_norm[:, idx_x])
            plt.title(f"Вертикальный срез (x={cx})")
            plt.xlabel("y, мм")
            plt.ylabel("E/Emax")
            plt.grid(True)
            plt.savefig(ver_file, bbox_inches="tight")
            plt.close()

            messagebox.showinfo("Готово",
                                f"Файлы сохранены:\n{hor_file}\n{ver_file}")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    LightGUI().mainloop()
