import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk



#   Класс точечного источника света
class PointLight:
    def __init__(self, x, y, z, intensity):
        # Позиция источника света в 3D
        self.position = np.array([x, y, z], dtype=float)
        # Сила излучения (Вт/ср)
        self.intensity = float(intensity)

    def get_direction_to(self, point):
        """Направление от источника к точке (нормализованный вектор)."""
        direction = point - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            return np.zeros(3)
        return direction / dist

    def get_distance_to(self, point):
        """Расстояние от источника до точки."""
        return np.linalg.norm(point - self.position)



#   Класс сферы — задаёт геометрию и пересечение с лучом
class Sphere:
    def __init__(self, cx, cy, cz, radius):
        self.center = np.array([cx, cy, cz], dtype=float)
        self.radius = float(radius)

    def get_normal(self, point):
        """Вычисляем нормаль поверхности как (P - С) / R."""
        normal = point - self.center
        n = np.linalg.norm(normal)
        if n < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return normal / n

    def intersect_ray(self, origin, direction):
        """
        Пересечение луча со сферой.
        Решаем квадратное уравнение.
        Возвращаем ближайшую положительную точку пересечения.
        """

        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius**2

        disc = b*b - 4*a*c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # выбираем ближайшее положительное решение
        t = min(t1, t2) if t1 > 0 else (t2 if t2 > 0 else None)
        if t is None:
            return None

        return origin + t * direction



#   Модель освещения Блинна–Фонга
class BlinnPhongModel:
    def __init__(self, ka, kd, ks, shininess):
        self.ka = float(ka)  # ambient
        self.kd = float(kd)  # diffuse
        self.ks = float(ks)  # specular
        self.n = float(shininess)

    def calculate_intensity(self, point, normal, view_dir, lights):
        """
        Основная формула Блинна–Фонга:
        I = ka + Σ ( kd*(N·L) + ks*(N·H)^n ) * atten
        где H = (L+V)/|L+V|
        """

        ambient = self.ka
        diffuse = 0.0
        specular = 0.0

        # нормализуем направление на наблюдателя
        v = view_dir / np.linalg.norm(view_dir)

        for light in lights:

            # направление: от точки к источнику
            L = light.get_direction_to(point)
            d = light.get_distance_to(point)

            if d < 1e-10:
                continue

            # вектор в сторону источника света
            L = -L  # инверсия (луч от точки к свету)

            # диффузная компонента (Ламберта)
            ndotl = max(0.0, np.dot(normal, L))

            # ослабление (1/r²) — реалистичное поведение
            atten = 1.0 / (d * d)

            # добавляем diffuse
            diffuse += light.intensity * self.kd * ndotl * atten

            # спекулярная компонента
            if ndotl > 0:
                H = v + L
                H_norm = np.linalg.norm(H)
                if H_norm > 1e-10:
                    H /= H_norm
                    ndoth = max(0.0, np.dot(normal, H))
                    specular += light.intensity * self.ks * (ndoth ** self.n) * atten

        return max(0.0, ambient + diffuse + specular)



#   Рендерер сцены: построение изображения сферы
class Renderer:
    def __init__(self, screen_w, screen_h, res_w, res_h,
                 observer_z, sphere, lights, material):

        self.screen_width = float(screen_w)
        self.screen_height = float(screen_h)
        self.res_w = int(res_w)
        self.res_h = int(res_h)

        # камера расположена по оси Z
        self.observer = np.array([0.0, 0.0, float(observer_z)])

        self.sphere = sphere
        self.lights = lights
        self.material = material

        # реальный размер пикселя в мм
        self.pixel_w = self.screen_width / self.res_w
        self.pixel_h = self.screen_height / self.res_h

    # ---------------------------------------------------------
    def render(self):
        """Основной рендер: трассировка лучей от камеры к экрану."""
        image = np.zeros((self.res_h, self.res_w), dtype=float)

        for y in range(self.res_h):
            for x in range(self.res_w):

                # координаты точки на виртуальном экране (в мм)
                sx = (x + 0.5) * self.pixel_w - self.screen_width / 2
                sy = -(y + 0.5) * self.pixel_h + self.screen_height / 2
                screen_point = np.array([sx, sy, 0.0])

                # луч: от наблюдателя к экрану
                direction = screen_point - self.observer
                direction /= np.linalg.norm(direction)

                # пересечение со сферой
                hit = self.sphere.intersect_ray(self.observer, direction)
                if hit is None:
                    continue

                # нормаль и направление взгляда
                normal = self.sphere.get_normal(hit)
                view_dir = -direction

                # интенсивность по Блинн–Фонгу
                image[y, x] = self.material.calculate_intensity(
                    hit, normal, view_dir, self.lights
                )

        # нормируем изображение в диапазон 0..255
        max_i = image.max()
        if max_i > 1e-10:
            img_norm = (image / max_i) * 255.0
        else:
            img_norm = np.zeros_like(image)

        return Image.fromarray(img_norm.astype(np.uint8), 'L')


class ParameterControl:
    """GUI-элемент: label + slider + entry"""
    def __init__(self, parent, label, min_val, max_val, default_val, 
                 resolution=1.0, callback=None):
        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') else '#ffffff'
        self.frame = tk.Frame(parent, bg=parent_bg)
        self.label = tk.Label(self.frame, text=label, width=18, anchor='w', 
                              font=('Segoe UI', 9), bg=parent_bg, fg='#2c3e50')
        self.label.pack(side='left', padx=5)
        
        self.var = tk.DoubleVar(value=default_val)
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        self.callback = callback
        
        self.scale = tk.Scale(
            self.frame,
            from_=min_val,
            to=max_val,
            resolution=resolution,
            orient='horizontal',
            length=200,
            variable=self.var,
            command=self._on_scale_change,
            bg='#ffffff',
            troughcolor='#e8e8e8',
            activebackground='#3498db',
            highlightbackground=parent_bg
        )
        self.scale.pack(side='left', padx=5)
        
        self.entry = tk.Entry(self.frame, width=12, font=('Consolas', 9), 
                             bg='#ffffff', fg='#2c3e50', relief='solid', 
                             borderwidth=1, insertbackground='#3498db')
        self.entry.insert(0, str(default_val))
        self.entry.pack(side='left', padx=5)
        self.entry.bind('<Return>', self._on_entry_change)
        self.entry.bind('<FocusOut>', self._on_entry_change)
        
        self._updating = False
    
    def _on_scale_change(self, value=None):
        if self._updating:
            return
        self._updating = True
        val = self.var.get()
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{val:.2f}")
        if self.callback:
            self.callback()
        self._updating = False
    
    def _on_entry_change(self, event=None):
        if self._updating:
            return
        try:
            val = float(self.entry.get())
            val = max(self.min_val, min(self.max_val, val))
            self._updating = True
            self.var.set(val)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{val:.2f}")
            if self.callback:
                self.callback()
            self._updating = False
        except ValueError:
            val = self.var.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{val:.2f}")
    
    def get(self):
        return self.var.get()
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class App:
    """Графический интерфейс приложения"""
    def __init__(self, master):
        self.master = master
        master.title("Расчет яркости на сфере")
        master.configure(bg='#ecf0f1')
        
        self.setup_ui()
        self.update_render()
    
    def setup_ui(self):
        main_frame = tk.Frame(self.master, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_container = tk.Frame(main_frame, bg='#ecf0f1')
        left_container.pack(side='left', fill='both', padx=5)
        
        canvas = tk.Canvas(left_container, width=400, bg='#ecf0f1', 
                          highlightthickness=0)
        scrollbar = tk.Scrollbar(left_container, orient="vertical", 
                                command=canvas.yview, bg='#bdc3c7', 
                                troughcolor='#ecf0f1', activebackground='#95a5a6')
        scrollable_frame = tk.Frame(canvas, bg='#ecf0f1')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        left_frame = scrollable_frame
        
        right_frame = tk.Frame(main_frame, bg='#ecf0f1')
        right_frame.pack(side='right', fill='both', padx=5)
        
        params_frame = tk.LabelFrame(left_frame, text="Параметры экрана", 
                                     font=('Segoe UI', 10, 'bold'), 
                                     fg='#2c3e50', bg='#ffffff', 
                                     relief='flat', bd=2, padx=5, pady=5)
        params_frame.pack(fill='x', pady=5)
        
        self.screen_width = ParameterControl(
            params_frame, "Ширина экрана (мм):", 100, 10000, 5000, 
            resolution=100, callback=self.update_render
        )
        self.screen_width.pack(fill='x', pady=2)
        
        self.screen_height = ParameterControl(
            params_frame, "Высота экрана (мм):", 100, 10000, 5000,
            resolution=100, callback=self.update_render
        )
        self.screen_height.pack(fill='x', pady=2)
        
        self.screen_w_res = ParameterControl(
            params_frame, "Разрешение по ширине:", 200, 800, 400,
            resolution=10, callback=self.update_render
        )
        self.screen_w_res.pack(fill='x', pady=2)
        
        self.screen_h_res = ParameterControl(
            params_frame, "Разрешение по высоте:", 200, 800, 400,
            resolution=10, callback=self.update_render
        )
        self.screen_h_res.pack(fill='x', pady=2)
        
        observer_frame = tk.LabelFrame(left_frame, text="Наблюдатель", 
                                      font=('Segoe UI', 10, 'bold'), 
                                      fg='#2c3e50', bg='#e8f5e9', 
                                      relief='flat', bd=2, padx=5, pady=5)
        observer_frame.pack(fill='x', pady=5)
        
        self.observer_z = ParameterControl(
            observer_frame, "Z наблюдателя (мм):", 100, 10000, 5000,
            resolution=100, callback=self.update_render
        )
        self.observer_z.pack(fill='x', pady=2)
        
        sphere_frame = tk.LabelFrame(left_frame, text="Сфера", 
                                     font=('Segoe UI', 10, 'bold'), 
                                     fg='#2c3e50', bg='#fff3e0', 
                                     relief='flat', bd=2, padx=5, pady=5)
        sphere_frame.pack(fill='x', pady=5)
        
        self.sphere_x = ParameterControl(
            sphere_frame, "X центра (мм):", -10000, 10000, 0,
            resolution=100, callback=self.update_render
        )
        self.sphere_x.pack(fill='x', pady=2)
        
        self.sphere_y = ParameterControl(
            sphere_frame, "Y центра (мм):", -10000, 10000, 0,
            resolution=100, callback=self.update_render
        )
        self.sphere_y.pack(fill='x', pady=2)
        
        self.sphere_z = ParameterControl(
            sphere_frame, "Z центра (мм):", 100, 10000, 3000,
            resolution=100, callback=self.update_render
        )
        self.sphere_z.pack(fill='x', pady=2)
        
        self.sphere_radius = ParameterControl(
            sphere_frame, "Радиус (мм):", 100, 5000, 1000,
            resolution=50, callback=self.update_render
        )
        self.sphere_radius.pack(fill='x', pady=2)
        
        light_frame = tk.LabelFrame(left_frame, text="Источник света 1", 
                                    font=('Segoe UI', 10, 'bold'), 
                                    fg='#2c3e50', bg='#fff9c4', 
                                    relief='flat', bd=2, padx=5, pady=5)
        light_frame.pack(fill='x', pady=5)
        
        self.light1_x = ParameterControl(
            light_frame, "X источника (мм):", -10000, 10000, 2000,
            resolution=100, callback=self.update_render
        )
        self.light1_x.pack(fill='x', pady=2)
        
        self.light1_y = ParameterControl(
            light_frame, "Y источника (мм):", -10000, 10000, 2000,
            resolution=100, callback=self.update_render
        )
        self.light1_y.pack(fill='x', pady=2)
        
        self.light1_z = ParameterControl(
            light_frame, "Z источника (мм):", 100, 10000, 4000,
            resolution=100, callback=self.update_render
        )
        self.light1_z.pack(fill='x', pady=2)
        
        self.light1_intensity = ParameterControl(
            light_frame, "Сила излучения (Вт/ср):", 0.01, 10000, 500,
            resolution=10, callback=self.update_render
        )
        self.light1_intensity.pack(fill='x', pady=2)
        
        light2_frame = tk.LabelFrame(left_frame, text="Источник света 2", 
                                     font=('Segoe UI', 10, 'bold'), 
                                     fg='#2c3e50', bg='#ffe0b2', 
                                     relief='flat', bd=2, padx=5, pady=5)
        light2_frame.pack(fill='x', pady=5)
        
        self.light2_x = ParameterControl(
            light2_frame, "X источника (мм):", -10000, 10000, -2000,
            resolution=100, callback=self.update_render
        )
        self.light2_x.pack(fill='x', pady=2)
        
        self.light2_y = ParameterControl(
            light2_frame, "Y источника (мм):", -10000, 10000, 2000,
            resolution=100, callback=self.update_render
        )
        self.light2_y.pack(fill='x', pady=2)
        
        self.light2_z = ParameterControl(
            light2_frame, "Z источника (мм):", 100, 10000, 4000,
            resolution=100, callback=self.update_render
        )
        self.light2_z.pack(fill='x', pady=2)
        
        self.light2_intensity = ParameterControl(
            light2_frame, "Сила излучения (Вт/ср):", 0.01, 10000, 500,
            resolution=10, callback=self.update_render
        )
        self.light2_intensity.pack(fill='x', pady=2)
        
        material_frame = tk.LabelFrame(left_frame, text="Модель Блинн-Фонга", 
                                       font=('Segoe UI', 10, 'bold'), 
                                       fg='#2c3e50', bg='#f3e5f5', 
                                       relief='flat', bd=2, padx=5, pady=5)
        material_frame.pack(fill='x', pady=5)
        
        self.ka = ParameterControl(
            material_frame, "ka (ambient):", 0.0, 1.0, 0.0,
            resolution=0.01, callback=self.update_render
        )
        self.ka.pack(fill='x', pady=2)
        
        self.kd = ParameterControl(
            material_frame, "kd (diffuse):", 0.0, 1.0, 0.5,
            resolution=0.01, callback=self.update_render
        )
        self.kd.pack(fill='x', pady=2)
        
        self.ks = ParameterControl(
            material_frame, "ks (specular):", 0.0, 1.0, 0.8,
            resolution=0.01, callback=self.update_render
        )
        self.ks.pack(fill='x', pady=2)
        
        self.n = ParameterControl(
            material_frame, "n (shininess):", 1.0, 200.0, 100.0,
            resolution=1.0, callback=self.update_render
        )
        self.n.pack(fill='x', pady=2)
        
        button_frame = tk.Frame(left_frame, bg='#ecf0f1')
        button_frame.pack(fill='x', pady=10)
        
        save_btn = tk.Button(
            button_frame, text="💾 Сохранить изображение",
            command=self.save_image, width=25,
            font=('Segoe UI', 11, 'bold'),
            bg='#27ae60', fg='white', activebackground='#229954',
            activeforeground='white', relief='raised', bd=3, padx=15, pady=10,
            cursor='hand2', highlightthickness=2, highlightbackground='#1e8449',
            highlightcolor='#1e8449'
        )
        save_btn.pack(pady=5)
        
        stats_frame = tk.LabelFrame(right_frame, text="📊 Статистика яркости", 
                                    font=('Segoe UI', 10, 'bold'), 
                                    fg='#2c3e50', bg='#e3f2fd', 
                                    relief='flat', bd=2, padx=5, pady=5)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=50, wrap='word', 
                                  state='disabled', font=('Consolas', 9),
                                  bg='#ffffff', fg='#2c3e50', 
                                  relief='solid', borderwidth=1, padx=8, pady=8)
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        image_frame = tk.LabelFrame(right_frame, text="🖼️ Результат", 
                                    font=('Segoe UI', 10, 'bold'), 
                                    fg='#2c3e50', bg='#f5f5f5', 
                                    relief='flat', bd=2, padx=5, pady=5)
        image_frame.pack(fill='both', expand=True)
        
        self.image_label = tk.Label(image_frame, text="Ожидание расчета...", 
                                    font=('Segoe UI', 11), 
                                    bg='#ffffff', fg='#7f8c8d',
                                    relief='solid', borderwidth=1)
        self.image_label.pack(expand=True, fill='both', padx=5, pady=5)
        
        self.current_image = None
    
    def update_render(self):
        try:
            self.master.config(cursor="watch")
            self.master.update()
            
            screen_w = self.screen_width.get()
            screen_h = self.screen_height.get()
            screen_w_res = int(self.screen_w_res.get())
            screen_h_res = int(self.screen_h_res.get())
            observer_z = self.observer_z.get()
            
            sphere = Sphere(
                self.sphere_x.get(),
                self.sphere_y.get(),
                self.sphere_z.get(),
                self.sphere_radius.get()
            )
            
            light1 = PointLight(
                self.light1_x.get(),
                self.light1_y.get(),
                self.light1_z.get(),
                self.light1_intensity.get()
            )
            
            light2 = PointLight(
                self.light2_x.get(),
                self.light2_y.get(),
                self.light2_z.get(),
                self.light2_intensity.get()
            )
            
            material = BlinnPhongModel(
                self.ka.get(),
                self.kd.get(),
                self.ks.get(),
                self.n.get()
            )
            
            renderer = Renderer(
                screen_w, screen_h, screen_w_res, screen_h_res,
                observer_z, sphere, [light1, light2], material
            )
            
            statistics = renderer.calculate_statistics()
            self.update_statistics(statistics)
            
            self.current_image = renderer.render()
            self.display_image(self.current_image)
            
            self.master.config(cursor="")
            
        except Exception as e:
            self.master.config(cursor="")
            messagebox.showerror("Ошибка", f"Ошибка при расчете: {str(e)}")
    
    def update_statistics(self, statistics):
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        p1 = statistics['point1']['point']
        p2 = statistics['point2']['point']
        p3 = statistics['point3']['point']
        
        text = "═══════════════════════════════════════\n"
        text += "  Яркость в трех точках сферы\n"
        text += "═══════════════════════════════════════\n\n"
        text += f"📍 Точка 1 (X+):\n"
        text += f"   Координаты: ({p1[0]:.1f}, {p1[1]:.1f}, {p1[2]:.1f}) мм\n"
        text += f"   Яркость: {statistics['point1']['intensity']:.6f}\n\n"
        text += f"📍 Точка 2 (Y+):\n"
        text += f"   Координаты: ({p2[0]:.1f}, {p2[1]:.1f}, {p2[2]:.1f}) мм\n"
        text += f"   Яркость: {statistics['point2']['intensity']:.6f}\n\n"
        text += f"📍 Точка 3 (Z+):\n"
        text += f"   Координаты: ({p3[0]:.1f}, {p3[1]:.1f}, {p3[2]:.1f}) мм\n"
        text += f"   Яркость: {statistics['point3']['intensity']:.6f}\n\n"
        text += "═══════════════════════════════════════\n"
        text += f"📈 Максимальная яркость: {statistics['max_intensity']:.6f}\n"
        text += f"📉 Минимальная яркость:  {statistics['min_intensity']:.6f}\n"
        text += "═══════════════════════════════════════\n"
        
        self.stats_text.insert(1.0, text)
        self.stats_text.config(state='disabled')
    
    def display_image(self, pil_image):
        max_display_size = 600
        w, h = pil_image.size
        
        scale = min(max_display_size / w, max_display_size / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        display_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Нет изображения", "Сначала выполните расчет.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_image.save(filename)
                messagebox.showinfo("Сохранено", f"Изображение сохранено: {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()