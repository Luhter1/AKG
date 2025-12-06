import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import os


class PointLight:
    def __init__(self, x, y, z, intensity):
        self.position = np.array([x, y, z], dtype=float)
        self.intensity = float(intensity)
    
    def get_direction_to(self, point):
        direction = point - self.position
        distance = np.linalg.norm(direction)
        if distance < 1e-10:
            return np.array([0.0, 0.0, 0.0])
        return direction / distance
    
    def get_distance_to(self, point):
        return np.linalg.norm(point - self.position)
    
    def get_lambert_intensity(self, point, normal):
        direction = self.get_direction_to(point)
        distance = self.get_distance_to(point)
        
        if distance < 1e-10:
            return 0.0
        
        cos_theta = np.dot(-direction, normal)
        cos_theta = max(0.0, cos_theta)
        
        attenuation = 1.0 / (distance * distance)
        return self.intensity * cos_theta * attenuation


class Sphere:
    def __init__(self, center_x, center_y, center_z, radius):
        self.center = np.array([center_x, center_y, center_z], dtype=float)
        self.radius = float(radius)
    
    def get_normal(self, point):
        normal = point - self.center
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return normal / norm
    
    def intersect_ray(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        t = min(t1, t2) if t1 > 0 else (t2 if t2 > 0 else None)
        
        if t is None or t < 0:
            return None
        
        return origin + t * direction


class BlinnPhongModel:
    def __init__(self, ka, kd, ks, n):
        self.ka = float(ka)
        self.kd = float(kd)
        self.ks = float(ks)
        self.n = float(n)
    
    def calculate_intensity(self, point, normal, view_dir, lights):
        ambient = self.ka
        
        view_dir_norm = np.linalg.norm(view_dir)
        if view_dir_norm > 1e-10:
            view_dir_normalized = view_dir / view_dir_norm
        else:
            view_dir_normalized = view_dir
        
        diffuse = 0.0
        specular = 0.0
        
        for light in lights:
            light_to_point = light.get_direction_to(point)
            distance = light.get_distance_to(point)
            
            if distance < 1e-10:
                continue
            
            light_dir_norm = np.linalg.norm(light_to_point)
            if light_dir_norm > 1e-10:
                light_dir_normalized = light_to_point / light_dir_norm
                light_dir = -light_dir_normalized
            else:
                continue
            
            light_emission_direction = np.array([0.0, 0.0, -1.0])
            cos_theta_emission = np.dot(light_dir_normalized, light_emission_direction)
            cos_theta_emission = max(0.0, cos_theta_emission)
            
            cos_theta = np.dot(light_dir, normal)
            cos_theta = max(0.0, cos_theta)
            
            min_distance = 10.0
            safe_distance = max(distance, min_distance)
            attenuation_factor = 1.0 / (safe_distance * safe_distance)
            light_intensity_attenuated = light.intensity * cos_theta_emission * attenuation_factor
            
            diffuse += self.kd * light_intensity_attenuated * cos_theta
            
            if cos_theta > 0:
                half_vector = view_dir_normalized + light_dir
                half_norm = np.linalg.norm(half_vector)
                if half_norm > 1e-10:
                    half_vector = half_vector / half_norm
                    cos_alpha = np.dot(half_vector, normal)
                    cos_alpha = max(0.0, cos_alpha)
                    specular += self.ks * light_intensity_attenuated * (cos_alpha ** self.n)
        
        total = ambient + diffuse + specular
        return max(0.0, total)


class Renderer:
    def __init__(self, screen_width, screen_height, screen_w_res, screen_h_res, 
                 observer_z, sphere, lights, material):
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)
        self.screen_w_res = int(screen_w_res)
        self.screen_h_res = int(screen_h_res)
        self.observer = np.array([0.0, 0.0, float(observer_z)])
        self.sphere = sphere
        self.lights = lights
        self.material = material
        
        self.pixel_width = self.screen_width / self.screen_w_res
        self.pixel_height = self.screen_height / self.screen_h_res
    
    def calculate_intensity_at_point(self, point):
        normal = self.sphere.get_normal(point)
        direction_to_observer = self.observer - point
        direction_norm = np.linalg.norm(direction_to_observer)
        if direction_norm < 1e-10:
            view_dir = np.array([0.0, 0.0, 1.0])
        else:
            view_dir = direction_to_observer / direction_norm
        
        intensity = self.material.calculate_intensity(
            point, normal, view_dir, self.lights
        )
        return intensity
    
    def get_three_points_on_sphere(self):
        center = self.sphere.center
        radius = self.sphere.radius
        
        point1 = center + np.array([radius, 0.0, 0.0])
        point2 = center + np.array([0.0, radius, 0.0])
        point3 = center + np.array([0.0, 0.0, radius])
        
        return point1, point2, point3
    
    def calculate_statistics(self):
        point1, point2, point3 = self.get_three_points_on_sphere()
        
        intensity1 = self.calculate_intensity_at_point(point1)
        intensity2 = self.calculate_intensity_at_point(point2)
        intensity3 = self.calculate_intensity_at_point(point3)
        
        image = np.zeros((self.screen_h_res, self.screen_w_res), dtype=float)
        
        for y in range(self.screen_h_res):
            for x in range(self.screen_w_res):
                screen_x = (x + 0.5) * self.pixel_width - self.screen_width / 2.0
                screen_y = -(y + 0.5) * self.pixel_height + self.screen_height / 2.0
                screen_point = np.array([screen_x, screen_y, 0.0])
                
                direction = screen_point - self.observer
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-10:
                    continue
                direction = direction / direction_norm
                
                intersection = self.sphere.intersect_ray(self.observer, direction)
                
                if intersection is not None:
                    normal = self.sphere.get_normal(intersection)
                    view_dir = -direction
                    
                    intensity = self.material.calculate_intensity(
                        intersection, normal, view_dir, self.lights
                    )
                    
                    image[y, x] = intensity
        
        max_intensity = np.max(image)
        min_intensity = np.min(image[image > 0]) if np.any(image > 0) else 0.0
        
        return {
            'point1': {'point': point1, 'intensity': intensity1},
            'point2': {'point': point2, 'intensity': intensity2},
            'point3': {'point': point3, 'intensity': intensity3},
            'max_intensity': max_intensity,
            'min_intensity': min_intensity
        }
    
    def render(self):
        image = np.zeros((self.screen_h_res, self.screen_w_res), dtype=float)
        
        for y in range(self.screen_h_res):
            for x in range(self.screen_w_res):
                screen_x = (x + 0.5) * self.pixel_width - self.screen_width / 2.0
                screen_y = -(y + 0.5) * self.pixel_height + self.screen_height / 2.0
                screen_point = np.array([screen_x, screen_y, 0.0])
                
                direction = screen_point - self.observer
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-10:
                    continue
                direction = direction / direction_norm
                
                intersection = self.sphere.intersect_ray(self.observer, direction)
                
                if intersection is not None:
                    normal = self.sphere.get_normal(intersection)
                    view_dir = -direction
                    
                    intensity = self.material.calculate_intensity(
                        intersection, normal, view_dir, self.lights
                    )
                    
                    image[y, x] = intensity
        
        max_intensity = np.max(image)
        min_intensity = np.min(image[image > 0]) if np.any(image > 0) else 0
        
        if max_intensity > 1e-10:
            if max_intensity - min_intensity > 1e-6:
                image_shifted = image - min_intensity
                image_shifted = np.maximum(image_shifted, 0.0)
                scale_factor = 50.0 / max_intensity
                image_scaled = image_shifted * scale_factor
                image_log = np.log1p(image_scaled)
                max_log = np.log1p((max_intensity - min_intensity) * scale_factor)
                if max_log > 1e-10:
                    image_normalized = image_log / max_log
                else:
                    image_normalized = image_log
            else:
                image_normalized = image / max_intensity
            image_normalized = np.clip(image_normalized, 0.0, 1.0)
            image = (image_normalized * 255.0).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
        
        return Image.fromarray(image, mode='L')


class ParameterControl:
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