import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os

FILENAME   = "image1.png"  # картинка рядом со скриптом
MAX_WIDTH  = 500
MAX_HEIGHT = 400


from PIL import Image as PILImage
try:
    RESAMPLING = PILImage.Resampling.LANCZOS
except Exception:
    RESAMPLING = PILImage.ANTIALIAS  # для старых Pillow

class App:
    def __init__(self, master):
        self.master = master
        master.title("Filters: Grey, Blur, Contrast, Brightness, Invert - Капарулин Тимофей P3308")

        # Верхняя панель: кнопки и ползунки
        top = tk.Frame(master); top.pack(fill="x", padx=8, pady=8)

        # Кнопки фильтров - первый ряд
        row1 = tk.Frame(top); row1.pack(fill="x", pady=(0,8))
        
        tk.Button(row1, text="Grey", command=self.make_gray).pack(side="left", padx=(0,8))
        tk.Button(row1, text="Blur", command=self.apply_blur).pack(side="left", padx=(0,16))

        # Ползунок радиуса размытия
        tk.Label(row1, text="Radius:").pack(side="left")
        self.radius = tk.Scale(row1, from_=0, to=20, orient="horizontal")
        self.radius.set(2)
        self.radius.pack(side="left", padx=(6,16))

        # Второй ряд - контраст и яркость
        row2 = tk.Frame(top); row2.pack(fill="x", pady=(0,8))
        
        # Кнопка контраста
        tk.Button(row2, text="Contrast", command=self.apply_contrast).pack(side="left", padx=(0,8))

        # Ползунок фактора контраста (0.0–3.0)
        tk.Label(row2, text="Factor:").pack(side="left")
        self.contrast_factor = tk.Scale(row2, from_=0.0, to=3.0, resolution=0.1, orient="horizontal", length=120)
        self.contrast_factor.set(1.0)  # 1.0 = без изменений
        self.contrast_factor.pack(side="left", padx=(6,16))

        # Кнопка яркости
        tk.Button(row2, text="Brightness", command=self.apply_brightness).pack(side="left", padx=(0,8))

        # Ползунок фактора яркости (0.0–3.0)
        tk.Label(row2, text="Bright:").pack(side="left")
        self.brightness_factor = tk.Scale(row2, from_=0.0, to=3.0, resolution=0.1, orient="horizontal", length=120)
        self.brightness_factor.set(1.0)  # 1.0 = без изменений
        self.brightness_factor.pack(side="left", padx=6)

        # Третий ряд - инверсия и сохранение
        row3 = tk.Frame(top); row3.pack(fill="x")
        tk.Button(row3, text="Invert Colors", command=self.invert_colors).pack(side="left", padx=(0,16))
        tk.Button(row3, text="Save PNG", command=self.save_image,).pack(side="left")

        # Область изображений
        imgs = tk.Frame(master); imgs.pack(padx=8, pady=8)

        self.left_label  = tk.Label(imgs, text="Исходное", compound="top")
        self.right_label = tk.Label(imgs, text="Результат", compound="top")
        self.left_label.grid(row=0, column=0, padx=8)
        self.right_label.grid(row=0, column=1, padx=8)

        # Загрузка исходника
        if not os.path.exists(FILENAME):
            messagebox.showerror("Нет файла", f"Положите рядом файл {FILENAME} или измените FILENAME в коде.")
            self.orig = None
            return

        self.orig = PILImage.open(FILENAME).convert("RGBA")
        self.orig_scaled = self._resize(self.orig)
        self._show_left(self.orig_scaled)

        placeholder = PILImage.new("RGBA", self.orig_scaled.size, (240, 240, 240, 255))
        self._show_right(placeholder)
        
        # Переменная для хранения последнего обработанного изображения
        self.processed_image = None

    def _resize(self, image: PILImage.Image) -> PILImage.Image:
        w, h = image.size
        ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)  # только уменьшаем
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, RESAMPLING)

    def _show_left(self, pil_img):
        self.tk_left = ImageTk.PhotoImage(pil_img)
        self.left_label.config(image=self.tk_left)

    def _show_right(self, pil_img):
        self.tk_right = ImageTk.PhotoImage(pil_img)
        self.right_label.config(image=self.tk_right)

    # Фильтры

    def make_gray(self):
        """Серое (R+G+B)//3 из исходного изображения"""
        if self.orig is None:
            return

        w, h = self.orig.size
        src = self.orig.load()

        gray_img = PILImage.new("RGBA", (w, h))
        dst = gray_img.load()

        for y in range(h):
            for x in range(w):
                r, g, b, a = src[x, y]
                gray = (r + g + b) // 3
                dst[x, y] = (gray, gray, gray, a)

        self.processed_image = gray_img  # Сохраняем обработанное изображение
        self._show_right(self._resize(gray_img))

    def apply_blur(self):
        """Гауссово размытие исходного изображения, радиус с ползунка"""
        if self.orig is None:
            return

        r = int(self.radius.get())
        blurred = self.orig.filter(ImageFilter.GaussianBlur(radius=r))
        self.processed_image = blurred  # Сохраняем обработанное изображение
        self._show_right(self._resize(blurred))

    def apply_contrast(self):
        """Контраст по фактору (ImageEnhance.Contrast) из исходного изображения"""
        if self.orig is None:
            return

        factor = float(self.contrast_factor.get()) 
        rgb = self.orig.convert("RGB")
        a = self.orig.getchannel("A")

        enhanced_rgb = ImageEnhance.Contrast(rgb).enhance(factor)
        result = PILImage.merge("RGBA", (*enhanced_rgb.split(), a))

        self.processed_image = result  # Сохраняем обработанное изображение
        self._show_right(self._resize(result))

    def apply_brightness(self):
        """Яркость по фактору (ImageEnhance.Brightness) из исходного изображения"""
        if self.orig is None:
            return

        factor = float(self.brightness_factor.get())
        rgb = self.orig.convert("RGB")
        a = self.orig.getchannel("A")

        enhanced_rgb = ImageEnhance.Brightness(rgb).enhance(factor)
        result = PILImage.merge("RGBA", (*enhanced_rgb.split(), a))

        self.processed_image = result  # Сохраняем обработанное изображение
        self._show_right(self._resize(result))

    def invert_colors(self):
        """Инверсия цветов: каждый пиксель RGB становится (255-R, 255-G, 255-B)"""
        if self.orig is None:
            return

        w, h = self.orig.size
        src = self.orig.load()

        inverted_img = PILImage.new("RGBA", (w, h))
        dst = inverted_img.load()

        for y in range(h):
            for x in range(w):
                r, g, b, a = src[x, y]
                # Инвертируем RGB, альфа-канал оставляем без изменений
                inv_r = 255 - r
                inv_g = 255 - g
                inv_b = 255 - b
                dst[x, y] = (inv_r, inv_g, inv_b, a)

        self.processed_image = inverted_img  # Сохраняем обработанное изображение
        self._show_right(self._resize(inverted_img))

    def save_image(self):
        """Сохранение обработанного изображения в формате PNG"""
        if self.processed_image is None:
            messagebox.showwarning("Нет изображения", "Сначала примените какой-нибудь фильтр!")
            return
        
        # Открываем диалог сохранения файла
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Сохранить изображение как..."
        )
        
        if file_path:  # Если пользователь выбрал файл
            try:
                # Сохраняем изображение в оригинальном размере
                self.processed_image.save(file_path, "PNG")
                messagebox.showinfo("Успех", f"Изображение сохранено:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()