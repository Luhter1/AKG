import tkinter as tk
from PIL import Image, ImageTk, ImageStat
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_PATHS = [
    "image1.png",
    "image2.png",
]

TARGET_SIZE = (400, 250)

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.i = 0

        top = tk.Frame(self.root)
        top.pack(anchor="w", fill="x")
        tk.Label(top, text="Капарулин Тимофей P3308", font=("Menlo", 13, "bold")).pack(side="left", padx=12)
        tk.Button(top, text="Далее", command=self.next_image).pack(side="left", padx=6, pady=6)
        self.info = tk.Label(top, text="", justify="left", font=("Menlo", 11))
        self.info.pack(side="left", padx=12)

        self.canvas = tk.Canvas(self.root, width=TARGET_SIZE[0], height=TARGET_SIZE[1]); self.canvas.pack()

        # контейнер под график
        self.chart_frame = tk.Frame(self.root)
        self.chart_frame.pack(fill="x", pady=8)

        self.img_item = None
        self.mpl_canvas = None

        self.show(self.i)
        self.root.mainloop()

    def show(self, idx):
        path = IMAGE_PATHS[idx]
        src = Image.open(path).convert("RGB")
        view = src.resize(TARGET_SIZE)

        # показать картинку
        ph = ImageTk.PhotoImage(view)
        self.canvas.image = ph
        if self.img_item is None:
            self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=ph)
        else:
            self.canvas.itemconfig(self.img_item, image=ph)

        # среднее RGB (для строки сверху)
        r_m, g_m, b_m = ImageStat.Stat(src).mean
        self.info.config(text=f"Среднее RGB: ({r_m:.1f}, {g_m:.1f}, {b_m:.1f})")

        # доминирующие пиксели (R>G&B, G>R&B, B>R&G)
        r_cnt = g_cnt = b_cnt = 0
        for r, g, b in src.getdata():
            if r > g and r > b: r_cnt += 1
            elif g > r and g > b: g_cnt += 1
            elif b > r and b > g: b_cnt += 1

        # построить/обновить столбчатую диаграмму «как на фото»
        self.draw_bar_chart(r_cnt, g_cnt, b_cnt)

    def draw_bar_chart(self, r_cnt, g_cnt, b_cnt):
        data = [r_cnt, g_cnt, b_cnt]
        labels = ["R", "G", "B"]
        colors = ["red", "green", "blue"]

        # если график уже есть — удалим и перерисуем, чтобы не плодить виджеты
        if self.mpl_canvas is not None:
            self.mpl_canvas.get_tk_widget().destroy()
            self.mpl_canvas = None

        fig = Figure(figsize=(6, 3), dpi=100)          # одна диаграмма на фигуру
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, data, color=colors)      # цветные столбики R/G/B

        ax.set_title("Пиксели по каналам", pad=12)
        ax.set_ylabel("Количество пикселей")
        ax.set_ylim(0, max(data) * 1.15 if max(data) > 0 else 1)

        # подпишем значения над столбцами
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom')

        # встраиваем в Tkinter
        self.mpl_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(fill="x")

    def next_image(self):
        self.i = (self.i + 1) % len(IMAGE_PATHS)
        self.show(self.i)

App()