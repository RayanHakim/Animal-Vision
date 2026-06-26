# main.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from descriptions import ANIMAL_DESCRIPTIONS
import filters

class AnimalVisionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1100x750")
        self.window.configure(bg="#1a1a24")

        # Inisialisasi Kamera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Kamera tidak bisa diakses.")
            return

        self.current_mode = 'normal'
        self.buttons = {}
        self.last_frame = None
        self.frame_count = 0
        
        self.setup_gui()
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # --- KIRI: Area Video & Deskripsi ---
        self.left_frame = tk.Frame(self.window, bg="#1a1a24")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.left_frame, bg="#0d0d11")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.desc_frame = tk.Frame(self.left_frame, bg="#252538", height=120)
        self.desc_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        self.desc_frame.pack_propagate(False)

        self.desc_label = tk.Label(
            self.desc_frame, text=ANIMAL_DESCRIPTIONS['normal'], font=("Segoe UI", 11),
            fg="#e2e8f0", bg="#252538", justify=tk.LEFT, wraplength=700
        )
        self.desc_label.pack(side=tk.LEFT, padx=20, pady=15)

        # --- KANAN: Panel Menu Tombol (Bisa di-Scroll) ---
        self.right_panel = tk.Frame(self.window, bg="#252538", width=310)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.right_panel.pack_propagate(False)

        title_label = tk.Label(self.right_panel, text="50 FILTER PENGLIHATAN HEWAN", font=("Segoe UI", 11, "bold"), fg="#3b82f6", bg="#252538")
        title_label.pack(pady=15)

        # Integrasi Canvas & Scrollbar
        self.canvas = tk.Canvas(self.right_panel, bg="#252538", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#252538")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=280)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(5, 0))
        scrollbar.pack(side="right", fill="y")

        # Aktifkan Fitur Mousewheel Scroll
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)

        # Daftarkan Seluruh Kumpulan 50 Filter secara Berurutan
        modes = [
            ("Human (Normal Vision)", 'normal'),
            ("Dog (Anjing)", 'dog'), ("Cat (Kucing)", 'cat'), ("Bee (Lebah)", 'bee'),
            ("Snake (Ular pit)", 'snake'), ("Shark (Hiu)", 'shark'), ("Cuttlefish (Sotong)", 'cuttlefish'),
            ("Eagle (Elang)", 'eagle'), ("Owl (Burung Hantu)", 'owl'), ("Frog (Katak)", 'frog'),
            ("Horse (Kuda)", 'horse'), ("Pigeon (Merpati)", 'pigeon'), ("Bat (Kelelawar)", 'bat'),
            ("Rat (Tikus)", 'rat'), ("Chameleon (Bunglon)", 'chameleon'), ("Mantis Shrimp (Udang)", 'shrimp'),
            ("Deer (Rusa)", 'deer'), ("Rabbit (Kelinci)", 'rabbit'), ("Squid (Cumi)", 'squid'),
            ("Crocodile (Buaya)", 'crocodile'), ("Butterfly (Kupu)", 'butterfly'), ("Penguin (Penguin)", 'penguin'),
            ("Cow (Sapi)", 'cow'), ("Goat (Kambing)", 'goat'), ("Goldfish (Ikan Mas)", 'goldfish'),
            ("Spider (Laba-laba)", 'spider'), ("Reindeer (Rusa Kutub)", 'reindeer'), ("Mosquito (Nyamuk)", 'mosquito'),
            ("Archerfish (Ikan Sumpit)", 'archerfish'), ("Dragonfly (Capung)", 'dragonfly'), ("Dragonfish (Ikan Naga)", 'dragonfish'),
            ("Squirrel (Tupai)", 'squirrel'), ("Chicken (Ayam)", 'chicken'), ("Elephant (Gajah)", 'elephant'),
            ("Lion (Singa)", 'lion'), ("Mole (Tikus Tandan)", 'mole'), ("Jellyfish (Ubur-ubur)", 'jellyfish'),
            ("Snail (Siput)", 'snail'), ("Dolphin (Lumba-lumba)", 'dolphin'), ("Vulture (Burung Bangkai)", 'vulture'),
            ("Octopus (Gurita)", 'octopus'), ("Sheep (Domba)", 'sheep'), ("Lemur (Lemur)", 'lemur'),
            ("Gecko (Tokek)", 'gecko'), ("Scorpio (Kala Jengking)", 'scorpio'), ("Badger (Luwak)", 'badger'),
            ("Seal (Anjing Laut)", 'seal'), ("Firefly (Kunang)", 'firefly'), ("Sloth (Kangkang)", 'sloth'),
            ("Human Night Vision", 'human_night')
        ]

        # Membuat tombol dinamis dengan indeks nomor urut
        for idx, (display_name, mode) in enumerate(modes, start=1):
            initial_bg = "#10b981" if mode == self.current_mode else "#4b5563"
            text_str = f"{idx}. {display_name}"
            
            btn = tk.Button(
                self.scrollable_frame, text=text_str, font=("Segoe UI", 9, "bold"),
                bg=initial_bg, fg="white", activebackground="#059669", activeforeground="white",
                relief=tk.FLAT, height=2, cursor="hand2", anchor="w", padx=10,
                command=lambda m=mode: self.change_mode(m)
            )
            btn.pack(fill=tk.X, padx=5, pady=3)
            self.buttons[mode] = btn

    def on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def change_mode(self, mode):
        if self.current_mode in self.buttons:
            self.buttons[self.current_mode].config(bg="#4b5563")
        self.current_mode = mode
        if self.current_mode in self.buttons:
            self.buttons[self.current_mode].config(bg="#10b981")
        self.desc_label.config(text=ANIMAL_DESCRIPTIONS[mode])

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Optimasi filter Capung (Efek gerakan melambat/patah-patah)
            if self.current_mode == 'dragonfly':
                self.frame_count += 1
                if self.frame_count % 5 == 0 or self.last_frame is None:
                    self.last_frame = filters.apply_dragonfly_vision(frame)
                processed = self.last_frame
            else:
                filter_func = getattr(filters, f"apply_{self.current_mode}_vision", None)
                processed = filter_func(frame) if filter_func else frame.copy()

            processed = cv2.resize(processed, (680, 510))
            cv2image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(15, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = AnimalVisionApp(root, "Animal Vision Mega Simulator - 50 Ultimate Filters")
    root.mainloop()