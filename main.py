import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageDraw

from converter import convert_comic_to_video

# Application Config
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ComicToVideoApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Comic Book to 2D Animation Movie Converter")
        self.root.geometry("680x810")
        self.root.resizable(False, False)
        
        # State variables
        self.pdf_path = ""
        self.audio_path = ""
        self.output_path = ""
        self.is_converting = False
        
        self.build_ui()
        self.log("Ready. Please select a Comic PDF file to start.")

    def build_ui(self):
        # Header / Title Block
        header_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        header_frame.pack(fill="x", padx=25, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Comic to Motion Video Converter", 
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold")
        )
        title_label.pack(anchor="w")
        
        subtitle_label = ctk.CTkLabel(
            header_frame, 
            text="Transform PDF comic pages and panels into an MP4 motion movie with smooth transitions.", 
            font=ctk.CTkFont(family="Inter", size=13),
            text_color="gray70"
        )
        subtitle_label.pack(anchor="w", pady=(2, 0))

        # Main Scrollable settings box
        content_frame = ctk.CTkFrame(self.root, fg_color="gray10", corner_radius=12)
        content_frame.pack(fill="both", expand=True, padx=25, pady=10)
        
        # 1. FILE SELECTION
        files_section = ctk.CTkLabel(
            content_frame, 
            text="1. File Locations", 
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1a73e8"
        )
        files_section.grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 10))
        
        # PDF Input
        pdf_label = ctk.CTkLabel(content_frame, text="Comic PDF Input:")
        pdf_label.grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.pdf_entry = ctk.CTkEntry(content_frame, placeholder_text="Select a PDF file...", width=380)
        self.pdf_entry.grid(row=1, column=1, padx=5, pady=5)
        pdf_browse = ctk.CTkButton(content_frame, text="Browse", width=80, command=self.browse_pdf)
        pdf_browse.grid(row=1, column=2, padx=(5, 20), pady=5)
        
        # Audio Input
        audio_label = ctk.CTkLabel(content_frame, text="Background Audio (Optional):")
        audio_label.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.audio_entry = ctk.CTkEntry(content_frame, placeholder_text="Select an MP3/WAV file...", width=380)
        self.audio_entry.grid(row=2, column=1, padx=5, pady=5)
        audio_browse = ctk.CTkButton(content_frame, text="Browse", width=80, command=self.browse_audio)
        audio_browse.grid(row=2, column=2, padx=(5, 20), pady=5)
        
        # MP4 Output
        output_label = ctk.CTkLabel(content_frame, text="MP4 Video Output:")
        output_label.grid(row=3, column=0, sticky="w", padx=20, pady=5)
        self.output_entry = ctk.CTkEntry(content_frame, placeholder_text="Destination file path...", width=380)
        self.output_entry.grid(row=3, column=1, padx=5, pady=5)
        output_browse = ctk.CTkButton(content_frame, text="Save As", width=80, command=self.browse_output)
        output_browse.grid(row=3, column=2, padx=(5, 20), pady=5)
        
        # Divider line
        div1 = ctk.CTkFrame(content_frame, height=2, fg_color="gray20")
        div1.grid(row=4, column=0, columnspan=3, sticky="ew", padx=20, pady=15)
        
        # 2. CONVERSION SETTINGS
        settings_section = ctk.CTkLabel(
            content_frame, 
            text="2. Motion Customization Settings", 
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1a73e8"
        )
        settings_section.grid(row=5, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10))
        
        # Render Mode
        mode_label = ctk.CTkLabel(content_frame, text="Conversion Mode:")
        mode_label.grid(row=6, column=0, sticky="w", padx=20, pady=5)
        self.mode_switch = ctk.CTkSegmentedButton(
            content_frame, 
            values=["Smart Panels (Motion)", "Full Page Slideshow"],
            width=380
        )
        self.mode_switch.grid(row=6, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        self.mode_switch.set("Smart Panels (Motion)")
        
        # Target Resolution
        res_label = ctk.CTkLabel(content_frame, text="Video Resolution:")
        res_label.grid(row=7, column=0, sticky="w", padx=20, pady=5)
        self.res_switch = ctk.CTkSegmentedButton(
            content_frame, 
            values=["1080p (Full HD)", "720p (HD)"],
            width=250
        )
        self.res_switch.grid(row=7, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        self.res_switch.set("1080p (Full HD)")
        
        # Duration per Scene
        duration_label = ctk.CTkLabel(content_frame, text="Scene Duration:")
        duration_label.grid(row=8, column=0, sticky="w", padx=20, pady=5)
        
        self.duration_slider = ctk.CTkSlider(content_frame, from_=2.0, to=10.0, number_of_steps=16, width=280, command=self.update_duration_label)
        self.duration_slider.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        self.duration_slider.set(4.0)
        self.duration_val_label = ctk.CTkLabel(content_frame, text="4.0s / scene")
        self.duration_val_label.grid(row=8, column=2, sticky="w", padx=5, pady=5)
        
        # Crossfade Transition
        fade_label = ctk.CTkLabel(content_frame, text="Crossfade Duration:")
        fade_label.grid(row=9, column=0, sticky="w", padx=20, pady=5)
        
        self.fade_slider = ctk.CTkSlider(content_frame, from_=0.0, to=2.0, number_of_steps=20, width=280, command=self.update_fade_label)
        self.fade_slider.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        self.fade_slider.set(0.5)
        self.fade_val_label = ctk.CTkLabel(content_frame, text="0.5s fade")
        self.fade_val_label.grid(row=9, column=2, sticky="w", padx=5, pady=5)
        
        # Divider line 2
        div2 = ctk.CTkFrame(content_frame, height=2, fg_color="gray20")
        div2.grid(row=10, column=0, columnspan=3, sticky="ew", padx=20, pady=15)
        
        # 3. 2D / 2.5D ANIMATION SETTINGS
        anim_section = ctk.CTkLabel(
            content_frame, 
            text="3. 2D / 2.5D Animation Settings", 
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#1a73e8"
        )
        anim_section.grid(row=11, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10))
        
        # 2.5D Parallax
        self.parallax_var = tk.BooleanVar(value=True)
        self.parallax_check = ctk.CTkCheckBox(
            content_frame, 
            text="Enable 2.5D Character Parallax (Segments characters & animates layers)", 
            variable=self.parallax_var
        )
        self.parallax_check.grid(row=12, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        
        # Camera Bobbing
        self.bobbing_var = tk.BooleanVar(value=True)
        self.bobbing_check = ctk.CTkCheckBox(
            content_frame, 
            text="Enable Camera Bobbing (Sinusoidal floating camera drift)", 
            variable=self.bobbing_var
        )
        self.bobbing_check.grid(row=13, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        
        # Action Bar (Progress bar and Start button)
        action_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        action_frame.pack(fill="x", padx=25, pady=10)
        
        # Progress info
        self.progress_bar = ctk.CTkProgressBar(action_frame, height=12)
        self.progress_bar.pack(fill="x", pady=(5, 10))
        self.progress_bar.set(0.0)
        
        self.start_btn = ctk.CTkButton(
            action_frame, 
            text="START CONVERSION", 
            height=45, 
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#1a73e8",
            hover_color="#155cb0",
            command=self.start_conversion
        )
        self.start_btn.pack(fill="x")
        
        # Log Terminal window
        log_frame = ctk.CTkFrame(self.root, fg_color="black", height=150, corner_radius=8)
        log_frame.pack(fill="both", expand=True, padx=25, pady=(0, 20))
        
        self.log_text = tk.Text(
            log_frame, 
            bg="black", 
            fg="#00ff00", 
            insertbackground="white", 
            font=("Consolas", 11),
            bd=0, 
            highlightthickness=0
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def update_duration_label(self, value):
        self.duration_val_label.configure(text=f"{value:.1f}s / scene")
        
    def update_fade_label(self, value):
        self.fade_val_label.configure(text=f"{value:.1f}s fade")

    def browse_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("Comic Book PDF", "*.pdf")])
        if path:
            self.pdf_path = path
            self.pdf_entry.delete(0, "end")
            self.pdf_entry.insert(0, path)
            # Suggest output filename
            base_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            suggested_out = os.path.join(base_dir, f"{base_name}.mp4")
            self.output_path = suggested_out
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, suggested_out)
            self.log(f"Selected PDF: {os.path.basename(path)}")

    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if path:
            self.audio_path = path
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)
            self.log(f"Selected Audio: {os.path.basename(path)}")

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Movie", "*.mp4")])
        if path:
            self.output_path = path
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)
            self.log(f"Selected Destination: {path}")

    def log(self, message):
        self.root.after(0, lambda: self._add_log_message(message))
        
    def _add_log_message(self, message):
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see("end")

    def set_progress(self, val):
        self.root.after(0, lambda: self.progress_bar.set(val / 100.0))

    def start_conversion(self):
        if self.is_converting:
            self.log("A conversion is already running.")
            return
            
        pdf = self.pdf_entry.get().strip()
        out = self.output_entry.get().strip()
        audio = self.audio_entry.get().strip()
        
        if not pdf or not os.path.exists(pdf):
            self.log("ERROR: Invalid PDF path selected.")
            return
        if not out:
            self.log("ERROR: Invalid output path specified.")
            return
            
        self.is_converting = True
        self.start_btn.configure(state="disabled", text="CONVERTING...", fg_color="gray30")
        self.progress_bar.set(0.0)
        self.log("Starting conversion pipeline...")
        
        # Read settings values
        mode = "Panels" if "Panels" in self.mode_switch.get() else "FullPage"
        resolution = "1080p" if "1080p" in self.res_switch.get() else "720p"
        duration = self.duration_slider.get()
        fade = self.fade_slider.get()
        bobbing = self.bobbing_var.get()
        parallax = self.parallax_var.get()
        
        # Spawn thread
        t = threading.Thread(
            target=self.run_pipeline,
            args=(pdf, out, audio, mode, resolution, duration, fade, bobbing, parallax),
            daemon=True
        )
        t.start()

    def run_pipeline(self, pdf, out, audio, mode, resolution, duration, fade, bobbing, parallax):
        try:
            convert_comic_to_video(
                pdf_path=pdf,
                output_mp4_path=out,
                mode=mode,
                duration_per_scene=duration,
                fps=30,
                resolution=resolution,
                zoom_speed=1.12,
                transition_sec=fade,
                audio_path=audio if audio else None,
                enable_bobbing=bobbing,
                enable_parallax=parallax,
                enable_particles=False,     # Visual effect overlays disabled per request
                enable_speed_lines=False,   # Visual effect overlays disabled per request
                log_callback=self.log,
                progress_callback=self.set_progress
            )
            self.log("SUCCESS: Movie generation completed successfully!")
        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
        finally:
            self.is_converting = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal", text="START CONVERSION", fg_color="#1a73e8"))
            self.set_progress(0)

if __name__ == "__main__":
    app = ComicToVideoApp()
    app.root.mainloop()
