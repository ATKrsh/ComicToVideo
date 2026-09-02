import os
import io
import time
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(pdf_path, dpi=150, log_callback=None):
    """
    Renders each page of the PDF into a PIL Image.
    """
    if log_callback:
        log_callback(f"Opening PDF: {os.path.basename(pdf_path)}...")
    doc = fitz.open(pdf_path)
    images = []
    num_pages = len(doc)
    
    for i in range(num_pages):
        if log_callback:
            log_callback(f"Rendering page {i+1}/{num_pages} (DPI={dpi})...")
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
        
    if log_callback:
        log_callback(f"Successfully extracted {num_pages} pages.")
    return images

def detect_panels(pil_image, log_callback=None):
    """
    Detects panels on a comic book page using OpenCV contour analysis.
    Sorts panels in reading order: top-to-bottom, left-to-right.
    """
    # Convert PIL Image to OpenCV BGR
    cv_img = np.array(pil_image)
    cv_img = cv_img[:, :, ::-1].copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to extract borders
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # Morphological dilation to connect lines/characters inside panels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = gray.shape
    min_area = width * height * 0.02  # Panel must cover at least 2% of the page
    
    panels = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out tiny noise and full page border overlays
        if w * h > min_area and w < width * 0.98 and h < height * 0.98:
            panels.append((x, y, w, h))
            
    # Fallback to full page if no panels are found
    if not panels:
        if log_callback:
            log_callback("No panels detected. Using full page.")
        return [(0, 0, width, height)]
        
    # Sort panels by row:
    # 1. Sort panels by Y coordinate first
    panels = sorted(panels, key=lambda p: p[1])
    
    sorted_panels = []
    current_row = []
    row_y_threshold = height * 0.08  # Panels within 8% height are considered on the same row
    
    for p in panels:
        if not current_row:
            current_row.append(p)
        else:
            if abs(p[1] - current_row[0][1]) < row_y_threshold:
                current_row.append(p)
            else:
                # Sort completed row by X (left to right)
                sorted_panels.extend(sorted(current_row, key=lambda r: r[0]))
                current_row = [p]
    if current_row:
        sorted_panels.extend(sorted(current_row, key=lambda r: r[0]))
        
    if log_callback:
        log_callback(f"Detected {len(sorted_panels)} panels on page.")
        
    return sorted_panels

def init_particles(width, height, count=30):
    """
    Initializes a persistent list of ambient particle structures.
    """
    particles = []
    for _ in range(count):
        particles.append({
            "x": np.random.uniform(0, width),
            "y": np.random.uniform(0, height),
            "vx": np.random.uniform(-1.0, 1.0),
            "vy": np.random.uniform(-1.5, -0.5), # floating upwards
            "size": np.random.randint(2, 6),
            "alpha": np.random.uniform(0.3, 0.7)
        })
    return particles

def update_and_draw_particles(frame, particles, color_type="White Dust"):
    """
    Updates ambient particles positions and overlays them on the BGR frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    color_map = {
        "White Dust": (255, 255, 255),
        "Orange Embers": (20, 120, 255),  # Orange in BGR
        "Blue Sparks": (255, 180, 50)      # Ice blue in BGR
    }
    color = color_map.get(color_type, (255, 255, 255))
    
    for p in particles:
        # Move
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        
        # Reset if off-screen
        if p["x"] < 0 or p["x"] > w or p["y"] < 0 or p["y"] > h:
            p["x"] = np.random.uniform(0, w)
            p["y"] = h + np.random.uniform(5, 20)
            p["vx"] = np.random.uniform(-1.0, 1.0)
            p["vy"] = np.random.uniform(-1.5, -0.5)
            
        cx, cy = int(p["x"]), int(p["y"])
        cv2.circle(overlay, (cx, cy), p["size"], color, -1)
        
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

def draw_speed_lines(frame, frame_idx):
    """
    Renders dynamic action speed lines starting from the screen boundary pointing inwards.
    """
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    overlay = frame.copy()
    
    num_lines = 40
    # Different seed per frame to make them flicker
    np.random.seed(frame_idx)
    
    diag = int(np.sqrt(w**2 + h**2) // 2)
    
    for _ in range(num_lines):
        angle = np.random.uniform(0, 2 * np.pi)
        start_r = diag
        end_r = int(diag * 0.70)
        
        x1 = int(cx + start_r * np.cos(angle))
        y1 = int(cy + start_r * np.sin(angle))
        x2 = int(cx + end_r * np.cos(angle))
        y2 = int(cy + end_r * np.sin(angle))
        
        thickness = np.random.randint(1, 3)
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

def segment_character(cv_img):
    """
    Segments the foreground character using OpenCV GrabCut.
    Returns a binary mask where 255 represents the foreground character.
    """
    h, w, _ = cv_img.shape
    if h < 50 or w < 50:
        return np.zeros((h, w), dtype=np.uint8)
        
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define bounding box around the center region (leaves a 10% edge padding)
    bx = w // 10
    by = h // 10
    bw = w - 2 * bx
    bh = h - 2 * by
    rect = (bx, by, bw, bh)
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        
        # Generate binary mask where values are GC_FGD (1) or GC_PR_FGD (3)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Sanity check: if mask is empty or fills too much (>85%), abort segmentation
        pixels_fg = np.sum(fg_mask == 255)
        total_pixels = w * h
        if pixels_fg == 0 or pixels_fg > total_pixels * 0.85:
            return np.zeros((h, w), dtype=np.uint8)
            
        # Smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        return fg_mask
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

def inpaint_background(cv_img, mask):
    """
    Inpaints (removes) the segmented character from the background plate.
    """
    if np.sum(mask == 255) == 0:
        return cv_img.copy()
    try:
        # Inpaint using Fast Marching Method (Telea)
        inpainted = cv2.inpaint(cv_img, mask, 5, cv2.INPAINT_TELEA)
        return inpainted
    except Exception:
        return cv_img.copy()

def render_ken_burns_frames(
    pil_image, 
    bbox, 
    width, 
    height, 
    num_frames, 
    zoom_speed=1.12, 
    enable_bobbing=False, 
    enable_parallax=False
):
    """
    Renders Ken Burns transition frames for a bounding box on the image.
    Handles tall panels (vertical pan), wide panels (horizontal pan), and square panels (center zoom).
    If enable_parallax is True, segments character using GrabCut, inpaints the background,
    and animates layers independently (2.5D Parallax).
    Returns list of OpenCV BGR frames.
    """
    px, py, pw, ph = bbox
    panel_crop = pil_image.crop((px, py, px + pw, py + ph))
    w_p, h_p = panel_crop.size
    
    panel_cv = np.array(panel_crop)[:, :, ::-1].copy()
    target_aspect = width / height
    panel_aspect = w_p / h_p
    
    mask = None
    bg_inpainted = None
    fg_char = None
    
    if enable_parallax:
        mask = segment_character(panel_cv)
        if np.sum(mask == 255) > 0:
            bg_inpainted = inpaint_background(panel_cv, mask)
            fg_char = cv2.bitwise_and(panel_cv, panel_cv, mask=mask)
        else:
            mask = None
            
    frames = []
    
    # ----------------------------------------------------
    # CASE 1: Tall Panel -> Vertical Pan (from top to bottom)
    # ----------------------------------------------------
    if panel_aspect < (target_aspect * 0.75):
        scale = width / w_p
        scaled_h = int(h_p * scale)
        
        if mask is not None:
            resized_bg = cv2.resize(bg_inpainted, (width, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            resized_fg = cv2.resize(fg_char, (width, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(mask, (width, scaled_h), interpolation=cv2.INTER_NEAREST)
            
            max_scroll = scaled_h - height
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                
                # Pan BG slightly slower (0.8x)
                y_offset_bg = int(t_smooth * max_scroll * 0.8)
                # Pan FG character slightly faster (1.2x)
                y_offset_fg = int(t_smooth * max_scroll * 1.2)
                
                if enable_bobbing:
                    bob_amplitude = int(height * 0.012)
                    y_offset_fg += int(bob_amplitude * np.cos(2.5 * 2 * np.pi * t))
                    y_offset_bg += int(bob_amplitude * 0.3 * np.cos(2.5 * 2 * np.pi * t))
                    
                y_offset_bg = max(0, min(scaled_h - height, y_offset_bg))
                y_offset_fg = max(0, min(scaled_h - height, y_offset_fg))
                
                bg_frame = resized_bg[y_offset_bg : y_offset_bg + height, 0 : width].copy()
                fg_frame = resized_fg[y_offset_fg : y_offset_fg + height, 0 : width]
                mask_frame = resized_mask[y_offset_fg : y_offset_fg + height, 0 : width]
                
                # Composite
                np.copyto(bg_frame, fg_frame, where=(mask_frame[:, :, None] > 0))
                frames.append(bg_frame)
        else:
            resized_panel = cv2.resize(panel_cv, (width, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            max_scroll = scaled_h - height
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                y_offset = int(t_smooth * max_scroll)
                if enable_bobbing:
                    y_offset += int(height * 0.012 * np.cos(2.5 * 2 * np.pi * t))
                y_offset = max(0, min(scaled_h - height, y_offset))
                frame = resized_panel[y_offset : y_offset + height, 0 : width]
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                frames.append(frame)
                
    # ----------------------------------------------------
    # CASE 2: Wide Panel -> Horizontal Pan (from left to right)
    # ----------------------------------------------------
    elif panel_aspect > (target_aspect * 1.5):
        scale = height / h_p
        scaled_w = int(w_p * scale)
        
        if mask is not None:
            resized_bg = cv2.resize(bg_inpainted, (scaled_w, height), interpolation=cv2.INTER_LANCZOS4)
            resized_fg = cv2.resize(fg_char, (scaled_w, height), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(mask, (scaled_w, height), interpolation=cv2.INTER_NEAREST)
            
            max_scroll = scaled_w - width
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                
                x_offset_bg = int(t_smooth * max_scroll * 0.8)
                x_offset_fg = int(t_smooth * max_scroll * 1.2)
                
                if enable_bobbing:
                    bob_amplitude = int(width * 0.012)
                    x_offset_fg += int(bob_amplitude * np.sin(2.5 * 2 * np.pi * t))
                    x_offset_bg += int(bob_amplitude * 0.3 * np.sin(2.5 * 2 * np.pi * t))
                    
                x_offset_bg = max(0, min(scaled_w - width, x_offset_bg))
                x_offset_fg = max(0, min(scaled_w - width, x_offset_fg))
                
                bg_frame = resized_bg[0 : height, x_offset_bg : x_offset_bg + width].copy()
                fg_frame = resized_fg[0 : height, x_offset_fg : x_offset_fg + width]
                mask_frame = resized_mask[0 : height, x_offset_fg : x_offset_fg + width]
                
                np.copyto(bg_frame, fg_frame, where=(mask_frame[:, :, None] > 0))
                frames.append(bg_frame)
        else:
            resized_panel = cv2.resize(panel_cv, (scaled_w, height), interpolation=cv2.INTER_LANCZOS4)
            max_scroll = scaled_w - width
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                x_offset = int(t_smooth * max_scroll)
                if enable_bobbing:
                    x_offset += int(width * 0.012 * np.sin(2.5 * 2 * np.pi * t))
                x_offset = max(0, min(scaled_w - width, x_offset))
                frame = resized_panel[0 : height, x_offset : x_offset + width]
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                frames.append(frame)
                
    # ----------------------------------------------------
    # CASE 3: Standard Panel -> Letterbox + Center Zoom / Float
    # ----------------------------------------------------
    else:
        scale = min(width / w_p, height / h_p)
        scaled_w = int(w_p * scale)
        scaled_h = int(h_p * scale)
        dx = (width - scaled_w) // 2
        dy = (height - scaled_h) // 2
        
        if mask is not None:
            resized_bg = cv2.resize(bg_inpainted, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            resized_fg = cv2.resize(fg_char, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            
            base_bg = np.zeros((height, width, 3), dtype=np.uint8)
            base_fg = np.zeros((height, width, 3), dtype=np.uint8)
            base_mask = np.zeros((height, width), dtype=np.uint8)
            
            base_bg[dy : dy + scaled_h, dx : dx + scaled_w] = resized_bg
            base_fg[dy : dy + scaled_h, dx : dx + scaled_w] = resized_fg
            base_mask[dy : dy + scaled_h, dx : dx + scaled_w] = resized_mask
            
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                
                zoom_bg = 1.0 + (zoom_speed - 1.0) * 0.4 * t_smooth
                zoom_fg = 1.0 + (zoom_speed - 1.0) * 1.3 * t_smooth
                
                M_bg = cv2.getRotationMatrix2D((width / 2, height / 2), 0, zoom_bg)
                M_fg = cv2.getRotationMatrix2D((width / 2, height / 2), 0, zoom_fg)
                
                if enable_bobbing:
                    bob_x_fg = int(width * 0.015 * np.sin(2.5 * 2 * np.pi * t))
                    bob_y_fg = int(height * 0.015 * np.cos(2.5 * 2 * np.pi * t))
                    
                    bob_x_bg = -int(width * 0.005 * np.sin(2.5 * 2 * np.pi * t))
                    bob_y_bg = -int(height * 0.005 * np.cos(2.5 * 2 * np.pi * t))
                    
                    M_fg[0, 2] += bob_x_fg
                    M_fg[1, 2] += bob_y_fg
                    
                    M_bg[0, 2] += bob_x_bg
                    M_bg[1, 2] += bob_y_bg
                    
                warped_bg = cv2.warpAffine(base_bg, M_bg, (width, height), flags=cv2.INTER_LANCZOS4)
                warped_fg = cv2.warpAffine(base_fg, M_fg, (width, height), flags=cv2.INTER_LANCZOS4)
                warped_mask = cv2.warpAffine(base_mask, M_fg, (width, height), flags=cv2.INTER_NEAREST)
                
                np.copyto(warped_bg, warped_fg, where=(warped_mask[:, :, None] > 0))
                frames.append(warped_bg)
        else:
            resized = cv2.resize(panel_cv, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            base = np.zeros((height, width, 3), dtype=np.uint8)
            base[dy : dy + scaled_h, dx : dx + scaled_w] = resized
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                t_smooth = t * t * (3 - 2 * t)
                zoom = 1.0 + (zoom_speed - 1.0) * t_smooth
                cw = int(width / zoom)
                ch = int(height / zoom)
                cx = (width - cw) // 2
                cy = (height - ch) // 2
                if enable_bobbing:
                    bob_x_amp = int((width - cw) * 0.06) if (width - cw) > 0 else 6
                    bob_y_amp = int((height - ch) * 0.06) if (height - ch) > 0 else 6
                    cx += int(bob_x_amp * np.sin(2.5 * 2 * np.pi * t))
                    cy += int(bob_y_amp * np.cos(2.5 * 2 * np.pi * t))
                cx = max(0, min(width - cw, cx))
                cy = max(0, min(height - ch, cy))
                crop = base[cy : cy + ch, cx : cx + cw]
                frame = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)
                
    return frames

def render_crossfade(frame_a, frame_b, alpha):
    return cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0)

def convert_comic_to_video(
    pdf_path,
    output_mp4_path,
    mode="Panels",            # "Panels" or "FullPage"
    duration_per_scene=4.0,   # Seconds per panel/page
    fps=30,
    resolution="1080p",       # "1080p" or "720p"
    zoom_speed=1.12,          # Max zoom level for standard pan
    transition_sec=0.5,       # Crossfade duration
    audio_path=None,          # Background audio path
    enable_bobbing=False,     # Sinusoidal drifting
    enable_parallax=False,    # 2.5D layer segmentation
    enable_particles=False,
    enable_speed_lines=False,
    particle_type="White Dust",
    log_callback=None,
    progress_callback=None
):
    """
    Core pipeline: extracts comic pages, processes panels, renders 2.5D Parallax and Ken Burns,
    compiles temporary video, and overlays audio track.
    """
    start_time = time.time()
    
    res_map = {
        "1080p": (1920, 1080),
        "720p": (1280, 720)
    }
    width, height = res_map.get(resolution, (1920, 1080))
    
    pages = pdf_to_images(pdf_path, dpi=150, log_callback=log_callback)
    if not pages:
        raise ValueError("Failed to extract pages from PDF.")
        
    scenes = []
    for idx, page in enumerate(pages):
        w_page, h_page = page.size
        if mode == "Panels":
            if log_callback:
                log_callback(f"Analyzing panels for Page {idx+1}...")
            bboxes = detect_panels(page, log_callback=log_callback)
            for bbox in bboxes:
                scenes.append((idx, bbox))
        else:
            scenes.append((idx, (0, 0, w_page, h_page)))
            
    num_scenes = len(scenes)
    if log_callback:
        log_callback(f"Total scenes to render: {num_scenes}.")
        
    temp_avi_path = output_mp4_path + ".temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
    
    frames_per_scene = int(duration_per_scene * fps)
    frames_transition = int(transition_sec * fps)
    
    last_frame = None
    particles = init_particles(width, height, count=35) if enable_particles else None
    total_frame_counter = 0
    
    for idx, (page_idx, bbox) in enumerate(scenes):
        if log_callback:
            log_callback(f"Rendering scene {idx+1}/{num_scenes} (Page {page_idx+1})...")
            
        page_img = pages[page_idx]
        
        # Render frames with Parallax if enabled
        scene_frames = render_ken_burns_frames(
            page_img, bbox, width, height, frames_per_scene, zoom_speed, enable_bobbing, enable_parallax
        )
        
        for f_idx, frame in enumerate(scene_frames):
            frame_mod = frame.copy()
            
            if enable_particles and particles is not None:
                update_and_draw_particles(frame_mod, particles, particle_type)
            if enable_speed_lines:
                draw_speed_lines(frame_mod, total_frame_counter)
                
            if last_frame is not None and f_idx < frames_transition:
                alpha = f_idx / max(1, frames_transition)
                blended = render_crossfade(last_frame, frame_mod, alpha)
                writer.write(blended)
            else:
                writer.write(frame_mod)
                
            total_frame_counter += 1
            
        final_mod_frame = scene_frames[-1].copy()
        if enable_particles and particles is not None:
            update_and_draw_particles(final_mod_frame, particles, particle_type)
        if enable_speed_lines:
            draw_speed_lines(final_mod_frame, total_frame_counter - 1)
        last_frame = final_mod_frame
        
        if progress_callback:
            progress_callback(int((idx + 1) / num_scenes * 90))
            
    writer.release()
    
    if log_callback:
        log_callback("Post-processing audio & final video encoding...")
        
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        
        video_clip = VideoFileClip(temp_avi_path)
        
        if audio_path and os.path.exists(audio_path):
            if log_callback:
                log_callback(f"Merging audio: {os.path.basename(audio_path)}...")
            audio_clip = AudioFileClip(audio_path)
            
            if audio_clip.duration < video_clip.duration:
                audio_clip = audio_clip.with_duration(video_clip.duration)
            else:
                audio_clip = audio_clip.with_duration(video_clip.duration)
                
            final_clip = video_clip.with_audio(audio_clip)
        else:
            if log_callback:
                log_callback("No background audio provided. Creating silent video.")
            final_clip = video_clip
            
        final_clip.write_videofile(
            output_mp4_path,
            codec="libx264",
            audio_codec="aac" if audio_path else None,
            fps=fps,
            logger=None
        )
        
        video_clip.close()
        if audio_path and os.path.exists(audio_path):
            audio_clip.close()
            
    except Exception as e:
        if log_callback:
            log_callback(f"Audio/Encoding error: {e}. Attempting direct format copy...")
        if os.path.exists(output_mp4_path):
            try:
                os.remove(output_mp4_path)
            except Exception:
                pass
        os.rename(temp_avi_path, output_mp4_path)
        
    finally:
        if os.path.exists(temp_avi_path):
            try:
                os.remove(temp_avi_path)
            except Exception:
                pass
                
    if progress_callback:
        progress_callback(100)
        
    elapsed = time.time() - start_time
    if log_callback:
        log_callback(f"Conversion finished in {elapsed:.2f} seconds!")
        log_callback(f"Output Video saved: {output_mp4_path}")
        
    return output_mp4_path
