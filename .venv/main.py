#!/usr/bin/env python3
"""
Synthwave Low-Poly Terrain — GUI + STL Viewer (Perspective-enforced)
- Larger mountains in the foreground, smaller toward the horizon
- Side walls stronger near the viewer, tapering in the distance
- Distinct small horizon ridge
- Binary STL export; no external libs

Mouse: Left-drag=rotate | Right-drag=pan | Wheel=zoom
"""

import math
import random
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- Terrain Generator with perspective-enforced mountains ----------

class TerrainGenerator:
    def __init__(self):
        # --- Size / density ---
        self.SEED = 42
        self.SIZE_X = 200.0
        self.SIZE_Y = 200.0
        self.HEIGHT_SCALE = 28.0
        self.GRID_NX = 60
        self.GRID_NY = 60

        # --- Style / shaping ---
        self.HORIZON_Y_RATIO = 0.80   # where the horizon ridge sits (0..1 front->back)
        self.HORIZON_BAND_WIDTH = 0.06  # width of the horizon ridge band (fraction of SIZE_Y^2 in Gaussian)
        self.HORIZON_RIDGE_GAIN = 0.45  # strength of the distant ridge (kept small to look far away)
        self.HORIZON_STEEPNESS = 14.0   # sharpness of horizon rise

        # Side mountains (left/right)
        self.EDGE_FALLOFF = 0.16        # smaller => steeper side rise
        self.EDGE_DEPTH_TAPER = 1.4     # >1: edges are much stronger in front than far

        # Foreground vs distance scaling (perspective)
        self.FRONT_PEAK_GAIN = 0.85     # boosts near heights
        self.FRONT_PEAK_GAMMA = 1.6     # how fast boost fades with distance
        self.DISTANCE_ATTEN = 0.55      # attenuates heights as y→horizon
        self.DISTANCE_ATTEN_GAMMA = 1.3

        # Valley flattening
        self.VALLEY_STRENGTH = 0.38     # flattens center foreground for a “valley”
        self.VALLEY_WIDTH = 0.08        # width of x-centered valley (fraction of SIZE_X^2)

        # Faceting / low-poly wobble
        self.WOBBLE = 0.22

        # Mesh cache
        self.facets = []
        self.bounds = (0, 0, 0, 0, 0, 0)

    @staticmethod
    def _vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    @staticmethod
    def _vcross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    @staticmethod
    def _vnorm(v):
        m = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
        return (v[0]/m, v[1]/m, v[2]/m)

    def _ridge_height(self, x, y, size_x, size_y, center_x, horizon_y):
        """Height = near-boosted side walls + small distant horizon ridge + wobble − valley flattening,
        with explicit depth attenuation to force perspective."""
        xn = x / size_x
        yn = y / size_y

        # -------- Side mountains (left/right), stronger in front, tapering with distance --------
        edge_scale = (size_x * self.EDGE_FALLOFF + 1e-6)
        edge_left  = 1.0 - math.exp(- x / edge_scale)
        edge_right = 1.0 - math.exp(- (size_x - x) / edge_scale)
        edge_term = max(edge_left, edge_right)

        # Taper side strength with distance so front walls > distant walls
        depth_taper = (1.0 - yn) ** self.EDGE_DEPTH_TAPER  # 1 at front, -> 0 toward far
        edge_term *= depth_taper

        # -------- Distant horizon ridge (kept small to read "far away") --------
        band_sigma = self.HORIZON_BAND_WIDTH * size_y * size_y
        band = math.exp(-((y - horizon_y) ** 2) / max(1e-6, band_sigma))
        horizon_t = 1.0 / (1.0 + math.exp(-(y - horizon_y) * (self.HORIZON_STEEPNESS / max(1.0, size_y))))
        horizon_ridge = self.HORIZON_RIDGE_GAIN * horizon_t * band  # small & sharp

        # -------- Foreground boost + distance attenuation (perspective) --------
        # Near boost: strong at front, fades with distance
        front_boost = 1.0 + self.FRONT_PEAK_GAIN * ((1.0 - yn) ** self.FRONT_PEAK_GAMMA)
        # Distance attenuation: reduces overall amplitude toward horizon
        dist_att = (1.0 - self.DISTANCE_ATTEN * (yn ** self.DISTANCE_ATTEN_GAMMA))

        # -------- Valley flattening in center foreground --------
        valley = 1.0 - self.VALLEY_STRENGTH * math.exp(-((x - center_x) ** 2) / (self.VALLEY_WIDTH * size_x * size_x)) * (1.0 - yn)**1.2

        # -------- Low-frequency trig wobble for crisp low-poly facets --------
        ang1 = 2.2 * math.pi * xn + 0.7 * math.pi * yn
        ang2 = 1.1 * math.pi * xn - 1.9 * math.pi * yn
        wob = (math.sin(ang1) * math.cos(ang2)) * self.WOBBLE \
            + (math.cos(ang1 * 0.5) * math.sin(ang2 * 0.6)) * (self.WOBBLE * 0.5)

        # Base structure: side walls + horizon ridge
        base = 0.75 * edge_term + horizon_ridge

        # Apply perspective shaping and valley
        h = max(0.0, (base * front_boost * dist_att * valley) + wob * 0.35)
        return h

    def build(self):
        random.seed(int(self.SEED))
        nx = max(4, int(self.GRID_NX))
        ny = max(4, int(self.GRID_NY))
        sx, sy = float(self.SIZE_X), float(self.SIZE_Y)

        xs = [i * (sx / (nx - 1)) for i in range(nx)]
        ys = [i * (sy / (ny - 1)) for i in range(ny)]
        cx = sx * 0.5
        hy = sy * self.HORIZON_Y_RATIO

        heights = [[0.0]*nx for _ in range(ny)]
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                heights[j][i] = self._ridge_height(x, y, sx, sy, cx, hy) * self.HEIGHT_SCALE

        facets = []
        # Top surface (two tris per quad)
        for j in range(ny - 1):
            for i in range(nx - 1):
                x0, x1 = xs[i], xs[i+1]
                y0, y1 = ys[j], ys[j+1]
                z00 = heights[j][i]
                z10 = heights[j][i+1]
                z01 = heights[j+1][i]
                z11 = heights[j+1][i+1]
                if (i + j) % 2 == 0:
                    tri1 = ((x0,y0,z00),(x1,y0,z10),(x1,y1,z11))
                    tri2 = ((x0,y0,z00),(x1,y1,z11),(x0,y1,z01))
                else:
                    tri1 = ((x0,y0,z00),(x1,y0,z10),(x0,y1,z01))
                    tri2 = ((x1,y0,z10),(x1,y1,z11),(x0,y1,z01))
                facets.append(tri1); facets.append(tri2)

        # Skirts to z=0 (close the mesh)
        j = 0
        for i in range(nx - 1):
            x0, x1 = xs[i], xs[i+1]; y = ys[j]; z0 = heights[j][i]; z1 = heights[j][i+1]
            facets.append(((x0,y,0.0),(x1,y,0.0),(x1,y,z1)))
            facets.append(((x0,y,0.0),(x1,y,z1),(x0,y,z0)))
        j = ny - 1
        for i in range(nx - 1):
            x0, x1 = xs[i], xs[i+1]; y = ys[j]; z0 = heights[j][i]; z1 = heights[j][i+1]
            facets.append(((x1,y,0.0),(x0,y,0.0),(x1,y,z1)))
            facets.append(((x0,y,0.0),(x0,y,z0),(x1,y,z1)))
        i = 0
        for j in range(ny - 1):
            x = xs[i]; y0, y1 = ys[j], ys[j+1]; z0 = heights[j][i]; z1 = heights[j+1][i]
            facets.append(((x,y0,0.0),(x,y1,0.0),(x,y1,z1)))
            facets.append(((x,y0,0.0),(x,y1,z1),(x,y0,z0)))
        i = nx - 1
        for j in range(ny - 1):
            x = xs[i]; y0, y1 = ys[j], ys[j+1]; z0 = heights[j][i]; z1 = heights[j+1][i]
            facets.append(((x,y1,0.0),(x,y0,0.0),(x,y1,z1)))
            facets.append(((x,y0,0.0),(x,y0,z0),(x,y1,z1)))

        # Flat bottom
        facets.append(((0.0,0.0,0.0),(sx,0.0,0.0),(sx,sy,0.0)))
        facets.append(((0.0,0.0,0.0),(sx,sy,0.0),(0.0,sy,0.0)))

        # Bounds for viewer
        xs_all = [v[0] for tri in facets for v in tri]
        ys_all = [v[1] for tri in facets for v in tri]
        zs_all = [v[2] for tri in facets for v in tri]
        self.bounds = (min(xs_all), max(xs_all), min(ys_all), max(ys_all), min(zs_all), max(zs_all))
        self.facets = facets

    def write_binary_stl(self, path):
        def vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
        def vcross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
        def vnorm(v):
            m = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
            return (v[0]/m, v[1]/m, v[2]/m)

        with open(path, "wb") as f:
            header = b"SYNTHWAVE LOWPOLY TERRAIN".ljust(80, b" ")
            f.write(header)
            f.write(struct.pack("<I", len(self.facets)))
            for (a, b, c) in self.facets:
                n = vnorm(vcross(vsub(b, a), vsub(c, a)))
                f.write(struct.pack("<3f", *n))
                f.write(struct.pack("<3f", *a))
                f.write(struct.pack("<3f", *b))
                f.write(struct.pack("<3f", *c))
                f.write(struct.pack("<H", 0))

# ---------- Minimal 3D Viewer (Tk Canvas) ----------

class STLViewer(ttk.Frame):
    def __init__(self, master, get_facets_callable, get_bounds_callable):
        super().__init__(master)
        self.get_facets = get_facets_callable
        self.get_bounds = get_bounds_callable
        self.canvas = tk.Canvas(self, width=720, height=480, bg="#101018", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.rot_x = math.radians(30)
        self.rot_y = math.radians(-30)
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.2

        self._dragging = None
        self._last_mouse = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1/1.1))
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.light_dir = self._norm((0.3, -0.5, 1.0))

    def _zoom(self, k): self.zoom *= k; self.redraw()
    def _on_left_down(self, e): self._dragging = "rotate"; self._last_mouse = (e.x, e.y)
    def _on_right_down(self, e): self._dragging = "pan";    self._last_mouse = (e.x, e.y)
    def _on_left_drag(self, e):
        if self._dragging != "rotate": return
        dx, dy = e.x - self._last_mouse[0], e.y - self._last_mouse[1]
        self.rot_y += dx * 0.01; self.rot_x += dy * 0.01
        self._last_mouse = (e.x, e.y); self.redraw()
    def _on_right_drag(self, e):
        if self._dragging != "pan": return
        dx, dy = e.x - self._last_mouse[0], e.y - self._last_mouse[1]
        self.pan_x += dx; self.pan_y += dy
        self._last_mouse = (e.x, e.y); self.redraw()
    def _on_wheel(self, e): self._zoom(1.1 if e.delta > 0 else 1/1.1)

    @staticmethod
    def _norm(v):
        m = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
        return (v[0]/m, v[1]/m, v[2]/m)
    @staticmethod
    def _sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    @staticmethod
    def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    @staticmethod
    def _dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _rotate(self, v, rx, ry):
        x, y, z = v
        cx, sx = math.cos(rx), math.sin(rx)
        y, z = (y*cx - z*sx, y*sx + z*cx)
        cy, sy = math.cos(ry), math.sin(ry)
        x, z = (x*cy + z*sy, -x*sy + z*cy)
        return (x, y, z)

    def _project(self, v3, bbox, w, h):
        (minx, maxx, miny, maxy, minz, maxz) = bbox
        cx, cy, cz = (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2
        sx, sy, sz = (maxx-minx or 1), (maxy-miny or 1), (maxz-minz or 1)
        scale = max(sx, sy, sz)

        x = (v3[0]-cx)/scale; y = (v3[1]-cy)/scale; z = (v3[2]-cz)/scale
        x, y, z = self._rotate((x, y, z), self.rot_x, self.rot_y)
        s = 0.9 * min(w, h) * self.zoom
        return (x*s + w*0.5 + self.pan_x, -y*s + h*0.5 + self.pan_y, z)

    def redraw(self):
        self.canvas.delete("all")
        facets = self.get_facets()
        if not facets:
            self._text("Click Generate to build terrain"); return
        bbox = self.get_bounds(); w = self.canvas.winfo_width(); h = self.canvas.winfo_height()

        drawlist = []
        for (a, b, c) in facets:
            ax, ay, az = self._project(a, bbox, w, h)
            bx, by, bz = self._project(b, bbox, w, h)
            cx, cy, cz = self._project(c, bbox, w, h)
            # Back-face cull
            ux, uy = (bx-ax, by-ay); vx, vy = (cx-ax, cy-ay)
            if ux*vy - uy*vx <= 0: continue

            n_world = self._norm(self._cross(self._sub(b, a), self._sub(c, a)))
            intensity = max(0.05, self._dot(n_world, self.light_dir))
            gray = int(40 + 215 * intensity)
            fill = f"#{gray:02x}{int(0.6*gray):02x}{int(0.9*gray):02x}"
            drawlist.append(((az+bz+cz)/3.0, fill, (ax,ay,bx,by,cx,cy)))

        drawlist.sort(key=lambda t: t[0])
        for _, fill, coords in drawlist:
            self.canvas.create_polygon(coords, fill=fill, outline="", width=0)
        # Sparse wire overlay
        for _, _, coords in drawlist[::max(1, len(drawlist)//400)]:
            self.canvas.create_line(coords[0:4], fill="#202040")
            self.canvas.create_line(coords[2:6], fill="#202040")
            self.canvas.create_line(coords[4:6] + coords[0:2], fill="#202040")

    def _text(self, s):
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        self.canvas.create_text(w//2, h//2, text=s, fill="#aab", font=("Segoe UI", 14))

# ---------- GUI ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthwave Terrain — Perspective Generator & STL Viewer")
        self.geometry("1100x640")

        self.gen = TerrainGenerator()
        self.columnconfigure(1, weight=1); self.rowconfigure(0, weight=1)

        self.controls = ttk.Frame(self, padding=10); self.controls.grid(row=0, column=0, sticky="ns")
        self.viewer = STLViewer(self, self._get_facets, self._get_bounds); self.viewer.grid(row=0, column=1, sticky="nsew")
        self._build_controls()

    def _get_facets(self): return self.gen.facets
    def _get_bounds(self): return self.gen.bounds

    def _build_controls(self):
        def add_num(parent, label, var, fr, to, is_int=False):
            row = ttk.Frame(parent); row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=20).pack(side="left")
            e = ttk.Entry(row, width=8, textvariable=var, justify="right"); e.pack(side="left", padx=5)
            s = ttk.Scale(row, from_=fr, to=to, command=lambda v: var.set(int(float(v)) if is_int else round(float(v),3)))
            s.pack(side="left", fill="x", expand=True, padx=5)
            def sync(*_):
                try: s.set(float(var.get()))
                except: pass
            var.trace_add("write", sync); sync()

        # Vars
        self.v_seed = tk.IntVar(value=self.gen.SEED)
        self.v_sx = tk.DoubleVar(value=self.gen.SIZE_X); self.v_sy = tk.DoubleVar(value=self.gen.SIZE_Y)
        self.v_h = tk.DoubleVar(value=self.gen.HEIGHT_SCALE)
        self.v_nx = tk.IntVar(value=self.gen.GRID_NX); self.v_ny = tk.IntVar(value=self.gen.GRID_NY)

        self.v_hyr = tk.DoubleVar(value=self.gen.HORIZON_Y_RATIO)
        self.v_hbw = tk.DoubleVar(value=self.gen.HORIZON_BAND_WIDTH)
        self.v_hrg = tk.DoubleVar(value=self.gen.HORIZON_RIDGE_GAIN)
        self.v_hst = tk.DoubleVar(value=self.gen.HORIZON_STEEPNESS)

        self.v_edge = tk.DoubleVar(value=self.gen.EDGE_FALLOFF)
        self.v_edge_taper = tk.DoubleVar(value=self.gen.EDGE_DEPTH_TAPER)

        self.v_front_gain = tk.DoubleVar(value=self.gen.FRONT_PEAK_GAIN)
        self.v_front_gamma = tk.DoubleVar(value=self.gen.FRONT_PEAK_GAMMA)
        self.v_dist_att = tk.DoubleVar(value=self.gen.DISTANCE_ATTEN)
        self.v_dist_gamma = tk.DoubleVar(value=self.gen.DISTANCE_ATTEN_GAMMA)

        self.v_valley = tk.DoubleVar(value=self.gen.VALLEY_STRENGTH)
        self.v_valley_w = tk.DoubleVar(value=self.gen.VALLEY_WIDTH)

        self.v_wobble = tk.DoubleVar(value=self.gen.WOBBLE)

        ttk.Label(self.controls, text="Size & Density", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        add_num(self.controls, "Seed", self.v_seed, 0, 9999, True)
        add_num(self.controls, "Size X (mm)", self.v_sx, 60, 400)
        add_num(self.controls, "Size Y (mm)", self.v_sy, 60, 400)
        add_num(self.controls, "Height Scale", self.v_h, 5, 60)
        add_num(self.controls, "Grid NX", self.v_nx, 10, 140, True)
        add_num(self.controls, "Grid NY", self.v_ny, 10, 140, True)

        ttk.Separator(self.controls).pack(fill="x", pady=6)
        ttk.Label(self.controls, text="Perspective & Horizon", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        add_num(self.controls, "Horizon Y Ratio", self.v_hyr, 0.6, 0.95)
        add_num(self.controls, "Horizon Band Width", self.v_hbw, 0.01, 0.15)
        add_num(self.controls, "Horizon Ridge Gain", self.v_hrg, 0.05, 1.0)
        add_num(self.controls, "Horizon Steepness", self.v_hst, 4.0, 24.0)

        ttk.Separator(self.controls).pack(fill="x", pady=6)
        ttk.Label(self.controls, text="Side Walls & Depth", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        add_num(self.controls, "Edge Falloff", self.v_edge, 0.05, 0.5)
        add_num(self.controls, "Edge Depth Taper", self.v_edge_taper, 0.5, 2.5)
        add_num(self.controls, "Front Peak Gain", self.v_front_gain, 0.0, 1.5)
        add_num(self.controls, "Front Peak Gamma", self.v_front_gamma, 0.5, 3.0)
        add_num(self.controls, "Distance Atten", self.v_dist_att, 0.0, 1.0)
        add_num(self.controls, "Distance Gamma", self.v_dist_gamma, 0.5, 3.0)

        ttk.Separator(self.controls).pack(fill="x", pady=6)
        ttk.Label(self.controls, text="Valley & Facets", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        add_num(self.controls, "Valley Strength", self.v_valley, 0.0, 0.9)
        add_num(self.controls, "Valley Width", self.v_valley_w, 0.02, 0.20)
        add_num(self.controls, "Wobble", self.v_wobble, 0.0, 0.6)

        btns = ttk.Frame(self.controls); btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Generate", command=self._on_generate).pack(side="left", fill="x", expand=True, padx=(0,5))
        ttk.Button(btns, text="Export STL", command=self._on_export).pack(side="left", fill="x", expand=True)

        help_txt = ("Viewer: Left-drag rotate • Right-drag pan • Wheel zoom\n"
                    "Tip: Keep Horizon Ridge Gain low so distance looks small.")
        ttk.Label(self.controls, text=help_txt, wraplength=260, foreground="#556").pack(fill="x", pady=6)

    def _pull(self):
        g = self.gen
        g.SEED = int(self.v_seed.get())
        g.SIZE_X = float(self.v_sx.get()); g.SIZE_Y = float(self.v_sy.get()); g.HEIGHT_SCALE = float(self.v_h.get())
        g.GRID_NX = max(4, int(self.v_nx.get())); g.GRID_NY = max(4, int(self.v_ny.get()))

        g.HORIZON_Y_RATIO = float(self.v_hyr.get())
        g.HORIZON_BAND_WIDTH = float(self.v_hbw.get())
        g.HORIZON_RIDGE_GAIN = float(self.v_hrg.get())
        g.HORIZON_STEEPNESS = float(self.v_hst.get())

        g.EDGE_FALLOFF = float(self.v_edge.get())
        g.EDGE_DEPTH_TAPER = float(self.v_edge_taper.get())

        g.FRONT_PEAK_GAIN = float(self.v_front_gain.get())
        g.FRONT_PEAK_GAMMA = float(self.v_front_gamma.get())
        g.DISTANCE_ATTEN = float(self.v_dist_att.get())
        g.DISTANCE_ATTEN_GAMMA = float(self.v_dist_gamma.get())

        g.VALLEY_STRENGTH = float(self.v_valley.get())
        g.VALLEY_WIDTH = float(self.v_valley_w.get())
        g.WOBBLE = float(self.v_wobble.get())

    def _on_generate(self):
        try:
            self._pull()
            self.gen.build()
            self.viewer.redraw()
        except Exception as e:
            messagebox.showerror("Generate failed", str(e))

    def _on_export(self):
        if not self.gen.facets:
            if not messagebox.askyesno("No mesh yet", "No current mesh. Generate now?"):
                return
            self._on_generate()
            if not self.gen.facets:
                return
        path = filedialog.asksaveasfilename(title="Save STL", defaultextension=".stl",
                                            filetypes=[("STL files", "*.stl"), ("All files", "*.*")])
        if not path: return
        try:
            self.gen.write_binary_stl(path)
            messagebox.showinfo("Export complete", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

if __name__ == "__main__":
    App().mainloop()
