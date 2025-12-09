# 🌾 Grain Segmentation & YOLO Labeling Pipeline  
### Interactive Multi-Threshold Segmentation • Stability Maps • Optimization • Full Batch Label Export

This notebook provides a complete **interactive and automated workflow** for segmenting small grain-like objects using multi-threshold *stability maps* and exporting YOLO-format bounding boxes.

It is designed for noisy, uneven-lighting datasets and supports both manual exploration and fully automated batch processing.

---

## 🚀 Features

- **Live interactive segmentation** (auto-updates when parameters change)
- Single-threshold & multi-threshold segmentation
- Multi-threshold **stability maps** with adjustable ranges
- One-click **YOLO label export**
- **Local optimization** of `(t_min, t_max)` for each image
- **Full automatic batch labeling** for an entire dataset
- Smooth plot replacement (no flicker), thanks to `clear_output(wait=True)`
- Clean, modular pipeline functions shared across all UI parts

---

## 📂 Dataset Structure

Your dataset folder must follow:

```
test/
├── images/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
└── labels/
```

`labels/` is created automatically if missing.

---

## 📦 Installation

Install required Python packages:

```bash
pip install opencv-python numpy matplotlib ipywidgets
```

Enable widgets:

```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## 📘 Notebook Structure

The notebook is divided into **four main parts**, all powered by the same backend segmentation functions.

---

### 1️⃣ Manual Single-Threshold Segmentation  
A basic interactive mode:

- Background subtraction  
- Gaussian blur  
- Adjustable threshold  
- Morphological cleaning  
- Bounding box overlay  
- YOLO export  

Best for basic dataset inspection.

---

### 2️⃣ Manual Multi-Threshold Stability Map  
More robust segmentation using:

- A range of thresholds (`t_min → t_max`)  
- Multiple steps (`n_steps`)  
- Pixel stability (`min_hits`)  
- Produces smoother, more reliable masks  

Segmentation is re-run instantly on any parameter change.

---

### 3️⃣ Multi-Threshold Stability + Optimization  
Same as Part 2, but with an **Optimize** button.

Pressing **🔍 Optimize t_min/t_max**:

1. Searches the slider ranges for the best `(t_min, t_max)`  
2. Selects the pair that maximizes object detections  
3. Updates sliders automatically  
4. Re-runs segmentation  

YOLO export still available.

---

### 4️⃣ Full Automatic Optimization & YOLO Labeling (No UI)

Run the entire dataset with:

```python
optimize_and_label_all_images()
```

This will:

1. For each image:
   - Run optimization
   - Run segmentation
   - Save YOLO labels (.txt)
2. Produce logs in the notebook
3. Create one `.txt` annotation file per image  

---

## 🧠 Optimization Notes

If you see:

```
Best t_min = X, t_max = Y, kept_count = 0
```

you should try:

- Lowering `min_hits`
- Increasing `n_steps`
- Expanding `tmin_range`, `tmax_range`
- Adjusting blur / background kernel size
- Relaxing `min_area` / `max_area`
- Switching `invert` True/False

---
---

## ⚡ Parallel Batch Optimization Script

For large datasets, running the full optimization inside the notebook can be slow.  
This project also includes a standalone Python script that parallelizes the work across multiple CPU cores.

The script:

- Scans the `images/` folder
- Runs `(t_min, t_max)` optimization **per image** in parallel
- Runs the stability-based segmentation
- Writes YOLO labels into the `labels/` folder

Example usage:

```bash
python grain_seg_optimal.py


## 🛠️ Troubleshooting

### No objects detected  
Increase range of thresholds, adjust preprocessing, or reduce area filtering.

### YOLO boxes look incorrect  
Remember YOLO uses **normalized center coordinates**.

### Plot jumps or blinks  
Ensure your Jupyter version supports `wait=True`.

---

## 📄 License  
MIT License — free to use and modify.

---

## 🙋 Need more?  
- A CLI tool  
- A dedicated Python package  
- Batch visualization exports  
- Advanced adaptive threshold optimization  

