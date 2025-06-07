# NematodeMorphoAnalyzer

## 🧬 Overview

![Main Preview](assets/img/preview.png)

**NematodeMorphoAnalyzer** is an open-source Python project designed to automatically analyze nematode images and extract **morphological features** such as body length, width, and tail shape. The pipeline applies specific procedures depending on the **coiling level** (Level 1, Level 2, or Level 3) assigned to each sample.

## 🎯 Highlights

| 🔍 Feature                     | 🧾 Description                                                                 |
| ----------------------------- | ----------------------------------------------------------------------------- |
| 🧠 **Morphological Analysis**  | Automatically detects morphological features from raw images                 |
| 🌀 **Coiling Level Handling**  | Applies different analysis routines for each nematode coiling level          |
| 📏 **Precise Measurements**    | Calculates length, width, area, tail geometry, and more                      |
| 🖼️ **Image Processing**        | Uses `OpenCV`, `skimage`, `PIL`, and `matplotlib` for preprocessing and visualization |
| 📊 **Structured Export**       | Saves results as structured tables using `pandas`                            |

## ⚙️ Workflow Overview

1. **Image Loading**: Reads input images and associated metadata (e.g. coiling level).
2. **Preprocessing**: Inversion, binarization, skeletonization, and morphological cleanup.
3. **Feature Detection**: Measures width, length, area, tail shape, and more.
4. **Conditional Processing**: Applies analysis routines based on coiling level.
5. **Output & Export**: Generates CSV summaries and optionally annotated images.


## 📁 Project Structure
```
nematode_morpho_analyzer/
├── data/                     # Input images and metadata
│   ├── level1/
│   ├── level2/
│   ├── level3/
│   └── labels.csv
├── outputs/                  # Results (CSV files, annotated images)
├── src/                      # Main source code
│   └── main.py               # Main entry point
├── assets/                  # Documentation assets (e.g., images)
│   └── img/
├── requirements.txt         # Python dependencies
├── README.md
└── LICENSE
```
## 🖥️ Sample Output Table
| Image ID       | Coiling Level | Length (px) | Avg Width (px) | Tail Shape | Area (px²) |
|----------------|---------------|-------------|----------------|------------|------------|
| sample001.png  | Level 1       | 435         | 28.5           | tapered    | 8120       |
| sample002.png  | Level 3       | 310         | 35.2           | rounded    | 8904       |

```
## 🌟 License
This project is open-source. Feel free to use, modify, and contribute! 🚀
