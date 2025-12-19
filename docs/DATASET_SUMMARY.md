# HHI Surgical Instrument Dataset (YOLO format)

This document provides an overview of the Fraunhofer HHI Surgical Instrument
dataset used in the REFRAME project, including dataset structure, splits,
ground-truth availability, and class definitions.

---

## Dataset Structure

The dataset follows the standard YOLO format:

- `images/{train,val,test}`  
  Contains the RGB image files (JPG).

- `labels/{train,val,test}`  
  Contains YOLO ground-truth annotation files (`.txt`) with bounding boxes.

- `Surgical_Instrument_Recognition_YOLO.yaml`  
  Dataset configuration file defining train/val/test splits and class names.

- `class_distribution_{train,val,test}.csv/png`  
  Statistics and visualizations of class frequencies per split.

---

## Splits and Ground-Truth Availability

| Split | #Images | #Label files | Ground truth available |
|------|--------:|-------------:|------------------------|
| train | 908 | 870 | Yes (YOLO txt; some images contain no annotations) |
| val | 302 | 289 | Yes (YOLO txt; some images contain no annotations) |
| test | 305 | 288 | Yes (YOLO txt; some images contain no annotations) |

**Note:**  
In YOLO format, images without objects do not necessarily have a corresponding
label file. Therefore, fewer label files than images is expected.

---

## Classes (23)

The dataset defines 23 surgical instrument classes:

0. Adson-Brown Forceps  
1. Anatomical Forceps  
2. Bipolar Forceps  
3. Dissecting Swabs  
4. Dressing Forceps  
5. Elevator double-ended  
6. Gauze Ball  
7. Grasping Forceps  
8. Jameson Dissecting Scissors  
9. Langenbeck Retractor  
10. Metzenbaum Dissecting Scissors  
11. Mini-Langenbeck Retractor  
12. Mosquito Forceps curved  
13. Needle Holder  
14. Overholt Dissecting Forceps  
15. Pean Forceps straight  
16. Roux Retractor  
17. Scalpel  
18. Self-Retaining Retractor  
19. Surgical Forceps  
20. Surgical Scissors  
21. Towel Clamp  
22. Volkmann Retractor

---

## Annotation Type

- **Task type:** Object Detection  
- **Annotation format:** YOLO bounding boxes (`class_id x_center y_center width height`)  
- **Segmentation masks:** Not available (detection-only dataset)

---

## Remarks

- The provided YOLO11 model is a **detection model** and outputs bounding boxes
  with class IDs and confidence scores.
- No instance or semantic segmentation masks are provided in the dataset.
- For downstream compatibility with segmentation-based pipelines, box-based
  binary masks could be generated as a post-processing step if required.
