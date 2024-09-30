# Ship-Detection-Super-Res

# Enhanced Ship Detection through Deep-Learning-Based Super-Resolution Embedding

### Authors:
- Akshit Sharma
- Satvik Pandey

## Project Description
This project focuses on enhancing the detection of small objects, specifically ships, in satellite and aerial imagery using deep learning and super-resolution techniques. The project integrates Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) to improve image quality and YOLOv5 for ship detection.

## Methodology
We employ ESRGAN for image super-resolution, which significantly improves the clarity of low-resolution maritime images. After super-resolution, YOLOv5 is used for accurate and real-time detection of ships. The key innovation in this project is the use of attention mechanisms in the super-resolution process, which ensures that small ships are better preserved during image upscaling.

## Project Structure
- `CS543_Project_Report.pdf`: The project report detailing methodology and results.
- `CS543.ipynb`: Jupyter notebook containing the code for preprocessing, model training, and ship detection.
- `data/`: Contains sample data for testing, if any.
- 'images/': Contains sample input,output and final bounding boxes sample image.
- `src/`: Contains any additional Python scripts for preprocessing or analysis.

## Requirements
- Python 3.8+
- TensorFlow 
- YOLOv5
- ESRGAN

