# End-to-end patch-level Gleason Grading with Fast Fourier Convolutions

## Gleason Grades

Gleason Grading characterizes the aggressiveness of prostate cancer given a tissue sample
1. Based on the glandular architecture present within the sample
2. Higher grades mean more aggressive cancer

Challenges of Automating Gleason Grading
1. Whitespace overtakes majority of the sample image
2. Uneven color gradations across samples due to staining

## Fast Fourier Convolutions
Works as a substitute to regular convolutions. The Fourier unit transforms the spatial features of an image unto its frequency counterparts, globally updates this set of embedding, then  converts it back unto the spatial domain. A local Fourier unit conducts ordinary convolutions with small kernels which captures information within a local neighborhood. 

## SICAPV2 Data
This database contains prostate histological samples from 155 patients. Has a predetermined train and test split. Samples were split into patches of 512x512 images based on detected tumors
and cancer cells each segment is expertly annotated with a corresponding Gleason Grade (3, 4, 5, or No Cancer).

## How-to-use

1. Replace the `DATASET_LOC` variable in `main.py` with the SiCAPv2 dataset path.
2. Install requirements.txt with `python -m pip install -r requirements.txt`
3. Run the script with `python main.py`