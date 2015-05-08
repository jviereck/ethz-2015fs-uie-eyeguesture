# Eye Guesture Project

## Convert .pgm to .png images

To convert `.pgm` images (like the one from BioID) to `.png` images, run:

```bash
$ cd tools
$ python convert_pgm_png.py ../../bioid_data/BioID-FaceDatabase-V1.2
```

This will convert the images and write them under `tools/`.

## Generate landmarks.csv file

To generate a single `.csv` file from the bioid landmarks:

```bash
$ cd tools
$ python gather_points.py ../../bioid_data/points_20
```

Will write the file `tools/landmarks.csv`.

## Filter images by eye distances

To make the first training easier and use faces that are recorded roughly at the same scale, the entire BioID face dataset if filtered by the eye distance. This is done by the following script:

```bash
$ cd tools
$ python filter_eye_distance.py
```

Which generates a new file `tools/landmarks_filtered.csv` based on `tools/landmarks.csv`, that only contains faces with a given min and max eye distance. (The min/max eye distance is adjustable in the `filter_eye_distance.py` file.)

Copying the previous with `convert_pgm_png.py` generated `.png` images for the filter-selected ones (assuming they are located under `tools/output`) to `tools/output_filtered` via:

```bash
$ cd tools
$ python copy_filtered_images.py
```

## Using liblinear

To install the `liblinear` library on OSX using Homebrew, run:

```bash
$ brew install liblinear
```

The liblinear repo comes with a [python wrapper](https://github.com/ninjin/liblinear/tree/master/python), which was imported under `liblinear/`.

# Useful references

- Face Alignment implementation using Matlab: https://github.com/jwyang/face-alignment

