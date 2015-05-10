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

## Coordinate systems

The x/y coordinates from the BioId are encoded as follows

```
  0----> x
  |
  | y
  v
```

The data read via the scipy.misc method is using the coordiante system as follows:

```python
#  0----> y
#  |
#  | x
#  v

data = misc.imread(join(img_root_path, 'BioID_%04d.pgm' % (img_id)))
pixel_value = data[x, y]
```

When the read `data` object is reshaped to a one-dimensional-array, then the array
index goes along the y coordinate first:

```python
# Reshaping the data.
reshaped_data = data.reshape(data.shape[0] * data.shape[1]);

# Then the following assertion holds connecting the original data shape and
# the resulting reshaped data:
idx = y + x * data.shape[1];
assert data[x, y] == reshaped_data[idx]
```

## Start the training process

Training the system is done by invoking the `backend/train.py` python script and passing in a CSV file containing the image-ids to train and the landmarks of these training images. Such a CSV file can be obtained by running `tools/filter_eye_distance.py`.

```bash
$ cd backend
$ python train.py ../tools/landmarks_filtered.csv ../../bioid_data/BioID-FaceDatabase-V1.2
```

# TODOS

- center the images before starting the training

# Useful references

- Face Alignment implementation using Matlab: https://github.com/jwyang/face-alignment

