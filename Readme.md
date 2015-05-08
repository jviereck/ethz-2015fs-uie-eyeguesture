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

Will write the file `landmarks.csv` under `tools/`.
