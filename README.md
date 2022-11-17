# 3D Vessel Tree Generator

This repository is a refactored version of the [3D Vessel Generator](https://github.com/kritiyer/vessel_tree_generator). For more documentation, please refer to the original repository.

## Major Changes

The major changes are summarized below:

- new `Generator` class that handles the generation of centerlines, surfaces, and projections
- nested tree generator

## Usage

All configurations are specified using a nested series of [YAML](https://yaml.org/) files and loaded with the [Hydra](https://hydra.cc/) framework. To generate vessels with the default configuration, simply run

```commandline
python main.py
```

More specific configurations can be specified by passing in command-line arguments as well:

```commandline
# Generate 300 trees (default is 10)
python main.py --n_trees=300
```

As with the original version, the script will produce three sets of files for each generated geometry:

- surface coordinates (npy)
- geometry generation parameters (json)
- binary projection images (png)

The output directory is structued as follows:

-- save_path directory\
&nbsp; &nbsp; --> dataset_name directory\
&nbsp; &nbsp;&nbsp; &nbsp; --> labels\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0000.npy\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0001.npy\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0002.npy\
&nbsp; &nbsp;&nbsp; &nbsp; --> info\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0000.info.0\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0001.info.0\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - 0002.info.0\
&nbsp; &nbsp;&nbsp; &nbsp; --> images (only if using `--generate_projections` flag)\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - image0000a.png, image0000b.png, image0000c.png\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - image0001a.png, image0001b.png, image0001c.png\
&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; - image0002a.png, image0002b.png, image0002c.png

## Citation

If you find this work useful, please cite the following:

**A Multi-Stage Neural Network Approach for Coronary 3D Reconstruction from Uncalibrated X-ray Angiography Images. Iyer, K., Nallamothu, B.K., Figueroa, C.A., Nadakuditi, R.R. *In submission*.**
