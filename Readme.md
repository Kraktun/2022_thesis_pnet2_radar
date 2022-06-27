# Info

Code used to write the thesis available [here](https://thesis.unipd.it/bitstream/20.500.12608/29055/1/Michelon_Luca.pdf).

Most of the functions called in `main.py` come from [utils_flow.py](splashlib/nn/utils_flow.py), so refer to that file for the documentation.

The prediction script takes cares of removing the normalization if it is applied so that the results are in real dimensions (meters). Note that the normalization is removed after the prediction, or the results would not be coherent.

Multiple metrics are also printed on screen and saved in csv files. They are self-descriptive, or otherwise commented.

The predicted markers and point clouds (if the model has multiple outputs) are saved in csv files for later analysis and comparison with the ground truth. The ground truth and the input point clouds are also saved for convenience.

Tested with Tensorflow 2.5 and 2.7.

Pointnet2 is written in pure python so it can run on any OS, as done for [pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). 

The code expects this folder structure:

```unknown
- datasets
	-- '$yyyy-mm-dd_place'
		--- source
			---- train
				----- '$dataset_abc_000'
					------ 002.csv
					------ 003.csv
					------ ...
				----- '$dataset_abc_001'
					------ 002.csv
					------ 003.csv
					------ ...
				----- ...
	-- ...
```

where each csv file contains the point cloud at a given frame (moment in time).

**Note: the code for the preprocessing of the dataset is not available here as it includes algorithms written by other people.**
