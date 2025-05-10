# MaCENCA 

The files in this directory are for training and visualising MaCENCA, a mass conserving version of Alexander Mordvinstev's NCA.

## To train 

Run the [MassConservingNCA.ipynb](MassConsevingNCA.ipynb) file top to bottom. Make sure to:
- Supply an image and its path in the <i><b>Define Image Path</b></i> section.
- Set the device correctly, default is "cuda:0" , it will train on the "cpu" but ver slowly so we highly discourage that.
- Set the hypermarkets in the <i><b>Hyperparameters</b></i> section, the larger the CA size the more Vram it requires, we suggest a size of 50X50 or 70X70.

# To Visualise 

Run the [visualise.py](visualise.py) file. Make sure to:
- Set the DEVICE and hyperparameters correctly in the <i><b>Device and Hyperparameters</b></i> section.
- Give the correct <i><b>Model_name.pth</b></i> in the <i><b>Model settings</b></i> section. Here you can also set the starting mass of the seed cell to play around with. 