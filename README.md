Datasets used for training, test and validation

We used image set BBBC039v1 Caicedo et al. 2018, available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].
We used the image set BBBC006v1 from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].


Required packages and citatons:

PyTorch for Dataset, DataLoader, tensors, neural network modules, and functional utilities. https://docs.pytorch.org/docs/2.11/data.html

NumPy for array operations, statistics, and numerical preprocessing. https://numpy.org/doc/stable/

pandas for data tables, CSV I/O, and metrics storage. https://pandas.pydata.org/docs/

Pillow for PNG image loading and resizing. https://pillow.readthedocs.io/en/stable/reference/Image.html

Matplotlib for plotting images and training curves with imshow and line plots. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

scikit-learn for train_test_split. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

tifffile for TIFF-based microscopy image loading. https://iridescent.ink/tifffile/tifffile.html

For the models we used UNet, DoubleConv
We measured IoU and Dice loss 