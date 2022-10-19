# TSNE_Visualizer

Visualizing the learned representations is extremely important in deep learning for interpretations. Additionally, tracing back to the original inputs from the features will provide a better understanding of the learned representations. In order to efficiently conduct this task and support research, I created a simple interactive tool using streamlit and bokeh.

### Input File format

The input file should be in .h5 format with following data:

  - 'images' : Original Unnormalized images
  - 'labels' : Labels (should be integers)
  - 'tsne'   : Array of the features after applying tsne
  
The following code block can be used as guide to save the files:

```
import h5py
hf = h5py.File("~/data.h5", 'w')
hf.create_dataset('images', data= "image data")
hf.create_dataset('labels', data= "label data")
hf.create_dataset('tsne', data= "tsne data")
hf.close()
```



### Installation Guide

```
pip install -r requirements.txt
```

### Run app

```
streamlit run ~/visualizer.py --server.maxUploadSize 1000
```

## Demo Video





https://user-images.githubusercontent.com/52663918/196756393-236d5ad5-1bd0-41c4-a7c9-f1d4f67bff6c.mov

