# TSNE_Visualizer

Visulaization of the learned representations are extremly important to deep learning for interpretations. Additionally, tracing back to the original inputs from the features will give better understanding of the learned representations. In order to effiently conduct this task and support research, I created a simple interative tool using streamlit and bokeh. 


### Input File format

The input file should be in .h5 format with following data:
  1.'images' : Original Unnormalized images
  2.'labels' : Labels (should be integers)
  3.'tsne'   : Array of the features after applying tsne
  
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


