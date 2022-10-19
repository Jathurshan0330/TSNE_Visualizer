import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10,Inferno10,Paired3,Category10_10
import h5py
from io import BytesIO
import io
from PIL import Image
import base64
from sklearn.manifold import TSNE

#streamlit run "/Users/jathurshanpradeepkumar/Documents/Harvard Research/Harvard Research/Nuclei Detection and Clustering/Cell_Analysis/TSNE_visualizer/visualizer.py" --server.maxUploadSize 1000


def main():
    st.header("TSNE Visualizer")
    uploaded_file = st.file_uploader("Choose a file")

    
    # file_path = '/Users/jathurshanpradeepkumar/Documents/Harvard Research/Harvard Research/Nuclei Detection and Clustering/Cell_Analysis/TSNE_visualizer/ki67_tsne.h5'
    if uploaded_file is not None:
        images, labels, tsne = load_data(uploaded_file)

        inten_scale = st.slider("Choose Scaling (intensity): ", min_value=0,   
                        max_value=255, value=30, step=1)
        norm_scale = st.slider("Choose Scaling (normalization): ", min_value=0,   
                        max_value=int(np.max(images)), value=1250, step=1)
        num_classes = st.slider("Choose No of Classes: ", min_value=0,   
                        max_value=10, value=2, step=1)
        n_components = st.slider("Choose No of Components: ", min_value=1,   
                        max_value=3, value=2, step=1)

        if st.button('Display T-SNE'):
            plot_tsne(images,tsne,labels,n_components = n_components, inten_scale = inten_scale,num_classes = num_classes,norm_scale = norm_scale)
        # inten_scale = 30
        #norm_scale = 1250
        #num_classes =2
        #n_components = 2
    
@st.cache
def load_data(file_path):
    hf  = h5py.File(file_path, 'r')
    images = np.array(hf.get('images'))
    labels = np.array(hf.get('labels'))
    tsne = np.array(hf.get('tsne'))
    return images, labels, tsne

def embeddable_image(data):
    img_data = data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L')#.resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def plot_tsne(images,tsne,labels,n_components = 2, inten_scale = 30,num_classes = 2,norm_scale = 1250):
    if n_components ==2:
        digits_df = pd.DataFrame(tsne, columns=('x', 'y'))
    elif n_components ==3: 
        digits_df = pd.DataFrame(tsne, columns=('x', 'y','z'))
    else: 
        raise Exception("Error: n_components should be 2 or 3")
        
    digits_df['label'] = [str(x) for x in labels]
    # print(digits_df['label'].sum())
    print(f"Image Data Shape:{images.shape}")
    print("Plotting T-SNE projection of features ======>")
    digits_df['image'] = list(map(embeddable_image, (images/norm_scale)*inten_scale))
    digits_df['min'] = [x.min() for x in images]
    digits_df['max'] = [x.max() for x in images]
    datasource = ColumnDataSource(digits_df)
    
    color_mapping = CategoricalColorMapper(factors=[str(num_classes-1 - x) for x in [j for j in range (num_classes)]],
                                           palette=Category10_10)#Spectral10)


    plot_figure = figure(
        title='TSNE projection',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
     height="100" />
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Label:</span>
            <span style='font-size: 12px'>@label</span>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Min:</span>
            <span style='font-size: 12px'>@min</span>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Max:</span>
            <span style='font-size: 12px'>@max</span>
        </div>
    </div>
    """))
    if n_components ==2:
        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='label', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4,
            legend_field='label'
        )
        plot_figure.legend.location = "top_left"
        plot_figure.legend.click_policy="hide"
        output_file("interactive_legend.html", title="interactive_legend.py example")
        # show(plot_figure)
        # show(plot_figure)
    elif n_components ==3:
        plot_figure.circle(
            'x',
            'y','z',
            source=datasource,
            color=dict(field='label', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
        )
        # show(plot_figure)
    st.write(plot_figure)

if __name__ == "__main__":
    main()