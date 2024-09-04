import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib.gridspec as gridspec

fig, ax = plt.subplots()
ax.plot([1,3,2])

fig.subplots_adjust(bottom=0.4)
gs = gridspec.GridSpec(2,2)
gs.update(left=0.4, right=0.7, bottom=0.15, top=0.25, hspace=0.1)

axes = [fig.add_subplot(gs[i,j]) for i,j in [[0,0],[0,1],[1,0],[1,1]]]
# create the textboxes
xlim = ax.get_xlim()
ylim = ax.get_ylim()
tb_xmin = TextBox(axes[0],'x', initial = str(xlim[0]), hovercolor='0.975', label_pad=0.1)
tb_xmax = TextBox(axes[1],'',  initial = str(xlim[1]), hovercolor='0.975')
tb_ymin = TextBox(axes[2],'y', initial = str(ylim[0]), hovercolor='0.975', label_pad=0.1)
tb_ymax = TextBox(axes[3],'',  initial = str(ylim[1]), hovercolor='0.975')

def submit(val):
    lim = [float(tb.text) for tb in [tb_xmin,tb_xmax,tb_ymin,tb_ymax]]
    ax.axis(lim)
    fig.canvas.draw_idle()

for tb in [tb_xmin,tb_xmax,tb_ymin,tb_ymax]:
    tb.on_submit(submit)
plt.show()



#%% my test
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 

import h5py

filename = "Morphology_CNN.h5"

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        print(type(f))
        f.visit(keys.append) # append all keys to list
        
        print("keys : ", keys)
        
        for key in keys:  # key로 불러오는 객체 각각은 group
            print("now the key is : ", key)
            if ':' in key: # contains data if ':' in key
                weights[f[key].name] = f[key].values
    return weights

read_hdf5(filename)