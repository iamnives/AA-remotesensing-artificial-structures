import os
import sys

from tkinter import *
import numpy as np
 
from utils import data

import tuning_models as tm

master = Tk()
cbuts = []

algorithms = {
   "RF": tm.rftree,
   "SVM": tm.svm,
   "GTB": tm.gradtree,
}

def select_all():
    for j in cbuts:
        j.select()

def var_states():
   ds = np.array([x.get() for x in list(values.values())])
   features = datasets[ds==1]
   master.destroy()
   algorithms[v.get()](["ui_default" , features])
   
   

v = StringVar()
v.set("RF") # initialize
Radiobutton(master, text="Random Forest", variable=v, value="RF").pack(anchor=W)
Radiobutton(master, text="SVM", variable=v, value="SVM").pack(anchor=W)
Radiobutton(master, text="Boosting Trees", variable=v, value="GTB").pack(anchor=W)


Label(master, text="Features to use:").pack(anchor=W)
values = {}
datasets = data.get_features()
for idx, raster in enumerate(datasets):
   values[raster] = IntVar()
   cbuts.append(Checkbutton(master, text=raster, variable=values[raster]))
   cbuts[idx].pack(anchor=W)
Button(master, text='Train', command=var_states).pack(fill='both')
Button(master, text='All', command=select_all).pack(fill='both')

mainloop()