import os
import sys

from tkinter import *
import numpy as np
 
from utils import data

import tuning_models as tm

master = Tk()
canvas = Canvas(master, borderwidth=0)
frame = Frame(canvas)
vsb = Scrollbar(master, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)


vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((4,4), window=frame, anchor="nw")
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
   algorithms[v.get()](features)
   
def onFrameConfigure(canvas):
   '''Reset the scroll region to encompass the inner frame'''
   canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

v = StringVar()
v.set("RF") # initialize
Radiobutton(frame, text="Random Forest", variable=v, value="RF").pack(anchor=W)
Radiobutton(frame, text="SVM", variable=v, value="SVM").pack(anchor=W)
Radiobutton(frame, text="Boosting Trees", variable=v, value="GTB").pack(anchor=W)


Label(frame, text="Features to use:").pack(anchor=W)
values = {}
datasets = data.get_features()
for idx, raster in enumerate(datasets):
   values[raster] = IntVar()
   cbuts.append(Checkbutton(frame, text=raster, variable=values[raster]))
   cbuts[idx].pack(anchor=W)
Button(frame, text='Train', command=var_states).pack(fill='both')
Button(frame, text='All', command=select_all).pack(fill='both')

master.mainloop()