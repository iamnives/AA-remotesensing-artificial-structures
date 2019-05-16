import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal
import numpy as np

from tqdm import tqdm

#inicialize data location
ROI = "vila-de-rei/"
SRC_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"
DST_FOLDER =  "../sensing_data/" + "clipped/" + ROI + "ts/"
def main(argv):
	src_dss = [f for f in os.listdir(SRC_FOLDER) if ".jp2" in f]
	src_dss.sort()
	sets = [f.split("clipped")[0] for f in src_dss]
	sets = np.unique(sets)

	for f_id in enumerate(tqdm(sets)):
		print(f_id)

if __name__== "__main__":
  main(sys.argv)