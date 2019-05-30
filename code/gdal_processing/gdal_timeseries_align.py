from tqdm import tqdm
import numpy
import gdal
import os
import sys
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# inicialize data location
DATA_FOLDER = "../sensing_data/raw/timeseries/"

ROI = "vila-de-rei/"

REFS = "../sensing_data/clipped/" + ROI + "tsalg"
SRC = DATA_FOLDER + ROI + "2016_align/"

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def main(argv):
    ref_dss = [f for f in os.listdir(REFS) if ".zip" not in f]
    ref_dss = sorted(ref_dss, key=natural_key)

    src_dss = [SRC + f for f in os.listdir(SRC) if ".zip" not in f]
    src_dss.sort()

    for idx, f in enumerate(tqdm(ref_dss)):

        src_idx = int(f.split("_")[0])
        delta_x = int(f.split("_")[1])
        delta_y = int(f.split("_")[2].split(".")[0])

        for image in [f for f in os.listdir(src_dss[idx]) if ".tmp" in f]:
            to_rm = src_dss[idx] + "/" + image
            os.remove(to_rm[:-4])
            os.rename(to_rm, to_rm[:-4])



if __name__ == "__main__":
    main(sys.argv)
