import os
import sys
from tqdm import tqdm

MAIN_DIR = "C:/Users/amnev/Desktop/tese/Chapters/Figures/res/"

def main():
    src_dss = [f for f in os.listdir(MAIN_DIR) if 'pdf' in f]
    src_dss.sort()
    
    for f in tqdm(src_dss):
        compress = f"gswin64c -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile={MAIN_DIR}compressed/{f} {MAIN_DIR}/{f}"
        os.system(compress)

if __name__ == "__main__":
    main()
