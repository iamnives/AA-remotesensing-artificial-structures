# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:58:46 2019

@author: Igor
"""

import os
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import multiprocessing
from datetime import datetime, timedelta
import time
#import sys
import glob

def foo(api, products):
    api.download_all(products)

if __name__ == '__main__':
    #orig_stdout = sys.stdout
    #f = open('C:\\Users\\Igor\\.spyder-py3\\script_download_log.txt', 'w')
    #sys.stdout = f
    start = datetime(2016, 1, 1)
    end = datetime(2016, 12, 31)
    it = end + timedelta(days=1)
    os.chdir('D:\\AA-remotesensing-artificial-structures\\sensing_data\\raw\\timeseries\\lisboa-setubal\\s2')
    while it.date() != start.date():
        it -= timedelta(days=1)
        completedir = glob.glob('*' + it.date().strftime("%Y%m%d") + '*')
        completes = [x for x in completedir if x not in glob.glob('*' + it.date().strftime("%Y%m%d") + '*.incomplete')]
        if(len(completes) > 0):
            print("Dia: " + str(it.date()) + " já obtido previamente. Skipping.")
            continue
        successful = False
        while not successful:
            for tries in range(0, 5, 1):
                try:
                    print("Dia: " + str(it.date()))
                    api = SentinelAPI('amneves', 'Amnandre12')
                    footprint = geojson_to_wkt(read_geojson('geo.geojson'))
                    products = api.query(footprint,
                                         date=(it.date().strftime("%Y%m%d"), (it + timedelta(days=1)).date().strftime("%Y%m%d")),
                                        platformname='Sentinel-2',
                                        producttype='S2MSI1C',
                                        area_relation='Contains',
                                        cloudcoverpercentage=(0, 30))
                    dataframe = api.to_dataframe(products)
                    count = dataframe.shape[0]
                    print(str(count) + " produto(s) neste dia.")
                    #api.download_all(products)
                    #download(api, products)
                    if count == 1:
                        nome = dataframe.get_values()[0][0]
                        p = multiprocessing.Process(target=foo, name="Foo", args=(api,products))
                        p.start()
                        print("A aguardar download de " + str(nome) + ".zip")
                        p.join(60 * 60)#60*60 = 1h
                        if p.is_alive():
                            print("Demorou tempo demais. Voltar a tentar...")
                            # Terminate foo
                            p.terminate()
                            p.join()
                            time.sleep(10)
                            break
                        print("Terminado.")
                        #check se ficou mesmo (fileexists)
                        time.sleep(5)
                        if not os.path.isfile(nome + ".zip"):
                            print("Download de " + nome + " falhou.")
                            if tries == 4:
                                successful = True
                            else:
                                print("A retentar.")
                            continue
                    successful = True
                    break
                except Exception as e:
                    time.sleep(5)
                    if(str(e).startswith("HTTPSConnectionPool(host='scihub.copernicus.eu', port=443): Max retries exceeded with url")):
                        print("Connection error. retrying forever.")
                        break
                    else:
                        print("Unknown error:")
                        print(e)
                        if tries == 4:
                            successful = True
                        continue
                else:
                    print("Something went wrong...")
                    successful = True
                    break
    #sys.stdout = orig_stdout
    #f.close()