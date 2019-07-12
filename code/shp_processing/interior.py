# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:22:47 2019

@author: Ricardo
"""
import os
import traceback
import gdal
from gdal import ogr, osr
import numpy as np
import sys
import gc
import importlib
import zipfile
from timeit import default_timer as timer
from concavehull import ConcaveHull
import pandas as pd
import numba
import traceback
import multiprocessing
from multiprocessing import Process,Pool, Queue
import subprocess 
from rasterio.mask import mask


#cria na layer dest os fields da layer src
def createFieldsFrom(src, dest):
    lyrDefn = src.GetLayerDefn()

    for i in range( lyrDefn.GetFieldCount() ):
        dest.CreateField(lyrDefn.GetFieldDefn(i))
#        fieldName =  lyrDefn.GetFieldDefn(i).GetName()
    
    return dest 

#Obtem os pontos que fazem parte das fronteiras das geometrias. (Apenas funciona com os ficheiros orginais das faixas, ou seja, sem merge ou union..)
def get_points_from_geomety(file):
    print("Getting geometry points")    
    fgc_shp = ogr.Open(file)
    lyr = fgc_shp.GetLayer()
    
    points_by_id = pd.DataFrame(data=[], columns=["id","points"])
    
    for feature in  lyr:
#        print("entrou")
        i_desc_fgc = feature.GetFieldIndex("ID_SEQ")
        fgc_id = feature.GetFieldAsInteger(i_desc_fgc)
        geom = feature.GetGeometryRef() 
           

        for j in range(0,geom.GetGeometryCount() ):
            ring = geom.GetGeometryRef(j)
            if ring != None:
                #Se a faixa ainda nao foi vista adiciona ao dataframe
                if(points_by_id[points_by_id.iloc[:,0]==fgc_id].shape[0] == 0):
                    points_by_id = points_by_id.append({'id' : fgc_id , 'points' :[] } , ignore_index=True)
                
                #Adicionar os pontos a lista de pontos da respetiva faixa
#                temp_points = points_by_id[points_by_id.iloc[:,0]==fgc_id].iloc[0,1]
#                print("  ",j," has", ring.GetPointCount(), " points" )
                for i in range(0, ring.GetPointCount()):
                    lon, lat, z = ring.GetPoint(i)
                    points_by_id[points_by_id.iloc[:,0]==fgc_id].iloc[0,1].append((lon,lat))
#                    temp_points.append((lon,lat))
#            else:
#                print("  ",j," is None" )
#                points_by_id[points_by_id.iloc[:,0]==fgc_id].iloc[0,1] =temp_points
       
    fgc_shp = None
    lyr = None    
    
    return points_by_id



def difference(f1, f2, out):
    try: 
        print("Difference between:")
        print("   ",f1)
        print("   ",f2)
        fgc_shp = ogr.Open(f1)
        lyr1 = fgc_shp.GetLayer() 
        
        fgc_shp2 = ogr.Open(f2)
        lyr2 = fgc_shp2.GetLayer()
    
        driver=ogr.GetDriverByName('ESRI Shapefile')
        ds=driver.CreateDataSource(out)
    
        diff_lyr = ds.CreateLayer('temp', lyr2.GetSpatialRef(), ogr.wkbMultiPolygon )
    #   diff_lyr = createFieldsFrom(lyr2, diff_lyr)
    
        
        union1 = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat1 in lyr1:
            geom1 = feat1.GetGeometryRef()
            if geom1 != None: 
                union1 = union1.Union(geom1)
                 
            geom1 = None
            
        union2 = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat2 in lyr2:
            geom2 = feat2.GetGeometryRef()
            if geom2 != None:        
                union2 = union2.Union(geom2) 
            geom2 = None
          
                
        union1 = union1.Buffer(0)
        union2 = union2.Buffer(0)
        
        print(union1.GetGeometryCount())
        
        diff = union1.Difference(union2)
        
        new_feat = ogr.Feature(diff_lyr.GetLayerDefn())
        new_feat.SetGeometry(diff)
        diff_lyr.CreateFeature(new_feat)   
    

#        union1 = ogr.Geometry(ogr.wkbMultiPolygon)
#        for feat1 in lyr1:
#            geom1 = feat1.GetGeometryRef()
#            fgc_shp2 = ogr.Open(f2)
#            lyr2 = fgc_shp2.GetLayer()
#            for feat2 in lyr2:   
#                geom2 = feat2.GetGeometryRef()
#                if geom1 != None  and geom2 != None:
##                    if geom2.Intersects(geom1) or geom2.Within(geom1) or geom2.Overlaps(geom1) or geom2.Crosses(geom1):
#                        diff = geom1.SymmetricDifference(geom2)
#                        if diff != None:
#                            union1.AddGeometry(diff)
##                            new_feat = ogr.Feature(diff_lyr.GetLayerDefn())
##                            new_feat.SetGeometry(diff)
##                            diff_lyr.CreateFeature(new_feat)
#                        
#                        geom2 = None
#                        new_feat = None
#                        diff=None
#            
#            geom1 = None
        
#        new_feat = ogr.Feature(diff_lyr.GetLayerDefn())
#        new_feat.SetGeometry(union1)
#        diff_lyr.CreateFeature(new_feat)
    
    except:
        print("exception thrown!")
        traceback.print_exc()
        

    new_feat= None
    union1 = None   
    union2 = None            
    diff_lyr = None
    fgc_shp = None
    fgc_shp2 = None
    ds = None
    lyr1 = None
    lyr2 = None


def remove_biggest_polygon(f1, area, out):
    print("Removing biggest polygon")
    
    fgc_shp = ogr.Open(f1)
    lyr = fgc_shp.GetLayer()
    
    driver=ogr.GetDriverByName('ESRI Shapefile')
    ds=driver.CreateDataSource(out)
    
    holes_lyr = ds.CreateLayer('temp', lyr.GetSpatialRef(), ogr.wkbMultiPolygon )        
    
    
    max_area = 0.0
    max_index = 0
    for feature in lyr:
        geom = feature.GetGeometryRef() 
        print(geom.GetGeometryCount())
        for j in range(0,geom.GetGeometryCount() ):
            ring = geom.GetGeometryRef(j)
            area = ring.GetArea()
            print(area)
            if area> max_area:
                max_index = j
                max_area = area
                
              
    lyr =None
    fgc_shp = None 
#    print("max= ", max_index)
    
    fgc_shp2 = ogr.Open(f1)
    lyr2 = fgc_shp2.GetLayer()
    
    for feature in lyr2:
        geom = feature.GetGeometryRef() 
        
        for j in range(0,geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(j)
            if j != max_index:
                new_feat = ogr.Feature(holes_lyr.GetLayerDefn())
                new_feat.SetGeometry(ring.Buffer(0))
                holes_lyr.CreateFeature(new_feat)
                new_feat = None
#                print(ring.GetArea())
        
    holes_lyr = None
    ds=None
    lyr =None
    fgc_shp = None

#cria um polygno do tamanho da extensão do ficheiro
def extent_polygon(file,out):
    fgc_shp = ogr.Open(file)
    lyr = fgc_shp.GetLayer()
    
    extent  = lyr.GetExtent()
    
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(extent[0], extent[2])
    ring.AddPoint(extent[1], extent[2])
    ring.AddPoint(extent[1], extent[3])
    ring.AddPoint(extent[0], extent[3])
    ring.AddPoint(extent[0], extent[2])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    driver=ogr.GetDriverByName('ESRI Shapefile')
    ds=driver.CreateDataSource(out)    
    extent_lyr = ds.CreateLayer('temp', lyr.GetSpatialRef(), ogr.wkbMultiPolygon )      
    new_feat = ogr.Feature(extent_lyr.GetLayerDefn())
    new_feat.SetGeometry(poly)
    extent_lyr.CreateFeature(new_feat)
    
    extent_lyr = None
    ds = None
    driver = None
    new_feat = None
    lyr = None
    fgc_shp = None


def merge_geometries_by_field(file,out, field):
    
    file_name = os.path.basename(file)
    
    if field == None :
        query = "SELECT ST_Union(geometry) AS geometry FROM '" + file_name[:-4]+"'"
    else:
        query = "SELECT ST_Union(geometry) AS geometry, "+field+" FROM '" + file_name[:-4]+"' GROUP BY "+field
         
    
    print(file)
    ds = gdal.VectorTranslate(out,file,
                         SQLDialect="sqlite",
                         SQLStatement=query,
                         format="ESRI Shapefile",
                         geometryType="PROMOTE_TO_MULTI")
    ds = None

def merge_files(f1, f2, out):
    print("Merging files...")
    try:        
        fgc_shp = ogr.Open(f1)
        lyr1 = fgc_shp.GetLayer() 
        
        fgc_shp2 = ogr.Open(f2)
        lyr2 = fgc_shp2.GetLayer()
    
        driver=ogr.GetDriverByName('ESRI Shapefile')
        ds=driver.CreateDataSource(out)
    
        merge_lyr = ds.CreateLayer('temp', lyr2.GetSpatialRef(), ogr.wkbMultiPolygon )
    #   diff_lyr = createFieldsFrom(lyr2, diff_lyr)
    
        print("    First file uninon..")
        union1 = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat1 in lyr1:
            geom1 = feat1.GetGeometryRef()
            if geom1 != None: 
                union1 = union1.Union(geom1)
                 
            geom1 = None
        print("    Second file uninon..")    
        union2 = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat2 in lyr2:
            geom2 = feat2.GetGeometryRef()
            if geom2 != None:        
                union2 = union2.Union(geom2) 
            geom2 = None
          
                
        union1 = union1.Buffer(0)
        union2 = union2.Buffer(0)
        
        print("    Final uninon..")
        merge = union1.Union(union2)
        
        new_feat = ogr.Feature(merge_lyr.GetLayerDefn())
        new_feat.SetGeometry(merge)
        merge_lyr.CreateFeature(new_feat)   
    
    except:
        print("exception thrown!")
        traceback.print_exc()
        

    new_feat= None
    union1 = None   
    union2 = None            
    diff_lyr = None
    fgc_shp = None
    fgc_shp2 = None
    ds = None
    lyr1 = None
    lyr2 = None


def fill_holes(file, out):
    print("Filling holes...")
    dir_path = os.path.dirname(os.path.realpath(file))
    file_name = os.path.basename(file)

    extent_polygon(file, dir_path +"/temp_extent_"+file_name)

    difference(dir_path +"/temp_extent_"+file_name, file, dir_path +"/temp_diff_extent_"+file_name )
    
    remove_biggest_polygon(dir_path +"/temp_diff_extent_"+file_name, 1,dir_path +"/temp_holes_"+file_name)

    merge_files(file, dir_path +"/temp_holes_"+file_name, out)


#cria um geometria concacva a partir de um cunjunto de polygnos    
def test_concav(file,out):
    
    points_by_id = get_points_from_geomety(file)
    
    
    fgc_shp = ogr.Open(file)
    lyr = fgc_shp.GetLayer()
    
    driver=ogr.GetDriverByName('ESRI Shapefile')
    
    ds1=driver.CreateDataSource(out)
    concave_lyr=ds1.CreateLayer('temp', lyr.GetSpatialRef(), ogr.wkbPolygon)
    concave_lyr = createFieldsFrom(lyr,concave_lyr)
    print("starting concave")


    ps = np.array([ ( points_by_id.iloc[xi,0] , np.array(points_by_id.iloc[xi,1]) ) for xi in range(0,points_by_id.shape[0]) ], dtype=object)

    #Processar em várias threads
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    res = pool.map( temp,ps ) 
   
    print("End of concave calculations.")
    
    
    print("Saving to file")
    res = np.asarray(res,dtype=object)


    for i in range(0, res.shape[0]):
        try:
            fgc_id = res[i][0]
            hull = res[i][1]
            
            new_ring = ogr.Geometry(ogr.wkbLinearRing)
            hull = np.asarray(hull)

            for i in range(0, hull.shape[0]):
                new_ring.AddPoint(hull[i,0], hull[i,1])
                        
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(new_ring)
            new_feat = ogr.Feature(concave_lyr.GetLayerDefn())
            new_feat.SetField("ID_SEQ",fgc_id)
            new_feat.SetGeometry(poly)
            concave_lyr.CreateFeature(new_feat)                   
            new_feat = None
            
        except:
            print("exception thrown!")
            traceback.print_exc()
            break


def temp (fgc):
    points = []
    fgc_id = fgc[0]
    all_points = fgc[1] 
    
    if(all_points.shape[0]>2):
        points = ConcaveHull.concaveHull(all_points, 3)
        
    return (fgc_id,points)


def addBuffer(file,size, new_file):

    fgc_shp = ogr.Open(file)
    layer = fgc_shp.GetLayer()
    
    driver=ogr.GetDriverByName('ESRI Shapefile')
    ds=driver.CreateDataSource(new_file)

    out_lyr=ds.CreateLayer('temp', layer.GetSpatialRef(), ogr.wkbPolygon)
    
    out_lyr = createFieldsFrom(layer, out_lyr)
    
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        
        if geom != None:
            new_feat = ogr.Feature(out_lyr.GetLayerDefn())
            new_feat.SetFrom(feature)
            new_feat.SetGeometry(geom.Buffer(size))
            
            
            out_lyr.CreateFeature(new_feat)
        # else:
            #out_lyr.CreateFeature(feature)
                        
            new_feat = None
            geom = None
                    
    out_lyr= None
    ds=None
    driver = None
    layer = None  
    fgc_shp  = None  


def interior(subdir, file):
    print("Merging geometries...")
    file_path = subdir+"/"+file
             
    if file_path.endswith(".shp") :
        new_file = subdir+'/merged.shp'
        temp_file = subdir+'/temp_'+file
        
        #Junta geometrias
        merge_geometries_by_field(file_path, temp_file, None)
        if file.startswith("clipped"):
            #calcular a concav hull
            print("A calcular o interior das faixas ao redor das habitações...")
            test_concav(file_path, subdir+"/temp_1_concave_"+file)
            test_concav( subdir+"/temp_1_concave_"+file, subdir+"/temp_2_concave_"+file)
            merge_files(subdir+"/temp_2_concave_"+file, file_path, subdir+"/temp_interior_"+file )
            fill_holes(subdir+"/temp_interior_"+file, subdir+"/interior_habi.shp")
            
        
        #Buffer para remover falhas entre poligonos
        addBuffer(temp_file, 0.0001, new_file)
        os.remove(temp_file)
        
                
if __name__ == '__main__':                              
    interior(".", "clipped_fgc_multipol.shp")                
    difference("./interior_habi.shp", "./merged.shp", "./final_result.shp")                
                
                