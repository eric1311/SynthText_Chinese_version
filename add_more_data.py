import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image

def get_data(filename):
    return h5py.File(filename, 'r')

def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path):
  db=h5py.File(DB_FNAME,'w')
  depth_db=get_data(more_depth_path)
  seg_db=get_data(more_seg_path)
  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')
  for imname in os.listdir(more_img_file_path):
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print full_path,imname

      j=Image.open(full_path)
      imgSize=j.size
      rawData=j.tobytes()
      img=Image.frombytes('RGB',imgSize,rawData)
      #img = img.astype('uint16')
      db['image'].create_dataset(imname,data=img)
      db['depth'].create_dataset(imname,data=depth_db[imname])
      db['seg'].create_dataset(imname,data=seg_db['mask'][imname])
      db['seg'][imname].attrs['area']=seg_db['mask'][imname].attrs['area']
      db['seg'][imname].attrs['label']=seg_db['mask'][imname].attrs['label']
  db.close()
  depth_db.close()
  seg_db.close()


# path to the data-file, containing image, depth and segmentation:
DB_FNAME = '/home/momo/liuchengyu/deep_learning/ocr/data/dset_8000.h5'

#add more data into the dset
more_depth_path='/home/momo/liuchengyu/deep_learning/ocr/data/tmp_depth.h5'
more_seg_path='/home/momo/liuchengyu/deep_learning/ocr/data/tmp_seg.h5'
more_img_file_path='/home/momo/liuchengyu/deep_learning/ocr/data/Train/tmp/'

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path)
