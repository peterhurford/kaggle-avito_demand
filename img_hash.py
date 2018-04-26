import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Using the code from http://blog.iconfinder.com/detecting-duplicate-images-using-python/

import zipfile
import os
import io
from PIL import Image
import datetime

def dhash(image,hash_size = 32):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    pixels = list(image.getdata())
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    return ''.join(hex_string)
    

for zip_counter in [0,1,2,3,4,5,6,7,8,9]:
    imgzipfile = zipfile.ZipFile('Images_'+str(zip_counter)+'.zip')
    print ('Doing zip file ' + str(zip_counter))
    namelist = imgzipfile.namelist()
    print ('Total elements ' + str(len(namelist)))

    img_id_hash = []
    counter = 1
    for name in namelist:
        print name
        try:
            imgdata = imgzipfile.read(name)
            if len(imgdata) >0:
                img_id = name[:-4]
                img_id = img_id.split(os.sep)
                img_id = img_id[2]
                print img_id
                stream = io.BytesIO(imgdata)
                img = Image.open(stream)
                img_hash = dhash(img)
                img_id_hash.append([img_id,img_hash])
                counter+=1
            if counter%10000==0:
                print 'Done ' + str(counter) , datetime.datetime.now()
        except:
            print ('Could not read ' + str(name) + ' in zip file ' + str(zip_counter))
    df = pd.DataFrame(img_id_hash,columns=['image_id','image_hash'])
    df.to_csv('image_hash_' + str(zip_counter) + '_v2.csv')
