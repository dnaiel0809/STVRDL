import h5py
import os
from config import get_cfg_defaults 
import torchvision.models as models
import cv2 
import glob


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    print(attrs)
    return attrs


if __name__ == "__main__":
    # name = []
    # bbox = []
    filenames = glob.glob("train/*.png")

    hdf5_data = h5py.File("digitStruct.mat",'r')
    
    for index in range(len(filenames)):
        name = get_name(index, hdf5_data)
        attrs = {}
        item = hdf5_data['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = hdf5_data[item][key]
            values = [hdf5_data[attr.value[i].item()].value[0][0]
                    for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[key] = values
     
        imgData = cv2.imread('train/' + str(index + 1) + '.png')
        bboxNum=len(attr)

        height, width, channels = imgData.shape
        imgDataHeight=height
        imgDataWidth=width
        for j in range(bboxNum):
            xCenter = attrs['left'][j] + attrs['width'][j] / 2		
            yCenter = attrs['top'][j] + attrs['height'][j] / 2
            xCenter = xCenter / imgDataWidth		
            yCenter = yCenter / imgDataHeight		
            imgWidth = attrs['width'][j] / imgDataWidth		
            imgHeight = attrs['height'][j] / imgDataHeight	
                
            file = open("label/{}.txt".format(str(index+1)), 'a')
            file.writelines(str((attrs['label'][j]-1)) + " " + str(xCenter) + " " + str(yCenter) + " " + str(imgWidth) + " " + str(imgHeight) + '\n')
            file.close()

    # cfg = get_cfg_defaults()
    # cfg.merge_from_file("Untitled-1.yml")
    # cfg.freeze()
    # print(cfg)
    # model = models.Darknet(opt.model_def).to(device)
    # model.apply(weights_init_normal)




# print(hdf5_data)
