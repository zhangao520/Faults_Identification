import numpy as np
from struct import *
import os

def annotation():
    annotation = np.zeros((360,1036,3500)).astype(np.int8)
    with open('../fault_data/FaultGroup.txt','r') as pos:
        lines=pos.readlines()
        for line in lines[1:]:
            line_split=line.split()
            fault_pos = np.float32(line_split[1:]).astype(np.int32)
            annotation[fault_pos[1],fault_pos[0],fault_pos[2]]=1
    with open('../fault_data/NOT-FaultGroup.txt','r') as neg:
        lines=neg.readlines()
        for line in lines[1:]:
            line_split = line.split()
            neg_pos = np.float32(line_split[1:]).astype(np.int32)
            annotation[neg_pos[1],neg_pos[0],neg_pos[2]]=2
    annotation=annotation.flatten()
    with open ('../fault_data/cube_A1_annotation.bin','wb') as ann:
        for i in annotation:
            i2b=pack('B',i)
            ann.write(i2b)
    print('Writing Done!')

def crop_and_save(dx=36,dy=31,dz=70):
    base = '../fault_data/crop/cube_A1'
    name_list = [
                 '_0',
                 '_dip_crossline',
                 '_dip_curvature_shape_index',
                 '_dip_inline',
                 '_energy_gradient_crossline',
                 '_energy_gradient_inline',
                 '_inst_freq',
                 '_sobel_filter_similarity']

    cx=int(np.floor(216/dx))
    cy=int(np.floor(620/dy))
    cz=int(np.floor(2100/dz))
    for name in name_list:
        bin = base + name + '.bin'
        print(bin)
        index=0
        with open(bin, 'rb') as bin_file:
            data_string = bin_file.read()
            data = unpack(len(data_string)//4*'f', data_string)
            data = np.reshape(data, (216, 620, 2100))

        for i in range(cx):
            for j in range(cy):
                for k in range(cz):
                    postfix="{:0>4d}".format(index)
                    print(postfix)
                    cube=data[dx*i:dx*(i+1),
                               dy*j:dy*(j+1),
                               dz*k:dz*(k+1)]
                    cube=cube.flatten()
                    with open(os.path.join('../fault_data/slice/'+name[1:],
                                          postfix+'.bin'),'wb') as file:
                        for num in cube:
                            num2b = pack('f', num)
                            file.write(num2b)
                    #np.save(os.path.join('./fault_data/slice/'+name,
                                           #postfix+'.npy'),slice)
                    index=index+1

    
def wash():
    base='../fault_data/cube_A1_0_'
    keep_list=[]
    for i in range(360):
        postfix = "{:0>3d}".format(i)
        binfile = base+postfix+'.bin'
        with open(binfile, 'rb') as bin_file:
            data_string = bin_file.read()
            data = unpack(len(data_string) // 4 * 'f', data_string)
            if not np.any(np.isnan(data)):
                keep_list.append(i)
    print (keep_list)
    
def find():
    base = '../fault_data/cube_A1'
    name = '_0'
    name_list = ['_0',
                 '_dip_crossline',
                 '_dip_curvature_shape_index',
                 '_dip_inline',
                 '_energy_gradient_crossline',
                 '_energy_gradient_inline',
                 '_inst_freq',
                 '_sobel_filter_similarity']
    for name in name_list[0:1]:
        bin = base+name+'.bin'
        print (bin)
        with open(bin,'rb') as bin_file:
            data_string = bin_file.read()
            data = unpack(len(data_string) // 4 * 'f', data_string)
            data = np.reshape(data, (360, 1036, 3500))
            dist_min=1
            for i in range(360):
                for j in range(1036):
                    for k in range(3500):
                        if np.isnan(data[i,j,k]):
                            norm = np.max(np.abs(np.asarray([(i-180)/180 , (j-518)/518, (k-1750)/1750])))
                            if norm < dist_min:
                                dist_min = norm
            print(dist_min)

def find2():
    base = './fault_data/cube_A1'
    name_list = ['_0',
                 '_dip_crossline',
                 '_dip_curvature_shape_index',
                 '_dip_inline',
                 '_energy_gradient_crossline',
                 '_energy_gradient_inline',
                 '_inst_freq',
                 '_sobel_filter_similarity']
    for name in name_list[1:]:
        print(base+name)
        dist_min = 1
        for i in range(360):
            index = "_{:0>3d}".format(i)
            bin = base+name+index+'.bin'
            #print(bin)
            with open (bin,'rb') as bin_file:
                data_string = bin_file.read()
                data = unpack(len(data_string) // 4 * 'f', data_string)
                data =  np.reshape(data, (1036,3500))
                for j in range(1036):
                    for k in range(3500):
                        if np.isnan(data[j,k]):
                            norm = np.max(np.abs(np.asarray([(i - 180) / 180, (j - 518) / 518, (k - 1750) / 1750])))
                            if norm < dist_min:
                                dist_min = norm
        print(dist_min)

def crop():
    base = './fault_data/cube_A1'
    name_list = ['_0',
                 '_dip_crossline',
                 '_dip_curvature_shape_index',
                 '_dip_inline',
                 '_energy_gradient_crossline',
                 '_energy_gradient_inline',
                 '_inst_freq',
                 '_sobel_filter_similarity']
    mean=[]
    var=[]
    for name in name_list:
        print (base + name)
        cube = np.zeros((0,620,2100))
        for i in range(72,288):
            index = "_{:0>3d}".format(i)
            bin = base + name + index + '.bin'
            #print (bin)
            with open(bin,'rb') as bin_file:
                data_string = bin_file.read()
                data = unpack(len(data_string) // 4 * 'f', data_string)
                data =  np.reshape(data, (1036,3500))
                data = data[208:828,700:2800]
                cube = np.concatenate((cube,[data]))
        cube = cube.flatten()
        m = np.mean(cube)
        v = np.var(cube)
        mean.append(m)
        var.append(v)
        print (m,v)
        with open(os.path.join('./fault_data/crop','cube_A1'+name+'.bin'),'wb') as crop_bin:
            for num in cube:
                num2b = pack('f', num)
                crop_bin.write(num2b)
    np.save('mean.npy',np.array(mean))
    np.save('var.npy',np.array(var))

def cut_annotaiton_save_slice(dx=36,dy=31,dz=70):
    if not os.path.exists("../fault_data/npdata/annotation/"):
        os.mkdir("../fault_data/npdata/annotation/")
    bin = '../fault_data/cube_A1_annotation.bin'
    cx=int(np.floor(216/dx))
    cy=int(np.floor(620/dy))
    cz=int(np.floor(2100/dz))
    with open(bin, 'rb') as bin_file:
        data_string = bin_file.read()
        data = unpack(len(data_string)  * 'b', data_string)
        data = np.reshape(data, (360, 1036, 3500))
        data = data[72:288,208:828,700:2800]
        index = 0
        for i in range(cx):
            for j in range(cy):
                for k in range(cz):
                    postfix="{:0>4d}".format(index)
                    print(postfix)
                    cube=data[dx*i:dx*(i+1),
                               dy*j:dy*(j+1),
                               dz*k:dz*(k+1)]
                    #cube=cube.flatten()
                    # with open(os.path.join('./fault_data/slice/annotation',
                    #                       postfix+'.bin'),'wb') as file:
                    #     for num in cube:
                    #         num2b = pack('b', num)
                    #         file.write(num2b)
                    np.save(os.path.join('../fault_data/npdata/annotation/',postfix+'.npy'),cube)
                    index=index+1 
def cut_cropped_data_save_slice(dx=36,dy=31,dz=70):
    base = '../fault_data/crop/cube_A1'
    name_list = [
                 '_0',
                 '_dip_crossline',
                 '_dip_curvature_shape_index',
                 '_dip_inline',
                 '_energy_gradient_crossline',
                 '_energy_gradient_inline',
                 '_inst_freq',
                 '_sobel_filter_similarity']

    cx=int(np.floor(216/dx))
    cy=int(np.floor(620/dy))
    cz=int(np.floor(2100/dz))
    for name in name_list:
        bin = base + name + '.bin'
        print(bin)
        index=0
        with open(bin, 'rb') as bin_file:
            data_string = bin_file.read()
            data = unpack(len(data_string)//4*'f', data_string)
            data = np.reshape(data, (216, 620, 2100))
        os.mkdir("../fault_data/npdata/"+name[1:])
        for i in range(cx):
            for j in range(cy):
                for k in range(cz):
                    postfix="{:0>4d}".format(index)
                    print(postfix)
                    cube=data[dx*i:dx*(i+1),
                               dy*j:dy*(j+1),
                               dz*k:dz*(k+1)]
                    #cube=cube.flatten()
                    #with open(os.path.join('./fault_data/slice/'+name[1:],
                                          #postfix+'.bin'),'wb') as file:
                        #for num in cube:
                            #num2b = pack('f', num)
                            #file.write(num2b)
                    
                    np.save(os.path.join('../fault_data/npdata/'+name[1:],
                                           postfix+'.npy'),cube)
                    index=index+1           
        




    
if __name__=='__main__':
    #annotation()
    # mean_and_var(name_list)
    #wash()
    #find2()
    #crop()
    #crop_and_save()
    #crop_annotaiton()
    cut_annotaiton_save_slice()
