import os

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


val_path =r"/mnt/Gpan/Mydata/pytorchPorject/datasets/ccpd/val_detect/danger_data_val/"

file_list = []
allFilePath(val_path,file_list)

for file_ in file_list:
    new_file = file_.replace(" ","")
    print(file_,new_file)
    os.rename(file_,new_file)