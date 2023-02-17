from detect_rec_plate_macao import detect_Recognition_plate,attempt_load,init_model,allFilePath,cv_imread,draw_result,four_point_transform,order_points
import os
import argparse
import torch
import cv2
import time
import shutil
import numpy as np
import json
import re
pattern_str = "([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]" \
              "{1}(([A-HJ-Z]{1}[A-HJ-NP-Z0-9]{5})|([A-HJ-Z]{1}(([DF]{1}[A-HJ-NP-Z0-9]{1}[0-9]{4})|([0-9]{5}[DF]" \
              "{1})))|([A-HJ-Z]{1}[A-D0-9]{1}[0-9]{3}警)))|([0-9]{6}使)|((([沪粤川云桂鄂陕蒙藏黑辽渝]{1}A)|鲁B|闽D|蒙E|蒙H)" \
              "[0-9]{4}领)|(WJ[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼·•]{1}[0-9]{4}[TDSHBXJ0-9]{1})" \
              "|([VKHBSLJNGCE]{1}[A-DJ-PR-TVY]{1}[0-9]{5})"


def is_car_number(pattern, string):
    if re.findall(pattern, string):
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'runs/train/yolov711/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'/mnt/Gpan/Mydata/pytorchPorject/datasets/macao_plate/download/', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default=r'test2', help='source')
    # parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default=r'/mnt/Gpan/Mydata/pytorchPorject/datasets/macao_plate/result/', help='source') 
    parser.add_argument('--kpt-label', type=int, default=4, help='number of keypoints')
    device  =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    opt = parser.parse_args()
    print(opt)
    model = attempt_load(opt.detect_model, map_location=device)
    # torch.save()
    plate_rec_model=init_model(device,opt.rec_model) 
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    file_list=[]
    index_1=0
    index_small=0
    # error_path =r"E:\study\plate\data\@shaixuan\train\pic"
    allFilePath(opt.source,file_list)
    time_b = time.time()
   
    for pic_ in file_list:
        try:
            image_name = os.path.basename(pic_)
            # ori_plate = image_name.split("_")[0]
            # ori_plate = image_name.split(".")[0]
            flag = 1
            index_1+=1
            label_dict_str=""
            lable_dict={}
            # label_dict={}                                 
            print(index_small,index_1,pic_)
            img = cv_imread(pic_)
            if img is None:
                continue
            if img.shape[-1]==4:
                img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            # img = my_letter_box(img)
            count=0
            dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,opt.img_size)
            for result_ in dict_list:
                if not result_:
                    continue
                index_small+=1
                landmarks=result_['landmarks']
                landmarks_np = np.array(landmarks).reshape(-1,2)
                img_roi = four_point_transform(img,landmarks_np)
                plate_no= result_['plate_no']
                # if len(plate_no)<6 or not is_car_number(pattern_str,plate_no):
                if len(plate_no)<6:
                    continue
                height = result_['roi_height']
                # if height<48:
                #     continue
                pic_name =plate_no+"_"+str(index_small)+".jpg"
                label = result_["label"]
                

                roi_save_folder = os.path.join(opt.output,str(label))
                if not os.path.exists(roi_save_folder):
                    os.mkdir(roi_save_folder)
                roi_path = os.path.join(roi_save_folder,pic_name)
                cv2.imencode('.jpg',img_roi)[1].tofile(roi_path)
        except:
            print("error!")
  
                