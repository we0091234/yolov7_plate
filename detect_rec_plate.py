import argparse
import time
import os
import copy
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import  non_max_suppression, scale_coords
from plate_recognition.plate_rec import get_plate_result,allFilePath,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from utils.datasets import letterbox
from utils.cv_puttext import cv2ImgAddText
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

def order_points(pts):  #关键点按照（左上，右上，右下，左下）排列
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  #透视变换
    # rect = order_points(pts)
    rect=pts.astype("float32")
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num,device,plate_rec_model):
    h,w,c = img.shape
    result_dict={}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height=y2-y1
    landmarks_np=np.zeros((4,2))
    rect=[x1,y1,x2,y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i]=np.array([point_x,point_y])

    class_label= int(class_num)  #车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img,landmarks_np)   #透视变换得到车牌小图
    # cv2.imwrite("roi.jpg",roi_img)
    # roi_img_h = roi_img.shape[0]
    # roi_img_w = roi_img.shape[1]
    # if roi_img_w/roi_img_h<3:
    #     class_label=
    # h_w_r = roi_img_w/roi_img_h
    if class_label :        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img=get_split_merge(roi_img)
    plate_number,rec_prob,plate_color,color_conf = get_plate_result(roi_img,device,plate_rec_model)                 #对车牌小图进行识别
    
    result_dict['rect']=rect
    result_dict['landmarks']=landmarks_np.tolist()
    result_dict['plate_no']=plate_number
    result_dict['rec_conf']=rec_prob   #每个字符的概率
    result_dict['plate_color']=plate_color 
    result_dict['color_conf']=color_conf 
    result_dict['roi_height']=roi_img.shape[0]
    result_dict['score']=conf
    result_dict['label']=class_label
    return result_dict

def detect_Recognition_plate(model, orgimg, device,plate_rec_model,img_size):
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list=[]
    im0 = copy.deepcopy(orgimg)
    imgsz=(img_size,img_size)
    img = letterbox(im0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x640X640
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, kpt_label=4,agnostic=True)
    for i, det in enumerate(pred):
        if len(det):
                # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=4, step=3)
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 6:].view(-1).tolist()
                landmarks = [landmarks[0],landmarks[1],landmarks[3],landmarks[4],landmarks[6],landmarks[7],landmarks[9],landmarks[10]]
                class_num = det[j, 5].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num,device,plate_rec_model)
                dict_list.append(result_dict)
    return dict_list


def draw_result(orgimg,dict_list):
    result_str =""
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=max(0,int(y-padding_h))
        rect_area[2]=min(orgimg.shape[1],int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))
        rect_area = [int(x) for x in rect_area]

        height_area = result['roi_height']
        landmarks=result['landmarks']
        result_p = result['plate_no']+" "+result['plate_color']
        result_str+=result_p+" "
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        labelSize = cv2.getTextSize(result_p,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
            rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
            
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.2*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)
        
        if len(result)>1:
            for i in range(4):  #关键点
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
            orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
            # orgimg=cv2ImgAddText(orgimg,result,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
    print(result_str)
    return orgimg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/yolov7-lite-s.pt', help='model.pt path(s)') #检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')  #车牌识别 +颜色识别
    parser.add_argument('--source', type=str, default='imgs', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result', help='source') 
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
    allFilePath(opt.source,file_list)
    time_b = time.time()
    for pic_ in file_list:
        print(pic_,end=" ")
        img = cv_imread(pic_)
        if img.shape[-1]==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        # img = my_letter_box(img)
        dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,opt.img_size)
        ori_img=draw_result(img,dict_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(opt.output,img_name)
        cv2.imwrite(save_img_path,ori_img)
    print(f"elasted time is {time.time()-time_b} s")