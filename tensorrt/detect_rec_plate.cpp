#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "include/utils.hpp"
#include "preprocess.h"
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000

struct bbox 
{
     float x1,x2,y1,y2;
     float landmarks[8];
     float score;
};


bool my_func(bbox a,bbox b)
{
    return a.score>b.score;
}
float get_IOU(bbox a,bbox b)
{
     float x1 = std::max(a.x1,b.x1);
     float x2 = std::min(a.x2,b.x2);
     float y1 = std::max(a.y1,b.y1);
     float y2 = std::min(a.y2,b.y2);

     float w = std::max(0.0f,x2-x1);
     float h = std::max(0.0f,y2-y1);

     float inter_area = w*h;
     float union_area = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter_area;

     float IOU = 1.0*inter_area/ union_area;
     return IOU;
}

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
const std::vector<std::string> plate_string={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖", \
"闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","深", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

const std::vector<std::string> plate_string_yinwen={"#","<beijing>","<hu>","<tianjin>","<chongqing>","<hebei>","<jing>","<meng>","<liao>","<jilin>","<hei>","<su>","<zhe>","<wan>", \
"<fujian>","<gan>","<lun>","<henan>","<hubei>","<hunan>","<yue>","<guangxi>","<qiong>","<chuan>","<guizhou>","<yun>","<zang>","<shanxi>","<gan>","<qinghai>",\
"<ning>","<xin>","<xue>","<police>","<hongkang>","<Macao>","<gua>","<shi>","<ling>","<min>","<hang>","<shen>", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 2;  //单层车牌，双层车牌两类


const char* INPUT_BLOB_NAME = "images"; //onnx 输入  名字
const char* OUTPUT_BLOB_NAME = "output"; //onnx 输出 名字
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img,int &top,int &left)  //对应yolov5中的letter_box
{
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    left = (INPUT_W-unpad_w)/2;
    top = (INPUT_H-unpad_h)/2;
    int right = INPUT_W-unpad_w-left;
    int bottom = INPUT_H-unpad_h-top;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
      
    cv::Mat out;
  
    cv::copyMakeBorder(re,out,top,bottom,left,right,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
 
    return out;
}

struct Object
{
    cv::Rect_<float> rect; //
    float landmarks[8]; //4个关键点
    int label;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

std::vector<int>  my_nms(std::vector<bbox> &bboxes, float nms_threshold)
{
    std:: vector<int> choice;
    for(int i = 0; i<bboxes.size(); i++)
    {
        int keep = 1;
        for(int j = 0; j<choice.size(); j++)
        {
            float  IOU = get_IOU(bboxes[i],bboxes[choice[j]]);
            if (IOU>nms_threshold)
            keep = 0;
        }

        if (keep)
         choice.push_back(i);
    }
    return choice;
}

int find_max(float *prob,int num) //找到类别
{
    int max= 0;
    for(int i=1; i<num; i++)
    {
        if (prob[max]<prob[i])
         max = i;
    }

    return max;

}


static void generate_yolox_proposals(float *feat_blob, float prob_threshold,
                                     std::vector<Object> &objects,int OUTPUT_CANDIDATES) {
  const int num_class = 2;
  const int ckpt=12  ; //yolov7 是12，yolov5是8

  const int num_anchors = OUTPUT_CANDIDATES;

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    // const int basic_pos = anchor_idx * (num_class + 5 + 1);
    // float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = feat_blob[basic_pos + 5];
    // float score = feat_blob[basic_pos + 5 + 1 + cls_id];
    // score *= box_objectness;


    const int basic_pos = anchor_idx * (num_class + 5 + ckpt); //5代表 x,y,w,h,object_score  8代表4个关键点
    float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = find_max(&feat_blob[basic_pos +5+ckpt],num_class);   //找到类别v5
    int cls_id = find_max(&feat_blob[basic_pos +5],num_class);   //v7
    // float score = feat_blob[basic_pos + 5 +8 + cls_id]; //v5
    float score = feat_blob[basic_pos + 5 + cls_id];  //v7
    score *= box_objectness; 


    if (score > prob_threshold) {
      // yolox/models/yolo_head.py decode logic
      float x_center = feat_blob[basic_pos + 0];
      float y_center = feat_blob[basic_pos + 1];
      float w = feat_blob[basic_pos + 2];
      float h = feat_blob[basic_pos + 3];
      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      
    //   float *landmarks=&feat_blob[basic_pos +5]; //v5
    float *landmarks=&feat_blob[basic_pos +5+num_class];

      Object obj;
      obj.rect.x = x0;
      obj.rect.y = y0;
      obj.rect.width = w;
      obj.rect.height = h;
      obj.label = cls_id;
      obj.prob = score;
      int k = 0;
    //   for (int i = 0; i<ckpt; i++)
    //   {
    //      obj.landmarks[k++]=landmarks[i];
    //   }

   
         obj.landmarks[0]=landmarks[0];
          obj.landmarks[1]=landmarks[1];
           obj.landmarks[2]=landmarks[3];
            obj.landmarks[3]=landmarks[4];
             obj.landmarks[4]=landmarks[6];
              obj.landmarks[5]=landmarks[7];
               obj.landmarks[6]=landmarks[9];
                obj.landmarks[7]=landmarks[10];
               
      
      

      objects.push_back(obj);
    }
  }
}


static void generate_proposals(float *feat_blob, float prob_threshold,
                                     std::vector<bbox> &bboxes,int OUTPUT_CANDIDATES) {
  const int num_class = 3;

  const int num_anchors = OUTPUT_CANDIDATES;

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    // const int basic_pos = anchor_idx * (num_class + 5 + 1);
    // float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = feat_blob[basic_pos + 5];
    // float score = feat_blob[basic_pos + 5 + 1 + cls_id];
    // score *= box_objectness;


    const int basic_pos = anchor_idx * (num_class + 5 + 8); //5代表 x,y,w,h,object_score  8代表4个关键点
    float box_objectness = feat_blob[basic_pos + 4];

    int cls_id = find_max(&feat_blob[basic_pos +5+8],num_class);   //找到类别
    float score = feat_blob[basic_pos + 5 +8 + cls_id];
    score *= box_objectness;


    if (score > prob_threshold) {
      // yolox/models/yolo_head.py decode logic
      float x_center = feat_blob[basic_pos + 0];
      float y_center = feat_blob[basic_pos + 1];
      float w = feat_blob[basic_pos + 2];
      float h = feat_blob[basic_pos + 3];
      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      
      float *landmarks=&feat_blob[basic_pos +5];
    

    bbox obj;
     obj.x1=x0;
     obj.y1=y0;
     obj.x2=x0+w;
     obj.y2=y0+h;
     obj.score = score;
      for (int i = 0; i<8; i++)
      {
         obj.landmarks[i]=landmarks[i];
      }
      

      bboxes.push_back(obj);
    }
  }
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    int k = 0;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                // blob[c * img_w * img_h + h * img_w + w] =
                //     (float)img.at<cv::Vec3b>(h, w)[c];
                    blob[k++] =
                    (float)img.at<cv::Vec3b>(h, w)[2-c]/255.0;
            }
        }
    }
    return blob;
}

void blobFromImage_plate(cv::Mat& img,float mean_value,float std_value,float *blob)
{
    // float* blob = new float[img.total()*3];
    // int channels = NUM_CLASSES;
    int img_h = img.rows;
    int img_w = img.cols;
    int k = 0;
    for (size_t c = 0; c <3; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                    blob[k++] =
                    ((float)img.at<cv::Vec3b>(h, w)[c]/255.0-mean_value)/std_value;
            }
        }
    }
    // return blob;
}

static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h,int OUTPUT_CANDIDATES,int top,int left) {
        std::vector<Object> proposals;
        std::vector<bbox> bboxes;
        generate_yolox_proposals(prob,  BBOX_CONF_THRESH, proposals,OUTPUT_CANDIDATES);
        // generate_proposals(prob,  BBOX_CONF_THRESH, bboxes,OUTPUT_CANDIDATES);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);
        // std::sort(bboxes.begin(),bboxes.end(),my_func);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
        // auto choice =my_nms(bboxes, NMS_THRESH);

        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x-left) / scale;
            float y0 = (objects[i].rect.y-top) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width-left) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height-top) / scale;
            
            float *landmarks = objects[i].landmarks;
            for(int i= 0; i<8; i++)
            {
                if(i%2==0)
                landmarks[i]=(landmarks[i]-left)/scale;
                else
                landmarks[i]=(landmarks[i]-top)/scale;
            }
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[4][3] =
{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
    int pos = f.find_last_of("/");
    auto substr = f.substr(pos+1);
    std::string savePath = "/mnt/Gpan/Mydata/pytorchPorject/yoloxNew/newYoloxCpp/result_pic/"+substr;
    cv::imwrite(savePath, image);
    // fprintf(stderr, "save vis file\n");
    // cv::imshow("image", image);
    // cv::waitKey(0);
}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape,const char *INPUT_BLOB_NAME,const char *OUTPUT_BLOB_NAME) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    // context.enqueueV2( buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

float getNorm2(float x,float y)
{
    return sqrt(x*x+y*y);
}

cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]) //透视变换
{
      cv::Point2f w1=order_rect[0]-order_rect[1];
            cv::Point2f w2=order_rect[2]-order_rect[3];
            auto width1 = getNorm2(w1.x,w1.y);
            auto width2 = getNorm2(w2.x,w2.y);
            auto maxWidth = std::max(width1,width2);

            cv::Point2f h1=order_rect[0]-order_rect[3];
            cv::Point2f h2=order_rect[1]-order_rect[2];
            auto height1 = getNorm2(h1.x,h1.y);
            auto height2 = getNorm2(h2.x,h2.y);
            auto maxHeight = std::max(height1,height2);
            //  透视变换
            std::vector<cv::Point2f> pts_ori(4);
            std::vector<cv::Point2f> pts_std(4);

            pts_ori[0]=order_rect[0];
            pts_ori[1]=order_rect[1];
            pts_ori[2]=order_rect[2];
            pts_ori[3]=order_rect[3];

            pts_std[0]=cv::Point2f(0,0);
            pts_std[1]=cv::Point2f(maxWidth,0);
            pts_std[2]=cv::Point2f(maxWidth,maxHeight);
            pts_std[3]=cv::Point2f(0,maxHeight);

            cv::Mat M = cv::getPerspectiveTransform(pts_ori,pts_std);
            cv:: Mat dstimg;
            cv::warpPerspective(src_img,dstimg,M,cv::Size(maxWidth,maxHeight));
            return dstimg;
}
 
cv::Mat get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));

    return out;
}


std::string decode_outputs(float *prob,int output_size)
{
    std::string plate ="";
    std::string pre_str ="#";
    for (int i = 0; i<output_size; i++)
    {
       int  index = int(prob[i]);
        if (plate_string[index]!="#" && plate_string[index]!=pre_str)
            plate+=plate_string[index];
        pre_str = plate_string[index];
        
    }
    return plate;
}

std::string decode_outputs_pingyin(float *prob,int output_size) //拼音
{
    std::string plate ="";
    std::string pre_str ="#";
    for (int i = 0; i<output_size; i++)
    {
       int  index = int(prob[i]);
        if (plate_string_yinwen[index]!="#" && plate_string_yinwen[index]!=pre_str)
            plate+=plate_string_yinwen[index];
        pre_str = plate_string_yinwen[index];
        
    }
    return plate;
}

void doInference_cu(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize,int OUTPUT_SIZE) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}



int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    char *trtModelStreamDet{nullptr};
    char *trtModelStreamRec{nullptr};
    size_t size{0};
    size_t size_rec{0};
    // argv[1]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/build/plate_detect.trt"; 
    // argv[2]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/build/plate_rec.trt";
    // argv[3]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/test_imgs/single_blue.jpg";
    // argv[4]="output.jpg";

    const std::string engine_file_path {argv[1]};  
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }

    const std::string engine_file_path_rec {argv[2]};
    std::ifstream file_rec(engine_file_path_rec, std::ios::binary);
    if (file_rec.good()) {
        file_rec.seekg(0, file_rec.end);
        size_rec = file_rec.tellg();
        file_rec.seekg(0, file_rec.beg);
        trtModelStreamRec = new char[size_rec];
        assert(trtModelStreamRec);
        file_rec.read(trtModelStreamRec, size_rec);
        file_rec.close();
    }

    //det模型trt初始化
    IRuntime* runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    ICudaEngine* engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr); 
    IExecutionContext* context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;

    //rec模型trt初始化
    IRuntime* runtime_rec = createInferRuntime(gLogger);
    assert(runtime_rec!= nullptr);
    ICudaEngine* engine_rec = runtime_rec->deserializeCudaEngine(trtModelStreamRec, size_rec);
    assert(engine_rec != nullptr); 
    IExecutionContext* context_rec = engine_rec->createExecutionContext();
    assert(context_rec != nullptr);
    delete[] trtModelStreamRec;

    float *buffers[2];
    const int inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
   

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    int OUTPUT_CANDIDATES = out_dims.d[1];

       for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }


    CHECK(cudaMalloc((void**)&buffers[inputIndex],  3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc((void**)&buffers[outputIndex], output_size * sizeof(float)));


     // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    auto out_dims_rec = engine_rec->getBindingDimensions(1);
    auto output_size_rec = 1;
    int OUTPUT_CANDIDATES_REC = out_dims_rec.d[1];

    for(int j=0;j<out_dims_rec.nbDims;j++) {
        output_size_rec *= out_dims_rec.d[j];
    }

    static float* prob = new float[output_size];

    static float* prob_rec = new float[output_size_rec];

      
 
 // 识别模型 参数
     int plate_rec_input_w = 168;  
    int plate_rec_input_h = 48;
    float* blob_rec=new float[plate_rec_input_w*plate_rec_input_h*3];

    float mean_value=0.588;
    float std_value =0.193;

    const char* plate_rec_input_name = "images"; //onnx 输入  名字
    const char* plate_rec_out_name= "output"; //onnx 输出 名字

//  识别模型 参数
    
    cv::Point2f rect[4];
    cv::Point2f order_rect[4];
    cv::Point  point[1][4];

    // std::string imgPath ="/mnt/Gpan/Mydata/pytorchPorject/Chinese_license_plate_detection_recognition/imgs";
    std::string input_image_path=argv[3];
     std::string imgPath=argv[3];
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);
    double sumTime = 0;
    int index = 0;
    for (auto &input_image_path:imagList) 
    {
        
        cv::Mat img = cv::imread(input_image_path);
          double begin_time = cv::getTickCount();
         float *buffer_idx = (float*)buffers[inputIndex];
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = INPUT_H * INPUT_W * 3;
        memcpy(img_host, img.data, size_image);
       
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
        double time_pre = cv::getTickCount();
        double time_pre_=(time_pre-begin_time)/cv::getTickFrequency()*1000;
        // std::cout<<"preprocessing time is "<<time_pre_<<" ms"<<std::endl;
      
        doInference_cu(*context_det,stream, (void**)buffers,prob,1,output_size);
        
         
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    int left = (INPUT_W-unpad_w)/2;
    int top = (INPUT_H-unpad_h)/2;
        //    if (index)
        //     {
        //         double use_time =(cv::getTickCount()-begin_time)/cv::getTickFrequency()*1000;
        //         sumTime+=use_time;
        //     }
        int img_w = img.cols;
        int img_h = img.rows;
        // int top=0;
        // int left= 0;
        // cv::Mat pr_img = static_resize(img,top,left);
        // float* blob_detect;
        
        // blob_detect = blobFromImage(pr_img);
      
       
        float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        
        //run inference
        // auto start = cv::getTickCount();
      
        // doInference(*context_det, blob_detect, prob, output_size, pr_img.size(),INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
        
        // auto end = cv::getTickCount();
       
        // if (index)
        // sumTime+=double((end-begin_time)/cv::getTickFrequency()*1000);
        // std::cout << double((end-start)/cv::getTickFrequency()*1000) << "ms" << std::endl;


        std::vector<Object> objects;
      
        decode_outputs(prob, objects, scale, img_w, img_h,OUTPUT_CANDIDATES,top,left);
        
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
       
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout<<input_image_path<<" ";
        
        for (int i = 0; i<objects.size(); i++)
        {
            // cv::rectangle(img, objects[i].rect, cv::Scalar(0,255,0), 2);
            for (int j= 0; j<4; j++)
            {
            // cv::Scalar color = cv::Scalar(color_list[j][0], color_list[j][1], color_list[j][2]);
            // cv::circle(img,cv::Point(objects[i].landmarks[2*j], objects[i].landmarks[2*j+1]),5,color,-1);
            order_rect[j]=cv::Point(objects[i].landmarks[2*j],objects[i].landmarks[2*j+1]);
            }
            
           cv::Mat roiImg = getTransForm(img,order_rect);  //根据关键点进行透视变换
           int label = objects[i].label;
           if (label)             //判断是否双层车牌，是的话进行分割拼接
                roiImg=get_split_merge(roiImg);
            //    cv::imwrite("roi.jpg",roiImg);
            cv::resize(roiImg,roiImg,cv::Size(plate_rec_input_w,plate_rec_input_h));
            cv::Mat pr_img =roiImg;
            // std::cout << "blob image" << std::endl;
          
            auto rec_b = cv::getTickCount();
            blobFromImage_plate(pr_img,mean_value,std_value,blob_rec);
            auto rec_e = cv::getTickCount();
            auto rec_gap = (rec_e-rec_b)/cv::getTickFrequency()*1000;
            
            doInference(*context_rec, blob_rec, prob_rec, output_size_rec, pr_img.size(),plate_rec_input_name,plate_rec_out_name);
            auto plate_number = decode_outputs(prob_rec,output_size_rec);
            auto plate_number_pinyin= decode_outputs_pingyin(prob_rec,output_size_rec); 
            cv::Point origin; 
            origin.x = objects[i].rect.x;
            origin.y = objects[i].rect.y;
            cv::putText(img, plate_number_pinyin, origin, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
            std::cout<<" "<<plate_number;
        
        }
          double end_time = cv::getTickCount();
          auto time_gap = (end_time-begin_time)/cv::getTickFrequency()*1000;
        std::cout<<"  time_gap: "<<time_gap<<"ms ";
         if (index)
            {
                // double use_time =(cv::getTickCount()-begin_time)/cv::getTickFrequency()*1000;
                sumTime+=time_gap;
            }
        std::cout<<std::endl;
        // delete [] blob_detect;
        index+=1;
    }

//    cv::imwrite("out.jpg",img);
 
    // destroy the engine
    std::cout<<"averageTime:"<<(sumTime/(imagList.size()-1))<<"ms"<<std::endl;
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();
 
    context_rec->destroy();
    engine_rec->destroy();
    runtime_rec->destroy();
   delete [] blob_rec;
    cudaStreamDestroy(stream);
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    return 0;
}
