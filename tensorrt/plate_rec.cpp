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

const std::vector<std::string> plate_string={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖", \
"闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","深", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};
using namespace nvinfer1;
// stuff we know about the network and the input/output blobs

static Logger gLogger;

float* blobFromImage_plate(cv::Mat& img,float mean_value,float std_value){
    float* blob = new float[img.total()*3];
    int channels = 3;
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
    return blob;
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

int main(int argc, char** argv) 
{
   
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    int plate_rec_input_w = 168;  
    int plate_rec_input_h = 48;


    float mean_value=0.588;
    float std_value =0.193;

    const char* plate_rec_input_name = "images"; //onnx 输入  名字
    const char* plate_rec_out_name= "output"; //onnx 输出 名字

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python3 yolox/deploy/trt.py -n yolox-{tiny, s, m, l, x}' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./yolox ../model_trt.engine -i ../../../assets/dog.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};

    //std::vector<std::string> file_names;
    //if (read_files_in_dir(argv[2], file_names) < 0) {
        //std::cout << "read_files_in_dir failed." << std::endl;
        //return -1;
    //}

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    int OUTPUT_CANDIDATES = out_dims.d[1];

    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static float* prob = new float[output_size];
    


    std::string imgPath ="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/result";

    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);
    double sumTime = 0;
    int right_label = 0;
    int file_num = imagList.size();
    for (auto &input_image_path:imagList) 
    {
        cv::Mat img = cv::imread(input_image_path);
        int img_w = img.cols;
        int img_h = img.rows;
        int top=0;
        int left= 0;
        cv::resize(img,img,cv::Size(plate_rec_input_w,plate_rec_input_h));
        cv::Mat pr_img =img;
        // std::cout << "blob image" << std::endl;
        float* blob;
        blob = blobFromImage_plate(pr_img,mean_value,std_value);
        doInference(*context, blob, prob, output_size, pr_img.size(),plate_rec_input_name,plate_rec_out_name);
        auto plate_number = decode_outputs(prob,output_size);
        
        int pos = input_image_path.find_last_of("/");
        auto image_name = input_image_path.substr(pos+1);

        int pos2= image_name.find_last_of("_");
        auto gt=image_name.substr(0,pos2);
        if(gt==plate_number)
          right_label+=1;

        std::cout<<input_image_path<<" "<<right_label<<" "<<plate_number<<std::endl;
    delete blob;
    }
    printf("sum is %d,right is %d ,accuracy is %.4f",file_num,right_label,1.0*right_label/file_num);
    
 
    // destroy the engine
    // std::cout<<"averageTime:"<<sumTime/imagList.size()<<std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
