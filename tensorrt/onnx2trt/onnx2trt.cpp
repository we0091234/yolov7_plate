#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
// #include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <dirent.h>
#include "NvInfer.h"

#include "NvOnnxParser.h"
// #include "NvInferRuntime.h"
#include "logging.h"
#include "cuda_runtime_api.h"
using namespace nvinfer1;
using namespace std;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

void saveToTrtModel(const char * TrtSaveFileName,IHostMemory*trtModelStream)
	{
		std::ofstream out(TrtSaveFileName, std::ios::binary);
		if (!out.is_open())
		{
		std::cout << "打开文件失败!" <<std:: endl;
		}
		out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
		out.close();
	}




void onnxToTRTModel(const std::string& modelFile,unsigned int maxBatchSize,IHostMemory*& trtModelStream,const char * TrtSaveFileName) 
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
 
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);//创建构建器(即指向Ibuilder类型对象的指针)
    IBuilderConfig *config = builder->createBuilderConfig();
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);    //必须加不然报错
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);/*等价于*bulider.createNetwork(),通过Ibulider定义的
    名为creatNetwork()方法，创建INetworkDefinition的对象，ntework这个指针指向这个对象*/ 
 
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());//创建解析器
 
    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();
 
    if (!parser->parseFromFile(modelFile.c_str(), verbosity)) //解析onnx文件，并填充网络
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
 
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 30);
    // builder->setMaxWorkspaceSize(1 << 30);
#ifdef USE_FP16
	config->setFlag(BuilderFlag::kFP16);
#endif
    // samplesCommon::enableDLA(builder, gUseDLACore);
    //当引擎建立起来时，TensorRT会复制
    // ICudaEngine* engine = builder->buildCudaEngine(*network);//通过Ibuilder类的buildCudaEngine()方法创建IcudaEngine对象，
    ICudaEngine *engine = builder->buildEngineWithConfig(*network,*config);
    assert(engine);
 
    // we can destroy the parser
    parser->destroy();
    
    // serialize the engine, 
    // then close everything down
    trtModelStream = engine->serialize();//将引擎序列化，保存到文件中
    engine->destroy();
    network->destroy();
    builder->destroy();
    config->destroy();
    saveToTrtModel(TrtSaveFileName,trtModelStream);
}

	void onnxToTRTModelDynamicBatch(const std::string& modelFile, unsigned int maxBatchSize, IHostMemory*& trtModelStream,const char * TrtSaveFileName,int input_h,int input_w) // output buffer for the TensorRT model 动态batch
{
            int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
        
            // create the builder
            IBuilder* builder = createInferBuilder(gLogger);//创建构建器(即指向Ibuilder类型对象的指针)
            IBuilderConfig *config = builder->createBuilderConfig();
            auto profile = builder->createOptimizationProfile();


            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);    //必须加不然报错
            nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);/*等价于*bulider.createNetwork(),通过Ibulider定义的
            名为creatNetwork()方法，创建INetworkDefinition的对象，ntework这个指针指向这个对象*/ 



            Dims dims = Dims4{1, 3, input_h, input_w};
            profile->setDimensions(INPUT_BLOB_NAME,
                        OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
            profile->setDimensions(INPUT_BLOB_NAME,
                        OptProfileSelector::kOPT, Dims4{maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
            profile->setDimensions(INPUT_BLOB_NAME,
                        OptProfileSelector::kMAX, Dims4{maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
            config->addOptimizationProfile(profile);

        
            auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());//创建解析器
        
            //Optional - uncomment below lines to view network layer information
            //config->setPrintLayerInfo(true);
            //parser->reportParsingInfo();
        
            if (!parser->parseFromFile(modelFile.c_str(), verbosity)) //解析onnx文件，并填充网络
            {
                string msg("failed to parse onnx file");
                gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
                exit(EXIT_FAILURE);
            }
        
            // Build the engine
            // builder->setMaxBatchSize(maxBatchSize);
            config->setMaxWorkspaceSize(1 << 30);
            // builder->setMaxWorkspaceSize(1 << 30);
        #ifdef USE_FP16
            config->setFlag(BuilderFlag::kFP16);
        #endif
            // samplesCommon::enableDLA(builder, gUseDLACore);
            //当引擎建立起来时，TensorRT会复制
            // ICudaEngine* engine = builder->buildCudaEngine(*network);//通过Ibuilder类的buildCudaEngine()方法创建IcudaEngine对象，
            ICudaEngine *engine = builder->buildEngineWithConfig(*network,*config);
            assert(engine);
        
            // we can destroy the parser
            parser->destroy();
            
            // serialize the engine, 
            // then close everything down
            trtModelStream = engine->serialize();//将引擎序列化，保存到文件中
            engine->destroy();
            network->destroy();
            builder->destroy();
            config->destroy();
            saveToTrtModel(TrtSaveFileName,trtModelStream);
        
}

// void  readTrtModel(const char * Trtmodel)  //读取onnx模型
// 	{
// 		size_t size{ 0 };
// 		std::ifstream file(Trtmodel, std::ios::binary);
// 		if (file.good()) {
// 			file.seekg(0, file.end);
// 			size = file.tellg();
// 			file.seekg(0, file.beg);
// 			_trtModelStream = new char[size];
// 			assert(_trtModelStream);
// 			file.read(_trtModelStream, size);
// 			file.close();
// 		}
// 		_trtModelStreamSize = size;

// 		_runtime = createInferRuntime(gLogger);
// 		_engine1 = _runtime->deserializeCudaEngine(_trtModelStream, _trtModelStreamSize);
// 		//cudaSetDevice(0);
// 		context = _engine1->createExecutionContext();
// 	}
	

int main(int argc, char** argv)
{
    IHostMemory* trtModelStream{nullptr};
    int batchSize = atoi(argv[3]);
    // int input_h = atoi(argv[4]);
    // int input_w=atoi(argv[5]);
    onnxToTRTModel(argv[1],batchSize,trtModelStream,argv[2]);
    std::cout<<"convert seccuss!"<<std::endl;
}