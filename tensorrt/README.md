# yolov7 车牌识别TensorRT

1. 修改CMakeLists.txt  换成你的cuda  tensorrt  以及opencv 路径

   ```
   #cuda 
   include_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/include)
   link_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/lib)

   #tensorrt 
   include_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/include/)
   link_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/lib/)
   ```
2. build

   ```
   1. mkdir build
   2. cmake ..
   3. make

   ```
3. onnx 转成tensorrt模型  onnx模型看这里[p868](https://pan.baidu.com/s/1FRqreQcu7mlaZXcI8eo_5Q)    或者训练自己的模型导出

   ```
   当前在build目录
   #1 生成检测模型
   ./onnx2trt/onnx2trt  ../v7_onnx_model/plate_detect.onnx ./plate_detect.trt  1
   #2 生成识别模型
   ./onnx2trt/onnx2trt  ../v7_onnx_model/plate_rec.onnx ./plate_rec.trt  1
   ```
4. 推理

   ```
   ./plate_rec ./plate_detect.trt  ./plate_rec.trt ../test_imgs
   ```
   结果显示在控制台
