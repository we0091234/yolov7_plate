#include "utils.hpp"
std::string getHouZhui(std::string fileName)
{
    //    std::string fileName="/home/xiaolei/23.jpg";
    int pos=fileName.find_last_of(std::string("."));
    std::string houZui=fileName.substr(pos+1);
    return houZui;
}

int readFileList(char *basePath,std::vector<std::string> &fileList,std::vector<std::string> fileType)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)
        {    ///file
            if (fileType.size())
            {
            std::string houZui=getHouZhui(std::string(ptr->d_name));
            for (auto &s:fileType)
            {
            if (houZui==s)
            {
            fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            break;
            }
            }
            }
            else
            {
                fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            }
        }
        else if(ptr->d_type == 10)    ///link file
            printf("d_name:%s/%s\n",basePath,ptr->d_name);
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            readFileList(base,fileList,fileType);
        }
    }
    closedir(dir);
    return 1;
}


void draw_rect(const cv::Mat& image, const std::vector<boundingBox>bboxes,const char* class_names[])
{
    // static const char* class_names[] = {
    //     "head", "leg", "hand", "back", "nostd", "body", "plate", "logo"};

    // cv::Mat image = bgr.clone();

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        // const Object& obj = objects[i];
        const boundingBox &obj= bboxes[i];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list1[obj.label][0], color_list1[obj.label][1], color_list1[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }
        cv::Rect myRect(obj.x,obj.y,obj.w,obj.h);
        cv::rectangle(image,myRect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.x;
        int y = obj.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color,-1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

}


bool cmpBox(boundingBox b1, boundingBox b2)
{
    return b1.score > b2.score;
}

float getIou(boundingBox b1,boundingBox b2)   //计算IOU
{
    int xl1 = b1.x;         //左
    int xr1 = b1.w+b1.x;  // 右
    int yt1=b1.y;          //顶
    int yb1 = b1.y+b1.h;  //底

    int xl2 = b2.x;         //左
    int xr2 = b2.w+b2.x;  // 右
    int yt2=b2.y;          //顶
    int yb2 = b2.y+b2.h;  //底

    int x11 =std::max(xl1,xl2);
    int y11 = std::max(yt1,yt2);
    int x22 = std::min(xr1,xr2);
    int y22 = std::min(yb1,yb2);  

    float intersectionArea= (x22-x11)*(y22-y11);  //交集

    float unionArea = (xr1-xl1)*(yb1-yt1)+(xr2-xl2)*(yb2-yt2)-intersectionArea; //并集
    
    return 1.0f*intersectionArea/unionArea;
}


void myNms(std::vector<boundingBox>&bboxes,float score)
{
     std::sort(bboxes.begin(),bboxes.end(),cmpBox);

    for(int i = 0; i<bboxes.size()-1; i++)
    {
        for(int j = i+1;j<bboxes.size(); j++)
        {
            if(getIou(bboxes[i],bboxes[j])>score)
            {
                bboxes.erase(bboxes.begin()+j);
                j--;
            }
        }
    }
}
