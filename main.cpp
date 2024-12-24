#include <opencv2/opencv.hpp>
#include <iostream>
#include "infer.hpp"
#include "yolo.hpp"
#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include "Detector.cuh"
#include <thread>  // For sleep_for example
/*
struct bbox_t {
    unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;        // class of object - from range [0, classes-1]
    //unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
    //unsigned int frames_counter;// counter of frames on which the object was detected
};
*/

yolo::Image cvimg(const cv::Mat& image) { return yolo::Image(image.data, image.cols, image.rows); }

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec) {
    for (auto& i : result_vec) {
        cv::Scalar color(60, 160, 260);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
        //if (obj_names.size() > i.obj_id)
        //    putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        //if (i.track_id > 0)
        //    putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
    }
    cv::namedWindow("window name", cv::WINDOW_NORMAL);
    cv::imshow("window name", mat_img);
    cv::waitKey(0);
}


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto& i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

int main()
{
    cv::Mat image = cv::imread("bus.jpg", cv::IMREAD_UNCHANGED);
    // 检查图像是否成功加载
    if (image.empty()) {
        //std::cerr << "图像加载失败!" << std::endl;
        return -1;
    }

    // 获取图像的宽度、高度和通道数
    int img_width = image.cols;      // 图像的宽度
    int img_height = image.rows;     // 图像的高度
    int img_channels = image.channels(); // 图像的通道数（例如 RGB 通道数为 3）

    std::unique_ptr<Detector> detector_ = std::make_unique<Detector>();

    // 重叠区域
    int overlap_x = 0;
    int overlap_y = 0;

    // 子图像尺寸
    int subWidth = 640;
    int subHeight = 640;

    // 步长
    int strideX = subWidth - overlap_x; // 横向步长
    int strideY = subHeight - overlap_y; // 纵向步长

    std::vector<roi_t> rois;
    //step1 cpu 设置参数
    ErrorCode error_code = detector_->split_image_with_overlap(img_width, img_height, 0, 0, subWidth, subHeight, strideX, strideY, rois);
    if (error_code != ErrorCode::SUCCESS) {

    }

    float confidence_threshold = 0.1;
    float nms_threshold = 0.4;
    const char* type = "V11Seg";
    const char* weights_file = "yolo11n-seg.transd.fp16.engine";
    error_code = detector_->init(type, weights_file, confidence_threshold, nms_threshold);
    if (error_code != ErrorCode::SUCCESS) {

    }

    //step2 设置显存 空间大小用于提取roi
    error_code = detector_->init_extract_rois(img_width, img_height, img_channels, subWidth, subHeight, rois.size());
    if (error_code != ErrorCode::SUCCESS) {

    }

    //step3 gpu提取图像的多个roi区域
    std::vector<unsigned char> h_output;
    error_code = detector_->extract_rois(image.data, h_output, rois, subWidth, subHeight, img_width, img_height, img_channels);
    if (error_code != ErrorCode::SUCCESS) {

    }

    // 创建裁剪图像并显示（假设裁剪区域的最大尺寸为 100x100）
    std::vector<bbox_t> result;
    for (int i = 0; i < rois.size(); ++i) {
        roi_t roi = rois[i];

        cv::Mat croppedImage(roi.h, roi.w, image.type(), h_output.data() + i * roi.w * roi.h * img_channels);
        cv::imwrite("Cropped Image " + std::to_string(i) + ".bmp", croppedImage);
  
        result.clear();
        image_t img;
        img.data = h_output.data() + i * roi.w * roi.h * img_channels;
        img.h = roi.h;
        img.w = roi.w;
        img.c = img_channels;
        
        detector_->detect(img, result);

        draw_boxes(croppedImage, result);
        
    }
    
    //step4 释放gpu资源
    error_code = detector_->dispose_extract_rois();
    if (error_code != ErrorCode::SUCCESS) {

    }

    error_code = detector_->dispose();
    if (error_code != ErrorCode::SUCCESS) {

    }

    

   


    /*auto yolo = yolo::load("yolo11n-seg.transd.fp16.engine", yolo::Type::V8Seg);
    if (yolo == nullptr)
        return -1;
    // 获取当前时间作为开始时间
    auto start = std::chrono::high_resolution_clock::now();
    yolo::BoxArray objs;
    for(int i  = 0; i < 100; i++)
        objs  = yolo->forward(cvimg(image));
    // 获取当前时间作为结束时间
    auto end = std::chrono::high_resolution_clock::now();
    // 计算程序运行时间，单位为毫秒
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // 输出程序运行时间
    std::cout << "程序运行时间: " << duration.count() / 100 << " 毫秒" << std::endl;
    auto obj_names = objects_names_from_file("voc.names");
    std::vector<bbox_t> result_vec;
    for (auto& obj : objs) {
        bbox_t box;
        box.x = obj.left;
        box.y = obj.top;
        box.w = obj.right - obj.left;
        box.h = obj.bottom - obj.top;
        result_vec.push_back(box);
    }

    draw_boxes(image, result_vec, obj_names);

    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("Result.jpg", image);
    */
    return 0;
}
