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
#include <filesystem>
#include "Detector.cuh"

#define SAVE_RESULT 1

namespace fs = std::filesystem;

std::vector<std::string> get_files_in_directory(const std::string& dir_path) {
    std::vector<std::string> files;

    // 检查目录是否存在
    if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            // 只获取文件
            if (fs::is_regular_file(entry.status())) {
                files.push_back(entry.path().string());
            }
        }
    }
    else {
        std::cerr << "目录不存在或不是有效的目录\n";
    }

    return files;
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> const obj_names,std::string img_name) {
    for (auto& i : result_vec) {
        cv::Scalar color(0, 0, 255);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
        if (obj_names.size() > i.obj_id)
            putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        //if (i.track_id > 0)
        //    putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
    }

    cv::imwrite(img_name, mat_img);
    //cv::namedWindow("window name", cv::WINDOW_NORMAL);
    //cv::imshow("window name", mat_img);
    //cv::waitKey(0);
}

void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> obj_names) {
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

cv::Mat readBMP(const char* filename) {
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return cv::Mat();
    }

    // 读取文件头
    unsigned char fileHeader[14];
    file.read(reinterpret_cast<char*>(fileHeader), 14);

    // 读取信息头
    unsigned char infoHeader[40];
    file.read(reinterpret_cast<char*>(infoHeader), 40);

    // 提取图像宽度和高度
    int width = *(int*)&infoHeader[4];
    int height = *(int*)&infoHeader[8];

    // 计算图像数据的大小
    int size = 3 * width * height;

    // 读取图像数据
    unsigned char* data = new unsigned char[size];
    file.read(reinterpret_cast<char*>(data), size);

    // 关闭文件
    file.close();

    // 创建Mat对象
    cv::Mat image(height, width, CV_8UC3, data);

    // OpenCV使用BGR格式，需要将RGB转换为BGR
    //cvtColor(image, image, COLOR_RGB2BGR);

    return image;
}

int main()
{
   std::string dir = "D:/codes/AIInference/src_images";  // 替换为你要检查的目录
    std::vector<std::string> files = get_files_in_directory(dir);
    for (const auto& file : files) {
        // 替换为实际的文件路径
        fs::path file_path = file;  
        // 提取文件名
        std::string file_name = file_path.filename().string();
        //读取图像  获取图像信息
        //cv::Mat image = cv::imread(file, cv::IMREAD_UNCHANGED);
        cv::Mat image = readBMP(file.c_str());
        // 
        // 检查图像是否成功加载
        if (image.empty()) {
            //std::cerr << "图像加载失败!" << std::endl;
            return -1;
        }
        // 获取图像的宽度、高度和通道数
        int img_width = image.cols;      // 图像的宽度
        int img_height = image.rows;     // 图像的高度
        int img_channels = image.channels(); // 图像的通道数（例如 RGB 通道数为 3）

        //创建检测类实例化对象
        std::unique_ptr<Detector> detector_ = std::make_unique<Detector>();
        auto obj_names = objects_names_from_file("voc.names");

        float confidence_threshold = 0.1;
        float nms_threshold = 0.1;
        const char* type = "V11Seg";
        const char* weights_file = "yolo11n-seg.transd.fp32.engine";
        ErrorCode error_code = detector_->init(type, weights_file, confidence_threshold, nms_threshold);
        if (error_code != ErrorCode::SUCCESS) {

        }

        auto start = std::chrono::high_resolution_clock::now();
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
        //step1 cpu 设置参数 分离roi信息
        error_code = detector_->split_image_with_overlap(img_width, img_height, 0, 0, subWidth, subHeight, strideX, strideY, rois);
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

        // step4 推理检测结果
        std::vector<bbox_t> result;
        float score = 0.45;
        for (int i = 0; i < rois.size(); ++i) {
            roi_t roi = rois[i];

#if SAVE_RESULT
            cv::Mat croppedImage(roi.h, roi.w, image.type(), h_output.data() + i * roi.w * roi.h * img_channels);
            //cv::imwrite("./result_images/" + file_name + std::to_string(i) + ".bmp", croppedImage);
#endif
            result.clear();
            image_t img;
            img.data = h_output.data() + i * roi.w * roi.h * img_channels;
            img.h = roi.h;
            img.w = roi.w;
            img.c = img_channels;
            detector_->detect(img, score, result);
#if SAVE_RESULT
            std::string result_image_name = "./result_images/result_" + file_name + std::to_string(i) + ".bmp";
            draw_boxes(croppedImage, result, obj_names, result_image_name);
#endif
        }
        // 获取当前时间作为结束时间
        auto end = std::chrono::high_resolution_clock::now();
        // 计算程序运行时间，单位为毫秒
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // 输出程序运行时间
        std::cout << "程序运行时间: " << duration.count() / 100 << " 毫秒" << std::endl;

        //step5 释放gpu资源
        error_code = detector_->dispose_extract_rois();
        if (error_code != ErrorCode::SUCCESS) {

        }

        error_code = detector_->dispose();
        if (error_code != ErrorCode::SUCCESS) {

        }

    }
    return 0;
}
