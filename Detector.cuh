#ifndef DETECTOR_CUH_
#define DETECTOR_CUH_


//#define OPENCV 
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#ifdef OPENCV
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/highgui/highgui_c.h>   // C
#include <opencv2/imgproc/imgproc_c.h>   // C
#endif

#include "Common.h"
#include "ErrorCode.h"
#include "infer.hpp"
#include "yolo.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Detector {
public:
	explicit Detector() = default;
	explicit Detector(int gpu);
	~Detector();
	Detector(const Detector& rhs) = delete;
	Detector& operator=(const Detector& rhs) = delete;

	ErrorCode split_image_with_overlap(int startX, int startY,int endX,int endY, int roiWidth, int roiHeight, int stepX, int stepY, std::vector<roi_t>& rois);

	ErrorCode init(const char* type, const char* weights_file, float confidence_threshold = 0.2, float nms_threshold = 0.45);
	ErrorCode init_extract_rois(int img_width, int img_height, int img_channels, int sub_img_width, int sub_img_height, int roi_count);
	ErrorCode extract_rois(const unsigned char* d_image, std::vector<unsigned char> &h_output , std::vector<roi_t> rois, int subWidth,int subHeight,int width, int height, int channels);
	ErrorCode detect(image_t img, float score,std::vector<bbox_t> &result);
	ErrorCode detectBatch(image_t img, int batch_size, int width, int height,  float score, std::vector<std::vector<bbox_t>> &result);
	ErrorCode dispose();
	ErrorCode dispose_extract_rois();

private:
	std::shared_ptr<yolo::Infer> yolo_;
	int gpu_;
	unsigned char* d_inputImage;
	unsigned char* d_outputImage;
	roi_t* d_rois;

};

#endif
