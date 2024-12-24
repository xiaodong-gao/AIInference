#include "Detector.cuh"

// CUDA�ںˣ����ڶ��ROI�ü�ͼ�� (2D Kernel)
__global__ void extractRoiKernel(const unsigned char* d_image, unsigned char* d_output, roi_t* d_rois, int roi_count, int width, int height, int channels) {
	// �����̵߳�ȫ������
	int tx = threadIdx.x + blockIdx.x * blockDim.x; // ��������
	int ty = threadIdx.y + blockIdx.y * blockDim.y; // ��������
	// ȷ����������Ч��Χ��
	if (tx < roi_count) {
		// ��ȡ��ǰ ROI
		roi_t roi = d_rois[tx];
		// ����� ROI ��λ�úʹ�С
		int roi_x = roi.x;
		int roi_y = roi.y;
		int roi_width = roi.w;
		int roi_height = roi.h;
		// ���� ROI ������ȡͼ������
		for (int i = 0; i < roi_height; i++) {
			for (int j = 0; j < roi_width; j++) {
				int global_x = roi_x + j;  // ��ǰ���ص�ȫ�� x ����
				int global_y = roi_y + i;  // ��ǰ���ص�ȫ�� y ����
				// ���������Ƿ���ͼ��Χ��
				if (global_x < width && global_y < height) {
					int image_index = (global_y * width + global_x) * channels;  // ͼ�������
					int output_index = (tx * roi_width * roi_height + i * roi_width + j) * channels;  // ���������
					// ��ȡÿ��ͨ��������ֵ
					for (int c = 0; c < channels; c++) {
						d_output[output_index + c] = d_image[image_index + c];
					}
				}
			}
		}
	}
}

/*
// CUDA �ںˣ����ݶ�� ROI �ü�ͼ��
__global__ void cropImageKernel(const uchar* inputImage, uchar* outputImage, int imageWidth, int imageHeight, int numROIs, ROI* rois) {
	int roiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (roiIdx < numROIs) {
		ROI roi = rois[roiIdx];

		// ����Խ��
		if (roi.x + roi.width <= imageWidth && roi.y + roi.height <= imageHeight) {
			// ����ÿ��ROI����ʼλ�ú����ͼ���ƫ����
			int outputOffset = roiIdx * roi.width * roi.height;

			for (int y = 0; y < roi.height; ++y) {
				for (int x = 0; x < roi.width; ++x) {
					int inputOffset = (roi.y + y) * imageWidth + (roi.x + x);
					int outputIdx = outputOffset + y * roi.width + x;
					outputImage[outputIdx] = inputImage[inputOffset];
				}
			}
		}
	}
}

*/

Detector::Detector(int gpu) 
	:gpu_{gpu} {
}

Detector::~Detector() {

}

ErrorCode Detector::init(const char* type, const char* weights_file,  float confidence_threshold, float nms_threshold) {
	std::string str_type(type);
	yolo::Type yolo_type;
	if (str_type == "V3") {
		yolo_type = yolo::Type::V3;
	}
	else if (str_type == "V5") {
		yolo_type = yolo::Type::V5;
	}
	else if (str_type == "X") {
		yolo_type = yolo::Type::X;
	}
	else if (str_type == "V7") {
		yolo_type = yolo::Type::V7;
	}
	else if (str_type == "V8" || str_type == "V11") {
		yolo_type = yolo::Type::V8;
	}
	else if (str_type == "V8Seg" || str_type == "V11Seg") {
		yolo_type = yolo::Type::V8Seg;
	}
	else {
		return ErrorCode::TYPE_ERROR;
	}

	yolo_ = yolo::load(weights_file, yolo_type, confidence_threshold, nms_threshold);
	if (nullptr == yolo_)
		return ErrorCode::YOLO_LOAD_ERROR;

	return ErrorCode::SUCCESS;
}

ErrorCode Detector::init_extract_rois(int img_width, int img_height, int img_channels, int sub_img_width, int sub_img_height, int roi_count) {
	cudaMalloc(&d_inputImage, img_width * img_height * img_channels * sizeof(unsigned char));
	cudaMalloc(&d_outputImage, roi_count * sub_img_width * sub_img_height * img_channels * sizeof(unsigned char));  // ����ÿ�� ROI �������Ϊ 100x100
	cudaMalloc(&d_rois, roi_count * sizeof(roi_t));
	return ErrorCode::SUCCESS;
}

ErrorCode Detector::split_image_with_overlap(int imgWidth, int imgHeight, int startX, int startY, int roiWidth, int roiHeight, int stepX, int stepY, std::vector<roi_t>& rois) {
	// ���� ROI ������
	for (int y = startY; y <= imgHeight; y += stepY) {
		for (int x = startX; x <= imgWidth; x += stepX) {
			//rois.push_back(roi_t{x,y,roiWidth ,roiHeight });
			int roi_x = x;
			int roi_y = y;
			int roi_w = roiWidth;
			int roi_h = roiHeight;
			// ȷ������ ROI ���ᳬ��ͼ��߽�
			if (roi_x + roi_w > imgWidth) {
				//roi_w = imgWidth - roi_x;
				roi_w = roiWidth;
			}
			if (roi_y + roi_h > imgHeight) {
				//roi_h = imgHeight - roi_y;
				roi_h = roiHeight;
			}
			// ����� ROI ��ӵ������
			rois.push_back(roi_t{ roi_x, roi_y, roi_w, roi_h });
		}
	}
	return ErrorCode::SUCCESS;
}

ErrorCode Detector::extract_rois(const unsigned char* d_image, std::vector<unsigned char> &h_output, std::vector<roi_t> rois, int subWidth, int subHeight,int img_width, int img_height, int img_channels) {
	
	// ��ͼ�����ݺ� ROI ���ݴ��������䵽 GPU
	int numROIs = rois.size();
	cudaMemcpy(d_inputImage, d_image, img_width * img_height * img_channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rois, rois.data(), numROIs * sizeof(roi_t), cudaMemcpyHostToDevice);
	dim3 blockSize(32, 16);
	dim3 gridSize((numROIs + blockSize.x - 1) / blockSize.x, 1);
	// ���� CUDA �˺���
	extractRoiKernel <<<gridSize, blockSize>>> (d_inputImage, d_outputImage, d_rois, numROIs, img_width, img_height, img_channels);
	// �ȴ� CUDA �˺���ִ�����
	cudaDeviceSynchronize();


	// ��� CUDA �ں�ִ���Ƿ�ɹ�
	if (cudaGetLastError() != cudaSuccess) {
		ErrorCode::EXTRACT_ROIS_FAILED;
	}
	// �� GPU �����ü������ CPU
	h_output = std::vector<unsigned char>(numROIs * subWidth * subHeight * img_channels);       // �������ߴ��� subWidth x subHeight
	cudaMemcpy(h_output.data(), d_outputImage, numROIs * subWidth * subHeight * img_channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return ErrorCode::SUCCESS;
}

ErrorCode Detector::detect(image_t img,  float score, std::vector<bbox_t>& result) {
	yolo::BoxArray objs = yolo_->forward(yolo::Image(img.data, img.w, img.h));
	for (auto& obj : objs) {
		bbox_t box;
		if (obj.confidence < score)
			continue;
		box.prob = obj.confidence;
		box.x = obj.left;
		box.y = obj.top;
		box.w = obj.right - obj.left;
		box.h = obj.bottom - obj.top;
		box.obj_id = obj.class_label;
		result.push_back(box);
	}
	return ErrorCode::SUCCESS;
}

ErrorCode Detector::detectBatch(image_t img, int batch_size, int width, int height,  float score, std::vector<std::vector<bbox_t>>& result) {
	return ErrorCode::SUCCESS;
}	

ErrorCode Detector::dispose() {
	yolo_.reset();
	return ErrorCode::SUCCESS;
}

ErrorCode Detector::dispose_extract_rois() {
	// �ͷ� GPU �ڴ�
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
	cudaFree(d_rois);
	return ErrorCode::SUCCESS;
}