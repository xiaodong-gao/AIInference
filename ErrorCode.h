#ifndef ERRORCODE_H_
#define ERRORCODE_H_

enum class ErrorCode {
    SUCCESS = 0,
    TYPE_ERROR = 1,
    YOLO_LOAD_ERROR = 2,
    EXTRACT_ROIS_FAILED = 3,
    OUT_OF_MEMORY = 4,
};

#endif
