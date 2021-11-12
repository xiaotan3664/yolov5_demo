#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"

#define MAX_BATCH 4

struct YoloV5Box {
    int x, y, width, height;
    float score;
    int class_id;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

class YoloV5 {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    bm_image m_resized_imgs[MAX_BATCH];
    bm_image m_converto_imgs[MAX_BATCH];

    //configuration
    float m_confThreshold= 0.5;
    float m_nmsThreshold = 0.5;
    float m_objThreshold = 0.5;


    std::vector<std::string> m_class_names;
    int m_class_num = 80; // default is coco names
    //const float m_anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
    std::vector<std::vector<std::vector<int>>> m_anchors{{{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}};
    const int m_anchor_num = 3;
    int m_net_h, m_net_w;

    TimeStamp *m_ts;


private:
    int pre_precess(const std::vector<cv::Mat>& images);
    int post_process(const std::vector<cv::Mat>& images, std::vector<YoloV5BoxVec> *boxes);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void NMS(YoloV5BoxVec &dets, float nmsConfidence);

public:
    YoloV5(std::shared_ptr<BMNNContext> context);
    virtual ~YoloV5();

    int Init(const std::string& coco_names_file="");
    int Detect(const std::vector<cv::Mat>& images, std::vector<YoloV5BoxVec> *boxes);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    void enableProfile(TimeStamp *ts);
};





#endif //!YOLOV5_H