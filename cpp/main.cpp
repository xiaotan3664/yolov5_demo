//
// Created by yuan on 1/22/21.
//

#include "opencv2/opencv.hpp"
#include "yolov5.hpp"

int main(int argc, char *argv[])
{
    const char *keys="{bmodel | ../data/models/yolov5s_fp32_640_1.bmodel | bmodel file path}"
                     "{tpuid | 0 | TPU device id}"
                     "{help | 0 | Print help information.}"
                     "{is_video | 0 | input video file path}"
                     "{input |../data/images/bus.jpg | input stream file path}"
                     "{classnames |../data/coco.names | class names' file path}";

    // profiling
    TimeStamp ts;

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    int dev_id = parser.get<int>("tpuid");
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    // Load bmodel
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    YoloV5 yolo(bm_ctx);

    yolo.enableProfile(&ts);
    std::string coco_names = parser.get<std::string>("classnames");

    CV_Assert(0 == yolo.Init(coco_names));

    if (!parser.get<bool>("is_video")) {
        std::string image_file = parser.get<std::string>("input");
        std::cout<<"input image: "<<image_file<<std::endl;
        cv::Mat img = cv::imread(image_file);
        std::vector<cv::Mat> images;
        // add image to list
        images.push_back(img);

        std::vector<YoloV5BoxVec> boxes;
        CV_Assert(0 == yolo.Detect(images, &boxes));

        std::cout<<std::endl;
        for (int i = 0; i < (int) images.size(); ++i) {

            cv::Mat frame = images[i];
            auto frame_boxes = boxes[i];
            for (auto bbox : boxes[i]) {
                std::cout << "class id =" << bbox.class_id << ",score = " << bbox.score
                          << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",y=" << bbox.height << ")"
                          << std::endl;
                yolo.drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width,
                              bbox.y + bbox.height, frame);
            }

            {
                std::string output_file = cv::format("output_%d.jpg", i);
                cv::imwrite(output_file, frame);
            }
        }

    }else {
        std::string input_url = parser.get<std::string>("input");
        // open stream
        cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
        if (!cap.isOpened()) {
            std::cout << "open stream " << input_url << " failed!" << std::endl;
            exit(1);
        }

        // get resolution
        int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "resolution of input stream: " << h << "," << w << std::endl;

        while(1) {
            cv::Mat img;
            if (!cap.read(img)) {
                std::cout << "Read frame failed!" << std::endl;
                exit(1);
            }

            std::vector<cv::Mat> images;
            images.push_back(img);

            std::vector<YoloV5BoxVec> boxes;

            CV_Assert(0 == yolo.Detect(images, &boxes));

            for (int i = 0; i < (int) images.size(); ++i) {

                cv::Mat frame = images[i];
                for (auto bbox : boxes[i]) {
                    //std::cout << "class id =" << bbox.class_id << ",score = " << bbox.score
                    //          << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",y=" << bbox.height << ")"
                    //         << std::endl;
                    yolo.drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width,
                                  bbox.y + bbox.height, frame);
                }
                std::cout << "detect boxes: " << boxes[i].size() << std::endl;
            }

        }


    }


    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("YoloV5");
    ts.show_summary("YoloV5 Demo");
    ts.clear();

}
