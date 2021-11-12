//
// Created by yuan on 1/22/21.
//

#include "opencv2/opencv.hpp"
#include "yolov5.hpp"

void run_once(cv::Mat& img, YoloV5& yolo, TimeStamp& ts){

	yolo.enableProfile(&ts);
	std::vector<cv::Mat> images;
	// add image to list
	images.push_back(img);

	std::vector<YoloV5BoxVec> boxes;
	CV_Assert(0 == yolo.Detect(images, &boxes));

/*
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
*/

}
int main(int argc, char *argv[])
{
    const char *keys="{bmodel | ../data/models/yolov5s_fp32_640_1.bmodel | bmodel file path}"
                     "{tpuid | 0 | TPU device id}"
                     "{help | 0 | Print help information.}"
                     "{input |../data/images/bus.jpg | input stream file path}"
                     "{run_count | 1 | run count}"
                     "{classnames |../data/coco.names | class names' file path}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    int dev_id = parser.get<int>("tpuid");
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    std::string image_file = parser.get<std::string>("input");
    // Load bmodel
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    YoloV5 yolo(bm_ctx);
    std::string coco_names = parser.get<std::string>("classnames");
    CV_Assert(0 == yolo.Init(coco_names));
    int run_count = parser.get<int>("run_count");
    // profiling
    TimeStamp ts;
    std::cout<<"input image: "<<image_file<<std::endl;
    cv::Mat img = cv::imread(image_file);
    for(int i=0; i<run_count; i++){
	    run_once(img, yolo, ts);
    }
    //time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    //ts.calbr_basetime(base_time);
    ts.build_timeline("YoloV5");
    ts.show_summary("YoloV5 Demo");
    ts.save_to_file("yolov5.txt");
    ts.clear();
}
