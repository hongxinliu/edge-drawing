/**
 * @brief: demo for ed
 * @author: hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date: Jul. 15, 2018
 */

#include "include/ed.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    // check input
	if(argc != 2)
    {
        std::cout<<"Usage: "<<argv[0]<<" IMAGE_PATH"<<std::endl;
        return -1;
    }

    // open image
    cv::Mat image = cv::imread(argv[1]);
    if(image.empty())
    {
        std::cout<<"Cannot open image file "<<argv[1]<<std::endl;
        return -2;
    }

    // detect edges
    ed::ED ed;
	double t = cv::getTickCount();
    auto edges = ed.detectEdges(image);
	std::cout << "Detect edges in " << (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms" << std::endl;

    // show output
    cv::imshow("image", image);
    cv::Mat out(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    for(const auto &edge : edges)
    {
        for(const auto &pt : edge)
        {
            out.at<uchar>(pt.y, pt.x) = 255;
            cv::imshow("draw", out);
            if(cv::waitKey(1) == 27)
            {
                cv::destroyAllWindows();
                exit(1);
            }
        }
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

	return 0;
}
