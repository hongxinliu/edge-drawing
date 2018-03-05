/**
 * @brief: demo of ED
 * @author: hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date: Mar. 05, 2018
 */

#include "ED/ED.hpp"

int main(int argc, char **argv)
{
    // check input
	if(argc != 2)
    {
        std::cout<<"Usage: "<<argv[0]<<" IMAGE_PATH"<<std::endl;
        return -1;
    }

    // open image
    cv::Mat img = cv::imread(argv[1]);
    if(img.empty())
    {
        std::cout<<"Cannot open image file "<<argv[1]<<std::endl;
        return -2;
    }

    // detect edges
    std::vector<std::list<cv::Point>> edges;
	double t = cv::getTickCount();
    ED::detectEdges(img, edges);
	std::cout << "Detect edges in " << (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms" << std::endl;

    // show output
    cv::imshow("image", img);
    cv::Mat out(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
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
