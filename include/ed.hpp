/**
 * @brief interface of ed
 * @author hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date Jul. 15, 2018
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <list>

namespace ed {

/**
 * @brief interface to perform edge detection
 * @param image [in] image to be processed
 * @param proposal_thresh [in] minimum gradient magnitude to be an edge point
 * @param anchor_interval [in] interval to search anchors
 * @param anchor_thresh [in] minimum gradient diff of anchors
 */
std::vector<std::list<cv::Point>> detectEdges(const cv::Mat &image, 
                                              const int proposal_thresh = 36, 
                                              const int anchor_interval = 4, 
                                              const int anchor_thresh = 8);
}
