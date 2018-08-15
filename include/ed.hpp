/**
 * @brief: interface of ed
 * @author: hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date: Jul. 15, 2018
 */

#ifndef _ED_ED_HPP
#define _ED_ED_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <memory>

namespace ed {

class ED_Internal;

class ED
{
public:
    ED();

public:
    std::vector<std::list<cv::Point>> detectEdges(const cv::Mat &image, 
                                                  const int proposal_thresh = 36, 
                                                  const int anchor_interval = 4, 
                                                  const int anchor_thresh = 8);

private:
    std::shared_ptr<ED_Internal> ed_;
};

}

#endif
