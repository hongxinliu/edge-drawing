/**
 * @brief: core of ED
 * @author: hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date: Mar. 05, 2018
 */

#ifndef _ED_HPP
#define _ED_HPP

#include <opencv2/opencv.hpp>

/**
 * @brief: direction of edge
 * @brief: if |Gx|>|Gy|, it is a proposal of vertical edge(EDGE_VER), or it is horizontal edge(EDGE_HOR)
 */ 
enum EDGE_DIR 
{
	EDGE_HOR, 
	EDGE_VER 
};

/**
 * @brief: a sign for recording status of each pixel in tracing
 */
enum STATUS
{
    STATUS_UNKNOWN = 0, 
    STATUS_BACKGROUND = 1, 
    STATUS_EDGE = 255
};

/**
 * @brief: Trace direction
 */
enum TRACE_DIR
{	
	TRACE_LEFT,
	TRACE_RIGHT,
	TRACE_UP,
	TRACE_DOWN
};

/**
 * @brief: wrapper of edge drawing functions
 * @brief: design all functions to static feature so it is not necessary to create an object of ED
 */
class ED
{
public:
	/**
	 * @brief: detect edges from an image
	 * @param: image [in] image to be processed
	 * @param: edges [out] detected edges, which is a vector of list of cv::Point, each list represents an edge
	 * @param: proposal_thresh [in] gradient blow this thresh should not be proposal of edge pixel
	 * @param: anchor_interval [in] the interval of rows and cols in searching anchors
	 * @param: anchor_thresh [in] the threshold to decision whether a pixel is an anchor
	 * @return: the number of detected edges
	 */
	static int detectEdges(const cv::Mat &image, 
						   std::vector<std::list<cv::Point>> &edges, 
						   const int proposal_thresh = 36, 
						   const int anchor_interval = 4, 
						   const int anchor_thresh = 8);

private:
	/**
	 * @brief: calculate gradient magnitude and orientation
	 * @param: gray [in] input grayscale image
	 * @param: M [out] gradient magnitude, actually |Gx|+|Gy|
	 * @param: O [out] gradient orientation, refer to the definition of EDGE_DIR
	 */
	static void getGradient(const cv::Mat &gray, 
							cv::Mat &M, 
							cv::Mat &O);

	/**
	 * @brief: get anchors
	 * @param: M [in] gradient magnitude
	 * @param: O [in] gradient orientation
	 * @param: proposal_thresh [in] see above
	 * @param: anchor_interval [in] see above
	 * @param: anchor_thresh [in] see above
	 * @param: anchors [out] anchors
	 */
	static void getAnchors(const cv::Mat &M, 
						   const cv::Mat &O, 
						   const int proposal_thresh, 
						   const int anchor_interval, 
						   const int anchor_thresh, 
						   std::vector<cv::Point> &anchors);

	/**
	 * @brief: trace edge from an anchor
	 * @param: M [in] gradient magnitude
	 * @param: O [in] gradient orientation
	 * @param: anchor [in] anchor point to be traced from
	 * @param: status [in|out] status record of each pixel, see the definition of STATUS
	 * @param: edges [out] traced edge would be push_back to
	 */
	static void traceFromAnchor(const cv::Mat &M, 
								const cv::Mat &O, 
								const int proposal_thresh, 
								const cv::Point &anchor, 
								cv::Mat &status, 
								std::vector<std::list<cv::Point>> &edges);

	/**
	 * @brief: main loop of tracing edge
	 * @param: M [in] gradient magnitude
	 * @param: O [in] gradient orientation
	 * @param: pt_last [in] last point
	 * @param: pt_cur [in] current point to be evaluated
	 * @param: dir_last [in] last trace direction
	 * @param: push_back [in] push the traced point to the back or front of list
	 * @param: status [in|out] status record
	 * @param: edge [out] traced edge
	 */
	static void trace(const cv::Mat &M, 
					  const cv::Mat &O, 
					  const int proposal_thresh, 
					  cv::Point pt_last, 
					  cv::Point pt_cur, 
					  TRACE_DIR dir_last, 
					  bool push_back, 
					  cv::Mat &status, 
					  std::list<cv::Point> &edge);
};

#endif
