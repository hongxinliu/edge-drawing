/**
 * @brief: interface of ed
 * @author: hongxinliu <github.com/hongxinliu> <hongxinliu.com>
 * @date: Jul. 15, 2018
 */

#include "include/ed.hpp"
#include <iostream>

namespace ed {

#define GAUSS_SIZE	(5)
#define GAUSS_SIGMA	(1.0)
#define SOBEL_ORDER	(1)
#define SOBEL_SIZE	(3)

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

class ED_Internal
{
public:
    std::vector<std::list<cv::Point>> detectEdges(const cv::Mat &image, 
						                          const int proposal_thresh, 
                                                  const int anchor_interval, 
                                                  const int anchor_thresh);

private:
    /**
	 * @brief: calculate gradient magnitude and orientation
	 * @param: gray [in] input grayscale image
	 * @param: M [out] gradient magnitude, actually |Gx|+|Gy|
	 * @param: O [out] gradient orientation, refer to the definition of EDGE_DIR
	 */
	void getGradient(const cv::Mat &gray, 
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
	void getAnchors(const cv::Mat &M, 
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
	void traceFromAnchor(const cv::Mat &M, 
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
	void trace(const cv::Mat &M, 
               const cv::Mat &O, 
               const int proposal_thresh, 
               cv::Point pt_last, 
               cv::Point pt_cur, 
               TRACE_DIR dir_last, 
               bool push_back, 
               cv::Mat &status, 
               std::list<cv::Point> &edge);
};

std::vector<std::list<cv::Point>> ED_Internal::detectEdges(const cv::Mat &image, 
                                                           const int proposal_thresh, 
                                                           const int anchor_interval, 
                                                           const int anchor_thresh)
{
	// 0.preparation
    cv::Mat gray;
    if(image.empty())
    {
        std::cout<<"Empty image input!"<<std::endl;
        return std::vector<std::list<cv::Point>>();
    }
    if(image.type() == CV_8UC1)
        gray = image.clone();
    else if(image.type() == CV_8UC3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else
    {
        std::cout<<"Unknow image type!"<<std::endl;
        return std::vector<std::list<cv::Point>>();
    }

    // 1.Gauss blur
    cv::GaussianBlur(gray, gray, cv::Size(GAUSS_SIZE, GAUSS_SIZE), GAUSS_SIGMA, GAUSS_SIGMA);

    // 2.get gradient magnitude and orientation
    cv::Mat M, O;
    getGradient(gray, M, O);

    // 3.get anchors
    std::vector<cv::Point> anchors;
    getAnchors(M, O, proposal_thresh, anchor_interval, anchor_thresh, anchors);

    // 4.trace edges from anchors
    cv::Mat status(gray.rows, gray.cols, CV_8UC1, cv::Scalar(STATUS_UNKNOWN)); //Init all status to STATUS_UNKNOWN
    std::vector<std::list<cv::Point>> edges;
    for(const auto &anchor : anchors)
        traceFromAnchor(M, O, proposal_thresh, anchor, status, edges);
    
    return edges;
}

void ED_Internal::getGradient(const cv::Mat &gray, 
					          cv::Mat &M, 
                              cv::Mat &O)
{
	cv::Mat Gx, Gy;
    cv::Sobel(gray, Gx, CV_16SC1, SOBEL_ORDER, 0, SOBEL_SIZE);
    cv::Sobel(gray, Gy, CV_16SC1, 0, SOBEL_ORDER, SOBEL_SIZE);
    
    M.create(gray.rows, gray.cols, CV_16SC1);
    O.create(gray.rows, gray.cols, CV_8UC1);

    for(int r=0; r<gray.rows; ++r)
    {
        for(int c=0; c<gray.cols; ++c)
        {
            short dx = abs(Gx.at<short>(r, c));
            short dy = abs(Gy.at<short>(r, c));

            M.at<short>(r, c) = dx + dy;
            O.at<uchar>(r, c) = (dx > dy ? EDGE_VER : EDGE_HOR);
        }
    }
}

void ED_Internal::getAnchors(const cv::Mat &M, 
					         const cv::Mat &O, 
                             const int proposal_thresh, 
                             const int anchor_interval, 
                             const int anchor_thresh, 
                             std::vector<cv::Point> &anchors)
{
	anchors.clear();

    for(int r = 1; r < M.rows - 1; r += anchor_interval)
    {   
        for(int c = 1; c < M.cols - 1; c += anchor_interval)
        {
            // ignore non-proposal pixels
            if(M.at<short>(r, c) < proposal_thresh)
                continue;
            
            // horizontal edge
            if(O.at<uchar>(r, c) == EDGE_HOR)
            {
                if(M.at<short>(r, c) - M.at<short>(r-1, c) >= anchor_thresh &&
                   M.at<short>(r, c) - M.at<short>(r+1, c) >= anchor_thresh)
                   anchors.emplace_back(c, r);
            }

            // vertical edge
            else
            {
                if(M.at<short>(r, c) - M.at<short>(r, c-1) >= anchor_thresh &&
                   M.at<short>(r, c) - M.at<short>(r, c+1) >= anchor_thresh)
                   anchors.emplace_back(c, r);
            }
        }
    }
}

void ED_Internal::traceFromAnchor(const cv::Mat &M, 
						          const cv::Mat &O, 
                                  const int proposal_thresh, 
                                  const cv::Point &anchor, 
                                  cv::Mat &status, 
                                  std::vector<std::list<cv::Point>> &edges)
{
	// if this anchor point has already been visited
    if(status.at<uchar>(anchor.y, anchor.x) != STATUS_UNKNOWN)
        return;
    
    std::list<cv::Point> edge;
    cv::Point pt_last;
    TRACE_DIR dir_last;

    // if horizontal edge, go left and right
    if(O.at<uchar>(anchor.y, anchor.x) == EDGE_HOR)
    {
        // go left first
		// sssume the last visited point is the right hand side point and TRACE_LEFT to current point, the same below
		pt_last = cv::Point(anchor.x + 1, anchor.y);
		dir_last = TRACE_LEFT;
        trace(M, O, proposal_thresh, pt_last, anchor, dir_last, false, status, edge);
        
        // reset anchor point
		// it has already been set in the previous traceEdge(), reset it to satisfy the initial while condition, the same below */
        status.at<uchar>(anchor.y, anchor.x) = STATUS_UNKNOWN;
        
        // go right then
		pt_last = cv::Point(anchor.x - 1, anchor.y);
		dir_last = TRACE_RIGHT;
		trace(M, O, proposal_thresh, pt_last, anchor, dir_last, true, status, edge);
    }

    // vertical edge, go up and down
    else
    {
        // go up first
		pt_last = cv::Point(anchor.x, anchor.y + 1);
		dir_last = TRACE_UP;
		trace(M, O, proposal_thresh, pt_last, anchor, dir_last, false, status, edge);

		// reset anchor point
		status.at<uchar>(anchor.y, anchor.x) = STATUS_UNKNOWN;

		// go down then
		pt_last = cv::Point(anchor.x, anchor.y - 1);
		dir_last = TRACE_DOWN;
		trace(M, O, proposal_thresh, pt_last, anchor, dir_last, true, status, edge);
    }

    edges.push_back(edge);
}

void ED_Internal::trace(const cv::Mat &M, 
			            const cv::Mat &O, 
                        const int proposal_thresh, 
			            cv::Point pt_last, 
			            cv::Point pt_cur, 
			            TRACE_DIR dir_last, 
			            bool push_back, 
			            cv::Mat &status, 
			            std::list<cv::Point> &edge)
{
	// current direction
    TRACE_DIR dir_cur;
    
    // repeat until reaches the visited pixel or non-proposal
	while (true)
	{   
        // terminate trace if that point has already been visited
        if(status.at<uchar>(pt_cur.y, pt_cur.x) != STATUS_UNKNOWN)
            break;

        // set it to background and terminate trace if that point is not a proposal edge
        if(M.at<short>(pt_cur.y, pt_cur.x) < proposal_thresh)
        {
            status.at<uchar>(pt_cur.y, pt_cur.x) = STATUS_BACKGROUND;
            break;
        }

        // set point pt_cur as edge
        status.at<uchar>(pt_cur.y, pt_cur.x) = STATUS_EDGE;
        if (push_back)
			edge.push_back(pt_cur);
		else
			edge.push_front(pt_cur);
        
        // if its direction is EDGE_HOR, trace left or right
		if (O.at<uchar>(pt_cur.y, pt_cur.x) == EDGE_HOR)
		{
            // calculate trace direction
			if (dir_last == TRACE_UP || dir_last == TRACE_DOWN)
			{
				if (pt_cur.x < pt_last.x)
					dir_cur = TRACE_LEFT;
				else
					dir_cur = TRACE_RIGHT;
			}
			else
				dir_cur = dir_last;

            // update last state
			pt_last = pt_cur;
            dir_last = dir_cur;
            
            // go left
			if (dir_cur == TRACE_LEFT)
			{
				auto leftTop = M.at<short>(pt_cur.y - 1, pt_cur.x - 1);
				auto left = M.at<short>(pt_cur.y, pt_cur.x - 1);
				auto leftBottom = M.at<short>(pt_cur.y + 1, pt_cur.x - 1);

				if (leftTop >= left && leftTop >= leftBottom)
					pt_cur = cv::Point(pt_cur.x - 1, pt_cur.y - 1);
				else if (leftBottom >= left && leftBottom >= leftTop)
					pt_cur = cv::Point(pt_cur.x - 1, pt_cur.y + 1);
				else
					pt_cur.x -= 1;

				// break if reaches the border of image, the same below
				if (pt_cur.x == 0 || pt_cur.y == 0 || pt_cur.y == M.rows - 1)
					break;
            }
            
            // go right
			else
			{
				auto rightTop = M.at<short>(pt_cur.y - 1, pt_cur.x + 1);
				auto right = M.at<short>(pt_cur.y, pt_cur.x + 1);
				auto rightBottom = M.at<short>(pt_cur.y + 1, pt_cur.x + 1);

				if (rightTop >= right && rightTop >= rightBottom)
					pt_cur = cv::Point(pt_cur.x + 1, pt_cur.y - 1);
				else if (rightBottom >= right && rightBottom >= rightTop)
					pt_cur = cv::Point(pt_cur.x + 1, pt_cur.y + 1);
				else
					pt_cur.x += 1;

				if (pt_cur.x == M.cols - 1 || pt_cur.y == 0 || pt_cur.y == M.rows - 1)
					break;
			}
        }

        // its direction is EDGE_VER, trace up or down
        else
        {
            // calculate trace direction
			if (dir_last == TRACE_LEFT || dir_last == TRACE_RIGHT)
			{
				if (pt_cur.y < pt_last.y)
					dir_cur = TRACE_UP;
				else
					dir_cur = TRACE_DOWN;
			}
			else
				dir_cur = dir_last;

			// update last state
			pt_last = pt_cur;
			dir_last = dir_cur;

			// go up
			if (dir_cur == TRACE_UP)
			{
				auto leftTop = M.at<short>(pt_cur.y - 1, pt_cur.x - 1);
				auto top = M.at<short>(pt_cur.y - 1, pt_cur.x);
				auto rightTop = M.at<short>(pt_cur.y - 1, pt_cur.x + 1);

				if (leftTop >= top && leftTop >= rightTop)
					pt_cur = cv::Point(pt_cur.x - 1, pt_cur.y - 1);
				else if (rightTop >= top && rightTop >= leftTop)
					pt_cur = cv::Point(pt_cur.x + 1, pt_cur.y - 1);
				else
					pt_cur.y -= 1;

				if (pt_cur.y == 0 || pt_cur.x == 0 || pt_cur.x == M.cols - 1)
					break;
			}

			// go down
			else
			{
				auto leftBottom = M.at<short>(pt_cur.y + 1, pt_cur.x - 1);
				auto bottom = M.at<short>(pt_cur.y + 1, pt_cur.x);
				auto rightBottom = M.at<short>(pt_cur.y + 1, pt_cur.x + 1);

				if (leftBottom >= bottom && leftBottom >= rightBottom)
					pt_cur = cv::Point(pt_cur.x - 1, pt_cur.y + 1);
				else if (rightBottom >= bottom && rightBottom >= leftBottom)
					pt_cur = cv::Point(pt_cur.x + 1, pt_cur.y + 1);
				else
					pt_cur.y += 1;

				if (pt_cur.y == M.rows - 1 || pt_cur.x == 0 || pt_cur.x == M.cols - 1)
					break;
			}
        }
    }
}

ED::ED() : 
ed_(new ED_Internal())
{
}

ED::~ED()
{
    delete ed_;
}

std::vector<std::list<cv::Point>> ED::detectEdges(const cv::Mat &image, 
						                          const int proposal_thresh, 
                                                  const int anchor_interval, 
                                                  const int anchor_thresh)
{
    return ed_->detectEdges(image, proposal_thresh, anchor_interval, anchor_thresh);
}

}
