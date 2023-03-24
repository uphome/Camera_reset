#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
using namespace ::std;
using namespace ::cv;
void matching(string address)
{
  Mat image= imread(address);
  Mat image_key;
  cv::namedWindow("camera", CV_WINDOW_NORMAL);
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  Mat dest;
  orb->detectAndCompute(image, Mat(), key_point, dest);   //寻找特征点
  for(int i =0;i<key_point.size();i++) {

    KeyPoint ima=key_point[i] ;
    cout<<ima.pt<<endl;
    cout<<ima.octave<<endl;//图像金字塔的层数
    cout<<ima.angle<<endl;//特征点的方向
    cout<<ima.class_id<<endl;
    cout<<"--"<<endl;

  }
  drawKeypoints(image, key_point, image_key, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  imshow("camera",image_key);


  waitKey();
  return;
}


int main()
{
  string image_address_new="/home/hu/CLionProjects/camera/image_test/old.jpg";
  string image_address_old="/home/hu/CLionProjects/camera/image_test/new.jpg";
  matching(image_address_new);
  matching(image_address_old);
}
