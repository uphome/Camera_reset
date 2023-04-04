#include "/home/hu/Libraries/cvui-2.7.0/cvui.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include "opencv2/viz.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
using namespace ::std;
using namespace ::cv;
using namespace ::Eigen;

struct key_img
{
  Mat image; // 原图
  vector<KeyPoint> key_point; //特征点
  Mat image_key; //画好特征点的图片
  Mat dest; // 描述点

};

void intercept(VideoCapture capture)
{
  Mat img;
  string  filename;
  while (1)
  {
    capture>>img;
    //img.resize(3468,3468);
    img.resize(480,640);
    namedWindow("choose",WINDOW_NORMAL);
    resizeWindow("choose",500,500);
    imshow("choose",img);
    char key = (char)waitKey(10);
    if (key == 32)
    {
      filename="/home/hu/CLionProjects/cmera_phone/img_phone/img_phone.jpg";
      imwrite(filename, img);
      cout<<"保存图片 位于"<<filename<<endl;
      break;
    }

  }
}

void feature_points(Mat met,struct key_img &fe_point,string tag)
{
  fe_point.image= met;
  Mat image_key;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);
  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);



  /*namedWindow("features_window"+tag, CV_WINDOW_NORMAL);
  resizeWindow("features_window"+tag,500,500);
  imshow("features_window"+tag,fe_point.image_key);
  waitKey(1);*/
}

vector<vector<cv::Point2f>> matching(struct key_img new_image, struct key_img old_image)
{
  vector<vector<Point2f>> maPoint;
  vector<Point2f>maPoint_old,maPoint_new;
  vector<DMatch> matches;
  BFMatcher bf_matcher(NORM_HAMMING);
  bf_matcher.match(new_image.dest,old_image.dest,matches);

  // 匹配对筛选
  double min_dist = 1000, max_dist = 0;
  // 找出所有匹配之间的最大值和最小值
  for (int i = 0; i < new_image.dest.rows; i++)
  {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  // 当描述子之间的匹配大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
  // 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
  vector<DMatch> good_matches;
  for (int i = 0; i < new_image.dest.rows; i++)
  {
    if (matches[i].distance <= max(2 * min_dist, 30.0))
      good_matches.push_back(matches[i]);
  }








/*
  BFMatcher matcher;
  vector<DMatch>matches;
  vector<Mat>trian(1,old_image.dest);
  matcher.add(trian);
  matcher.train();

  const float minRatio = 0.7;
maPoint_old
  vector<vector<DMatch>>matchpoints;
  matcher.knnMatch(new_image.dest,matchpoints,2);
  vector<DMatch>goodfeatur;

  for (int i =0;i<matchpoints.size();i++)
  {
    if (matchpoints[i][0].distance/matchpoints[i][1].distance<=minRatio)
    {
      goodfeatur.push_back(matchpoints[i][0]);
    }

  }
  cout<<"匹配特征点的筛选比值为"<<minRatio<<endl;
*/
  cout<<"筛选好之后的匹配特征点数目为"<<good_matches.size()<<endl;
  assert(("筛选好之后的匹配特征点数目小于8对",good_matches.size()>=0));

  for(int i =0;i<good_matches.size();i++)
  {
    maPoint_new.push_back(new_image.key_point[good_matches[i].queryIdx].pt);
    maPoint_old.push_back(old_image.key_point[good_matches[i].trainIdx].pt);
  }

  maPoint.push_back(maPoint_new);
  maPoint.push_back(maPoint_old);
  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, good_matches,result_img);
  drawKeypoints(old_image.image,old_image.key_point,old_image.image);
  drawKeypoints(new_image.image,new_image.key_point,new_image.image);


  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",1000,500);
  cv::imshow("匹配结果",result_img);
  waitKey(1);
  return maPoint;
}

int main() {
  key_img sou_image, tar_image;
  vector<vector<Point2f>> mPoint;


  Mat frame;
  VideoCapture capture;
  capture.open("http://admin:123456@192.168.73.14:8081");
  Mat capture_img;
  Mat tar_img;
  tar_img=imread("/home/hu/CLionProjects/cmera_phone/img_phone/img_phone.jpg");

  //截图
  intercept(capture);

  //开始处理图片
  while (1)
  {
    capture>>capture_img;
    feature_points(capture_img,sou_image,"sou");
    feature_points(tar_img,tar_image,"tar");

    mPoint = matching(sou_image, tar_image);

  }





}
