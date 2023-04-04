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
    img.resize(3468,3468);
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
  BFMatcher matcher;
  vector<DMatch>matches;
  vector<Mat>trian(1,old_image.dest);
  matcher.add(trian);
  matcher.train();

  const float minRatio = 0.7;

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
  cout<<"筛选好之后的匹配特征点数目为"<<goodfeatur.size()<<endl;
  assert(("筛选好之后的匹配特征点数目小于8对",goodfeatur.size()>=0));

  for(int i =0;i<goodfeatur.size();i++)
  {
    maPoint_new.push_back(new_image.key_point[goodfeatur[i].queryIdx].pt);
    maPoint_old.push_back(old_image.key_point[goodfeatur[i].trainIdx].pt);
  }

  maPoint.push_back(maPoint_new);
  maPoint.push_back(maPoint_old);
  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, goodfeatur,result_img);
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
