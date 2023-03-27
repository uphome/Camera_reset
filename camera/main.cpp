#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
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

void feature_points(Mat met,struct key_img &fe_point)
{
  fe_point.image= met;
  Mat image_key;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);   //寻找特征点
 /* for(int i =0;i<fe_point.key_point.size();i++) {

    KeyPoint ima=fe_point.key_point[i] ;
    cout<<ima.pt<<endl;
    cout<<ima.octave<<endl;//图像金字塔的层数
    cout<<ima.angle<<endl;//特征点的方向
    cout<<ima.class_id<<endl;
    cout<<"--"<<endl;

  }*/
  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
/*  cv::namedWindow("camera", CV_WINDOW_NORMAL);
  imshow("camera",fe_point.image_key);
  waitKey();*/

}

vector<vector<cv::Point2f>> matching(struct key_img new_image, struct key_img old_image)
{
  vector<vector<Point2f>> maPoint;
  vector<Point2f>maPoint_old,maPoint_new;
  BFMatcher matcher;
  Ptr<DescriptorMatcher> matches = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
  //vector<DMatch>matches;
  vector<Mat>trian(1,old_image.dest);
  matcher.add(trian);
  matcher.train();

  vector<vector<DMatch>>matchpoints;
  matcher.knnMatch(new_image.dest,matchpoints,2);
  vector<DMatch>goodfeatur;
  for (int i =0;i<matchpoints.size();i++)
  {
    if (matchpoints[i][0].distance<0.7*matchpoints[i][1].distance)
    {
      goodfeatur.push_back(matchpoints[i][0]);

    }
  }

  for(int i =0;i<goodfeatur.size();i++)
  {
    maPoint_new.push_back(new_image.key_point[goodfeatur[i].queryIdx].pt);
    maPoint_old.push_back(old_image.key_point[goodfeatur[i].trainIdx].pt);
  }
  maPoint.push_back(maPoint_new);
  maPoint.push_back(maPoint_old);
  cout<<goodfeatur.size()<<endl;
  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, goodfeatur,result_img);
  drawKeypoints(old_image.image,old_image.key_point,old_image.image);
  drawKeypoints(new_image.image,new_image.key_point,new_image.image);


/*
  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  imshow("匹配结果",result_img);

  waitKey(0);
  system("pause");
*/

  return maPoint;
}

vector<Mat> calmatrix(vector<vector<cv::Point2f>> mpoint)
{ Mat R,t;
  Point2f pricipal(323.1992,240.1797);
  double focal=477.7987;
  Mat ess_matrix;
  ess_matrix =findEssentialMat(mpoint[0],mpoint[1],focal,pricipal);
  /*cout<<"本质矩阵为"<<endl;
  cout<<ess_matrix<<endl;*/

/*  Mat homo_matrix;
  homo_matrix= findHomography(mpoint[0],mpoint[1],RANSAC,3);
  cout<<"homo"<<endl;
  cout<<homo_matrix<<endl;*/

  recoverPose(ess_matrix,mpoint[0],mpoint[1],R,t,focal,pricipal);
  vector<Mat>Pose;
  Pose.push_back(R);
  Pose.push_back(t);

  return Pose;

}

void drawing(MatrixXf result,struct key_img img)
{
  MatrixXf test(4,4);
  test<<0,1,0,1,0,0,1,1,0,0,0,0,1,1,1,1;
  Mat resut;
  namedWindow("最终结果",WINDOW_NORMAL);
resizeWindow("最终结果",500,500);
  line(img.image,Point(result(0,0)*100+img.image.cols/2,result(1,0)*100+img.image.rows/2),Point(result(0,1)*100+img.image.cols/2,result(1,1)*100+img.image.rows/2),Scalar(0, 255, 0),10);
  line(img.image,Point(result(0,1)*100+img.image.cols/2,result(1,1)*100+img.image.rows/2),Point(result(0,2)*100+img.image.cols/2,result(1,2)*100+img.image.rows/2),Scalar(0, 255, 0),10);
  line(img.image,Point(result(0,2)*100+img.image.cols/2,result(1,2)*100+img.image.rows/2),Point(result(0,3)*100+img.image.cols/2,result(1,3)*100+img.image.rows/2),Scalar(0, 255, 255),10);
  line(img.image,Point(result(0,3)*100+img.image.cols/2,result(1,3)*100+img.image.rows/2),Point(result(0,0)*100+img.image.cols/2,result(1,0)*100+img.image.rows/2),Scalar(0, 255, 255),10);
  /*cout<<result(0,0)<<"    "<<result(1,0)<<endl;
  cout<<result(0,1)<<"    "<<result(1,1)<<endl;
  cout<<result(0,2)<<"    "<<result(1,2)<<endl;
  cout<<result(0,3)<<"    "<<result(1,3)<<endl;*/
  //rectangle(img.image, Point(img.image.cols/2,img.image.rows/2), Point(img.image.cols/2+100,img.image.rows/2+100), Scalar(0, 0, 255),10);
 imshow("最终结果",img.image);


waitKey(0);
system("pause");

}

int main() {
  MatrixXf transfrom(4, 4);
  key_img oimage, nimage;
  vector<vector<Point2f>> mPoint;// [0] 为new [1] 为old；
  vector<Mat> Pose;
  MatrixXf test(4, 4);
  test << 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;
  transfrom.setZero();
  MatrixXf result(4, 4);

  Mat frame;
  VideoCapture capture;
  capture.open("http://admin:123456@192.168.1.105:8081");


    Mat K = (Mat_<double>(3, 3) << 477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1);
    Mat image_address_new = imread( "/home/hu/CLionProjects/camera/image_test/old.jpg");
    Mat image_address_old = imread( "/home/hu/CLionProjects/camera/image_test/old.jpg");
    feature_points(image_address_new, nimage);
    feature_points(image_address_old, oimage);
    mPoint = matching(nimage, oimage);
    Pose = calmatrix(mPoint);
    cout << "R 为" << endl;
    cout << Pose[0] << endl;
    cout << test << endl;
    transfrom.block<3, 3>(0, 0) << Pose[0].at<double>(0, 0), Pose[0].at<double>(0, 1), Pose[0].at<double>(0, 2),
        Pose[0].at<double>(1, 0), Pose[0].at<double>(1, 1), Pose[0].at<double>(1, 2),
        Pose[0].at<double>(2, 0), Pose[0].at<double>(2, 1), Pose[0].at<double>(2, 2);
    transfrom.col(3) << Pose[1].at<double>(0, 0), Pose[1].at<double>(1, 0), Pose[1].at<double>(2, 0), 1;
    cout << test * transfrom << endl;
    result = transfrom * test;
    drawing(result, oimage);
    cout << result << endl;
    waitKey(1);
  }
