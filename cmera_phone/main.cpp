#include "/home/hu/Libraries/cvui-2.7.0/cvui.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include "opencv2/viz.hpp"
#include <Eigen/Core>
#include<opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <random>
#include "Compute_RT.h"
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
   // img.resize(3468,3468);
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
      destroyAllWindows();
      break;
    }

  }
}

void feature_points(Mat met,struct key_img &fe_point,string tag)
{
  fe_point.image= met;
  Mat image_key;
  Mat imamg_result;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);
  drawKeypoints(fe_point.image, fe_point.key_point, imamg_result, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

  /*namedWindow("features_window"+tag, CV_WINDOW_NORMAL);
  resizeWindow("features_window"+tag,500,500);
  imshow("features_window"+tag,fe_point.image_key);
  waitKey(1);*/
}

vector<DMatch> matching(struct key_img new_image, struct key_img old_image)
{
  vector<vector<Point2f>> maPoint;
  vector<Point2f>maPoint_old,maPoint_new;
  vector<DMatch> matches;
  BFMatcher bf_matcher(NORM_HAMMING);
  bf_matcher.match(new_image.dest,old_image.dest,matches);

  double min_dist = 1000, max_dist = 0;
  // 找出所有匹配之间的最大值和最小值
  for (int i = 0; i < new_image.dest.rows; i++)
  {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  vector<DMatch> good_matches;
  for (int i = 0; i < new_image.dest.rows; i++)
  {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) // 可调整筛选条件
      good_matches.push_back(matches[i]);
  }
  cout<<"筛选好之后的匹配特征点数目为"<<good_matches.size()<<endl;
  assert(("筛选好之后的匹配特征点数目小于8对",good_matches.size()>=8));

  for(int i =0;i<good_matches.size();i++)
  {
    maPoint_new.push_back(new_image.key_point[good_matches[i].queryIdx].pt);
    maPoint_old.push_back(old_image.key_point[good_matches[i].trainIdx].pt);
  }

  maPoint.push_back(maPoint_new);
  maPoint.push_back(maPoint_old);
  Mat result_img;
  Mat result_new;
  Mat result_old;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, good_matches,result_img);
  //drawKeypoints(old_image.image,old_image.key_point,result_old);
  //drawKeypoints(new_image.image,new_image.key_point,result_new);


  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",1000,500);
  cv::imshow("匹配结果",result_img);
  waitKey(1);

  return good_matches;
}

vector<Mat> calmatrix(vector<vector<cv::Point2f>> mpoint)
{ Mat R,t;
  Point2f pricipal(323.1992,240.1797);
  float focal=477.7987;
  Mat ess_matrix;
  ess_matrix =findEssentialMat(mpoint[0],mpoint[1],focal,pricipal);
  recoverPose(ess_matrix,mpoint[0],mpoint[1],R,t,focal,pricipal);
  vector<Mat>Pose;
  Pose.push_back(R);
  Pose.push_back(t);
  cout<<R<<endl;
  return Pose;

}

void drawing_rot(vector<Mat>Pose,struct key_img oimg,struct key_img nimg,viz::Viz3d window)
{
  Vector3d euler;
  //euler=Pose[1].eulerAngles(2,1,0);  // ZYX顺序，即先绕x轴roll,再绕y轴pitch,最后绕z轴yaw,0表示X轴,1表示Y轴,2表示Z轴

  Matx33f K(477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1); // 内参矩阵
  viz::Camera mainCamera(K,Size(640,480)); //初始化

  viz::WCameraPosition camparmo(mainCamera.getFov(),oimg.image,1.0,viz::Color::white()); // 参数设置
  cv::Affine3f camPosition(Mat::eye(3,3,CV_32F),Vec3f(0,0,0));
 // cv::Affine3f camPosition(Matx33f (1,0,0,0,1,0,0,0,1),Vec3f(0,0,0));




  viz::WCameraPosition camparmn(mainCamera.getFov(),nimg.image,1.0,viz::Color::green());
  cv::Affine3d camPosition_2(Pose[0],Pose[1]);
  window.showWidget("oldimage",camparmo,camPosition);
  window.showWidget("newimage",camparmn,camPosition_2);



}

void  watermark(Mat src1,Mat src2)
{
  double alpha = 0.3;
  double beta = 1 - alpha;
  double gamma = 0;
  Mat dst;

  addWeighted(src1, alpha, src2, beta, gamma, dst, -1);
  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  cv::imshow("匹配结果",dst);
  waitKey(1);

}


int main() {
  Mat K = (Mat_<double>(3, 3) << 477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1);

  viz::Viz3d window("camere pose");
  key_img sou_image, tar_image;
  vector<vector<Point2f>> mPoint;
  vector<Mat> Pose;
  vector<DMatch> good_matches;

 // 引入摄像头
  Mat frame;
  VideoCapture capture;
 // capture.open("http://admin:123456@192.168.73.14:8081");

  capture.open("/home/hu/下载/VID20230425164019.mp4");
  Mat capture_img;
  Mat tar_img;

  //截图
  intercept(capture);
  tar_img=imread("/home/hu/CLionProjects/cmera_phone/img_phone/img_phone.jpg");


  Compute_RT camera;
  float msigm=1;
  int mMaxIterations=200;
  K.convertTo(K,CV_32F);


  //开始处理图片
  while (1)
  {


    capture>>capture_img;

    feature_points(capture_img,sou_image,"sou");
    feature_points(tar_img,tar_image,"tar");
    good_matches = matching(sou_image, tar_image);

    camera.set_number(msigm,mMaxIterations,K,good_matches,sou_image.key_point,tar_image.key_point);
    Mat R,T;
    vector<cv::Point3f> vP3D;
    vector<bool> vbTriangulated;
    bool Hu;

    Hu=camera.Initialize(R,T,vP3D,vbTriangulated);
    cout<<Hu<<endl;
    if(Hu)
    {
      cout<<R<<endl;
    }

    //watermark(capture_img,tar_img);
    //window.spinOnce();



  }





}
