#define CVUI_IMPLEMENTATION
#include "/home/hu/Libraries/cvui-2.7.0/cvui.h"
#include <random>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <stdlib.h>
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



#define WINDOW_NAME "最终结果"

//TODO：完成旋转角度的可视化。  采用欧拉角
//TODO: 完善计算R，T的方法。
struct key_img
{
  Mat image; // 原图
  vector<KeyPoint> key_point; //特征点
  Mat image_key; //画好特征点的图片
  Mat dest; // 描述点

};


void feature_points(Mat met,struct key_img &fe_point)
{ //met.resize(480,640);
  fe_point.image= met;
  Mat image_key;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);   //寻找特征点

  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

}

vector<DMatch>matching(struct key_img new_image, struct key_img old_image)
{
  vector<vector<Point2f>> maPoint;
  vector<Point2f>maPoint_old,maPoint_new;
  BFMatcher matcher;
  //Ptr<DescriptorMatcher> matches = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
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
  assert(("筛选好之后的匹配特征点数目小于8对",goodfeatur.size()>=8));

  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, goodfeatur,result_img);
  drawKeypoints(old_image.image,old_image.key_point,old_image.image);
  drawKeypoints(new_image.image,new_image.key_point,new_image.image);

  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  cv::imshow("匹配结果",result_img);

  waitKey(0);
 // system("pause");

  return goodfeatur;
}

vector<Mat> calmatrix(vector<vector<cv::Point2f>> mpoint)
{ Mat R,t;
  Point2f pricipal(323.1992,240.1797);
  float focal=477.7987;
  Mat ess_matrix,Fundamental,homo_matrix;
  Fundamental= findFundamentalMat(mpoint[0],mpoint[1],CV_FM_8POINT);
  float SH,SF;


  ess_matrix =findEssentialMat(mpoint[0],mpoint[1],focal,pricipal);
  cout<<"本质矩阵为"<<endl;
  cout<<ess_matrix<<endl;

  homo_matrix= findHomography(mpoint[0],mpoint[1],RANSAC,3);


  recoverPose(ess_matrix,mpoint[0],mpoint[1],R,t,focal,pricipal);
  vector<Mat>Pose;
  Pose.push_back(R);
  Pose.push_back(t);

  return Pose;

}



void drawing_rot(vector<Mat>Pose,struct key_img oimg,struct key_img nimg)
{
  Vector3d euler;
  //euler=Pose[1].eulerAngles(2,1,0);  // ZYX顺序，即先绕x轴roll,再绕y轴pitch,最后绕z轴yaw,0表示X轴,1表示Y轴,2表示Z轴
  viz::Viz3d window("camere pose");
  Matx33f K(477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1); // 内参矩阵
  viz::Camera mainCamera(K,Size(640,480)); //初始化
  cout<<Mat::eye(3,3,CV_32F)<<endl;

  viz::WCameraPosition camparmo(mainCamera.getFov(),oimg.image,1.0,viz::Color::white()); // 参数设置
  cv::Affine3f camPosition(Matx33f (1,0,0,0,1,0,0,0,1),Vec3f(0,0,0));


  viz::WCameraPosition camparmn(mainCamera.getFov(),nimg.image,1.0,viz::Color::green());
  cv::Affine3d camPosition_2(Pose[0],Pose[1]);
  window.showWidget("oldimage",camparmo,camPosition);
  window.showWidget("newimage",camparmn,camPosition_2);
  window.spin();


}


int main() {
  key_img oimage, nimage;
  vector<vector<Point2f>> mPoint;// [0] 为new [1] 为old；
  vector<Mat> Pose;
  Mat frame;
  VideoCapture capture;

    Mat K = (Mat_<double>(3, 3) << 477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1);
    Mat image_address_new = imread( "/home/hu/下载/IMG20230328182935.jpg");
    Mat image_address_old = imread( "/home/hu/下载/IMG20230328182918.jpg");
    feature_points(image_address_new, nimage);
    feature_points(image_address_old, oimage);

  vector<DMatch>goodfeatur;
  goodfeatur = matching(nimage, oimage);


  int mMaxIterations=200; //最大迭代次数
  int N=goodfeatur.size();
  vector<size_t> vAllIndices;
  vAllIndices.reserve(N); //储存特征点索引，并预分配空间
  vector<size_t> vAvailableIndices;//在RANSAC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引

  //初始化所有特征点对的索引，索引值0到N-1
  for(int i=0; i<N; i++)
  {
    vAllIndices.push_back(i);
  }
  //cout << '-----'<< endl;
  vector<vector<size_t>>mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));
  default_random_engine   e;//随机数引擎对象
  e.seed(time(NULL)); //初始化种子

  for(int it=0; it<mMaxIterations; it++)
  {

    vAvailableIndices = vAllIndices;

    // Select a minimum set
    //选择最小的数据样本集，使用八点法求，所以这里就循环了八次
    for(size_t j=0; j<8; j++)
    {
      // 随机产生一对点的id,范围从0到N-1
      uniform_int_distribution<unsigned>  u(0,vAvailableIndices.size()-1);
      int randi = u(e);
      // idx表示哪一个索引对应的特征点对被选中
      int idx = vAvailableIndices[randi];

      //将本次迭代这个选中的第j个特征点对的索引添加到mvSets中
      mvSets[it][j] = idx;


      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();  //删除尾部数据
    }//依次提取出8个特征点对
  }//迭代mMaxIterations次，选取各自迭代时需要用到的最小数据集

  vector<bool> vbMatchesInliersH, vbMatchesInliersF;
  float SH, SF; //score for H and F
  cv::Mat H, F;




}
