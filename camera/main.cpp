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
void Normalize(const vector <cv::KeyPoint> &vKeys, vector <cv::Point2f> &vNormalizedPoints, cv::Mat &T,int N) {
// step1. 计算均值
  float meanX = 0;
  float meanY = 0;
  for (int i = 0; i < N; i++) {
    meanX += vKeys[i].pt.x;
    meanY += vKeys[i].pt.y;
  }
  meanX = meanX / N;
  meanY = meanY / N;

// step2. 计算一阶中心矩
  float meanDevX = 0;
  float meanDevY = 0;
  for (int i = 0; i < N; i++) {
    vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
    vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
    meanDevX += fabs(vNormalizedPoints[i].x);
    meanDevY += fabs(vNormalizedPoints[i].y);
  }
  meanDevX = meanDevX / N;
  meanDevY = meanDevY / N;
  float sX = 1.0 / meanDevX;
  float sY = 1.0 / meanDevY;

// step3. 进行归一化
  for (int i = 0; i < N; i++) {
    vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
    vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
  }

// 记录归一化参数,以便后续恢复尺度
  T = cv::Mat::eye(3, 3, CV_32F);
  T.at<float>(0, 0) = sX;
  T.at<float>(1, 1) = sY;
  T.at<float>(0, 2) = -meanX * sX;
  T.at<float>(1, 2) = -meanY * sY;
}

float  CheckHomography(const cv::Mat &H21, const cv::Mat &H12,vector<bool> &vbMatchesInliers, float sigma,vector<DMatch>goodfeatures,key_img old_image,key_img new_image) {
  const int N = goodfeatures.size();
// 取出单应矩阵H各位上的值
  const float h11 = H21.at<float>(0, 0);
  const float h12 = H21.at<float>(0, 1);
  const float h13 = H21.at<float>(0, 2);
  const float h21 = H21.at<float>(1, 0);
  const float h22 = H21.at<float>(1, 1);
  const float h23 = H21.at<float>(1, 2);
  const float h31 = H21.at<float>(2, 0);
  const float h32 = H21.at<float>(2, 1);
  const float h33 = H21.at<float>(2, 2);

  const float h11inv = H12.at<float>(0, 0);
  const float h12inv = H12.at<float>(0, 1);
  const float h13inv = H12.at<float>(0, 2);
  const float h21inv = H12.at<float>(1, 0);
  const float h22inv = H12.at<float>(1, 1);
  const float h23inv = H12.at<float>(1, 2);
  const float h31inv = H12.at<float>(2, 0);
  const float h32inv = H12.at<float>(2, 1);
  const float h33inv = H12.at<float>(2, 2);
  vbMatchesInliers.resize(N); // 标记是否是内点
  float score = 0; // 置信度得分
  const float th = 5.991; // 自由度为2,显著性水平为0.05的卡方分布对应的临界阈值
  const float invSigmaSquare = 1.0 / (sigma * sigma); // 信息矩阵,方差平方的倒数

// 双向投影,计算加权投影误差
  for (int i = 0; i < N; i++) {
    bool bIn = true;

// step1. 提取特征点对
    const cv::KeyPoint &kp1 = old_image.key_point[goodfeatures[i].queryIdx];
    const cv::KeyPoint &kp2 = new_image.key_point[goodfeatures[i].trainIdx];
    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

// step2. 计算img2到img1的重投影误差
    const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
    const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
    const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;
    const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
    const float chiSquare1 = squareDist1 * invSigmaSquare;

// step3. 离群点标记上,非离群点累加计算得分
    if (chiSquare1 > th)
      bIn = false;
    else
      score += th - chiSquare1;

// step4. 计算img1到img2的重投影误差
    const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
    const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
    const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;
    const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);
    const float chiSquare2 = squareDist2 * invSigmaSquare;

// step5. 离群点标记上,非离群点累加计算得分
    if (chiSquare2 > th)
      bIn = false;
    else
      score += th - chiSquare2;
    if (bIn)
      vbMatchesInliers[i] = true;
    else
      vbMatchesInliers[i] = false;
  }
  return score;
}

float CheckFundamental(
    const cv::Mat &F21,             //当前帧和参考帧之间的基础矩阵
    vector<bool> &vbMatchesInliers, //匹配的特征点对属于inliers的标记
    float sigma,
    vector<DMatch>goodfeatures,
    key_img old_image,
    key_img new_image)                    //方差
{


  // 获取匹配的特征点对的总对数
  const int N = goodfeatures.size();

  // Step 1 提取基础矩阵中的元素数据
  const float f11 = F21.at<float>(0,0);
  const float f12 = F21.at<float>(0,1);
  const float f13 = F21.at<float>(0,2);
  const float f21 = F21.at<float>(1,0);
  const float f22 = F21.at<float>(1,1);
  const float f23 = F21.at<float>(1,2);
  const float f31 = F21.at<float>(2,0);
  const float f32 = F21.at<float>(2,1);
  const float f33 = F21.at<float>(2,2);

  // 预分配空间
  vbMatchesInliers.resize(N);

  // 设置评分初始值（因为后面需要进行这个数值的累计）
  float score = 0;

  // 基于卡方检验计算出的阈值
  // 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
  // ?是因为点到直线距离是一个自由度吗？
  const float th = 3.841;

  // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const float thScore = 5.991;

  // 信息矩阵，或 协方差矩阵的逆矩阵
  const float invSigmaSquare = 1.0/(sigma*sigma);


  // Step 2 计算img1 和 img2 在估计 F 时的score值
  for(int i=0; i<N; i++)
  {
    //默认为这对特征点是Inliers
    bool bIn = true;

    // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
    const cv::KeyPoint &kp1 = old_image.key_point[goodfeatures[i].queryIdx];
    const cv::KeyPoint &kp2 = new_image.key_point[goodfeatures[i].trainIdx];

    // 提取出特征点的坐标
    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    // Reprojection error in second image
    // Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
    const float a2 = f11*u1+f12*v1+f13;
    const float b2 = f21*u1+f22*v1+f23;
    const float c2 = f31*u1+f32*v1+f33;

    // Step 2.3 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const float num2 = a2*u2+b2*v2+c2;
    const float squareDist1 = num2*num2/(a2*a2+b2*b2);
    // 带权重误差
    const float chiSquare1 = squareDist1*invSigmaSquare;

    // Step 2.4 误差大于阈值就说明这个点是Outlier
    // ? 为什么判断阈值用的 th（1自由度），计算得分用的thScore（2自由度）
    // ? 可能是为了和CheckHomography 得分统一？
    if(chiSquare1>th)
      bIn = false;
    else
      // 误差越大，得分越低
      score += thScore - chiSquare1;

    // 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
    const float a1 = f11*u2+f21*v2+f31;
    const float b1 = f12*u2+f22*v2+f32;
    const float c1 = f13*u2+f23*v2+f33;

    // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const float num1 = a1*u1+b1*v1+c1;
    const float squareDist2 = num1*num1/(a1*a1+b1*b1);

    // 带权重误差
    const float chiSquare2 = squareDist2*invSigmaSquare;

    // 误差大于阈值就说明这个点是Outlier
    if(chiSquare2>th)
      bIn = false;
    else
      score += thScore - chiSquare2;

    // Step 2.5 保存结果
    if(bIn)
      vbMatchesInliers[i]=true;
    else
      vbMatchesInliers[i]=false;
  }
  //  返回评分
  return score;
}
Mat ComputeH21(
    const vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
    const vector<cv::Point2f> &vP2) //归一化后的点, in current frame
{
  // 基本原理：见附件推导过程：
  // |x'|     | h1 h2 h3 ||x|
  // |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
  // |1 |     | h7 h8 h9 ||1|
  // 使用DLT(direct linear tranform)求解该模型
  // x' = a H x
  // ---> (x') 叉乘 (H x)  = 0  (因为方向相同) (取前两行就可以推导出下面的了)
  // ---> Ah = 0
  // A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
  //     |-x -y -1  0  0  0 xx' yx' x'|
  // 通过SVD求解Ah = 0，A^T*A最小特征值对应的特征向量即为解
  // 其实也就是右奇异值矩阵的最后一列

  //获取参与计算的特征点的数目
  const int N = vP1.size();

  // 构造用于计算的矩阵 A
  cv::Mat A(2*N,				//行，注意每一个点的数据对应两行
            9,				//列
            CV_32F);      	//float数据类型

  // 构造矩阵A，将每个特征点添加到矩阵A中的元素
  for(int i=0; i<N; i++)
  {
    //获取特征点对的像素坐标
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    //生成这个点的第一行
    A.at<float>(2*i,0) = 0.0;
    A.at<float>(2*i,1) = 0.0;
    A.at<float>(2*i,2) = 0.0;
    A.at<float>(2*i,3) = -u1;
    A.at<float>(2*i,4) = -v1;
    A.at<float>(2*i,5) = -1;
    A.at<float>(2*i,6) = v2*u1;
    A.at<float>(2*i,7) = v2*v1;
    A.at<float>(2*i,8) = v2;

    //生成这个点的第二行
    A.at<float>(2*i+1,0) = u1;
    A.at<float>(2*i+1,1) = v1;
    A.at<float>(2*i+1,2) = 1;
    A.at<float>(2*i+1,3) = 0.0;
    A.at<float>(2*i+1,4) = 0.0;
    A.at<float>(2*i+1,5) = 0.0;
    A.at<float>(2*i+1,6) = -u2*u1;
    A.at<float>(2*i+1,7) = -u2*v1;
    A.at<float>(2*i+1,8) = -u2;

  }

  // 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
  cv::Mat u,w,vt;

  //使用opencv提供的进行奇异值分解的函数
  cv::SVDecomp(A,							//输入，待进行奇异值分解的矩阵
               w,							//输出，奇异值矩阵
               u,							//输出，矩阵U
               vt,						//输出，矩阵V^T
               cv::SVD::MODIFY_A | 		//输入，MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
                   cv::SVD::FULL_UV);		//FULL_UV=把U和VT补充成单位正交方阵

  // 返回最小奇异值所对应的右奇异向量
  // 注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
  return vt.row(8).reshape(0,
                           3);
}

Mat ComputeF21(
    const vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
    const vector<cv::Point2f> &vP2) //归一化后的点, in current frame
{
  // 原理详见附件推导
  // x'Fx = 0 整理可得：Af = 0
  // A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
  // 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解

  //获取参与计算的特征点对数
  const int N = vP1.size();

  //初始化A矩阵
  cv::Mat A(N,9,CV_32F); // N*9维

  // 构造矩阵A，将每个特征点添加到矩阵A中的元素
  for(int i=0; i<N; i++)
  {
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    A.at<float>(i,0) = u2*u1;
    A.at<float>(i,1) = u2*v1;
    A.at<float>(i,2) = u2;
    A.at<float>(i,3) = v2*u1;
    A.at<float>(i,4) = v2*v1;
    A.at<float>(i,5) = v2;
    A.at<float>(i,6) = u1;
    A.at<float>(i,7) = v1;
    A.at<float>(i,8) = 1;
  }

  //存储奇异值分解结果的变量
  cv::Mat u,w,vt;


  // 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
  cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  // 转换成基础矩阵的形式
  cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列

  //基础矩阵的秩为2,而我们不敢保证计算得到的这个结果的秩为2,所以需要通过第二次奇异值分解,来强制使其秩为2
  // 对初步得来的基础矩阵进行第2次奇异值分解
  cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

  // 秩2约束，强制将第3个奇异值设置为0
  w.at<float>(2)=0;

  // 重新组合好满足秩约束的基础矩阵，作为最终计算结果返回
  return  u*cv::Mat::diag(w)*vt;
}
vector<vector<size_t>> RAN_calfeatures(vector<DMatch>mvMatches12, int mMaxIterations  )
{

  const int N = mvMatches12.size();
  random_device rd;
  mt19937 gen(rd());
  vector<size_t> vAllIndices;
  for (int i = 0; i < N; i++) {
    vAllIndices.push_back(i);
  }
  auto mvSets = vector<vector<size_t> >(mMaxIterations, vector<size_t>(8, 0));
  vector<size_t> vAvailableIndices = vAllIndices;
  for (int it = 0; it < mMaxIterations; it++) {
    for (size_t j = 0; j < 8; j++) {
      uniform_int_distribution<> distrib(0, vAvailableIndices.size() - 1);
      int randi = distrib(gen);
      int idx = vAvailableIndices[randi];
      mvSets[it][j] = idx;
      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }
  }
  mvSets.push_back(vAvailableIndices);

  return mvSets;

}




void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21,struct key_img old_image,
struct key_img new_image,int mMaxIterations,vector<vector<size_t>>mvSets,vector<DMatch>goodfeatures ,float mSigma) {

const int N = goodfeatures.size();

// step1. 特征点归一化
vector<cv::Point2f> vPn1, vPn2;
cv::Mat T1, T2;
Normalize(old_image.key_point, vPn1, T1,mMaxIterations);
Normalize(new_image.key_point, vPn2, T2,mMaxIterations);
cv::Mat T2inv = T2.inv(); // 用于恢复原始尺度

// step2. RANSAC循环
score = 0.0; // 最优解得分
vbMatchesInliers = vector<bool>(N, false); // 最优解对应的内点
for (int it = 0; it < mMaxIterations; it++) {
vector<cv::Point2f> vPn1i(8);
vector<cv::Point2f> vPn2i(8);
cv::Mat H21i, H12i;
vector<bool> vbCurrentInliers(N, false);
float currentScore;

for (size_t j = 0; j < 8; j++) {
int idx = mvSets[it][j];
vPn1i[j] = vPn1[goodfeatures[idx].queryIdx]; // first存储在参考帧1中的特征点索引
vPn2i[j] = vPn2[goodfeatures[idx].trainIdx]; // second存储在当前帧2中的特征点索引
}

// step3. 八点法计算单应矩阵H
cv::Mat Hn = ComputeH21(vPn1i, vPn2i);

// step4. 恢复原始尺度
H21i = T2inv * Hn * T1;
H12i = H21i.inv();

// step5. 根据重投影误差进行卡方检验
currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma,goodfeatures,old_image,new_image);

// step6. 记录最优解
if (currentScore > score) {
H21 = H21i.clone();
vbMatchesInliers = vbCurrentInliers;
score = currentScore;
}
}
}

float FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21,struct key_img old_image,
    struct key_img new_image,int mMaxIterations,vector<vector<size_t>>mvSets,vector<DMatch>goodfeatures ,float mSigma) {

const int N = vbMatchesInliers.size();

// step1. 特征点归一化
vector<cv::Point2f> vPn1, vPn2;
cv::Mat T1, T2;
Normalize(old_image.key_point, vPn1, T1,N);
Normalize(new_image.key_point, vPn2, T2,N);
cv::Mat T2t = T2.t(); // 用于恢复原始尺度

// step2. RANSAC循环
score = 0.0; // 最优解得分
vbMatchesInliers = vector<bool>(N, false); // 最优解对应的内点
for (int it = 0; it < mMaxIterations; it++) {
vector<cv::Point2f> vPn1i(8);
vector<cv::Point2f> vPn2i(8);
cv::Mat F21i;
vector<bool> vbCurrentInliers(N, false);
float currentScore;

for (int j = 0; j < 8; j++) {
int idx = mvSets[it][j];
vPn1i[j] = vPn1[goodfeatures[idx].queryIdx]; // first存储在参考帧1中的特征点索引
vPn2i[j] = vPn2[goodfeatures[idx].trainIdx]; // second存储在当前帧2中的特征点索引
}

// step3. 八点法计算单应矩阵H
cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

// step4. 恢复原始尺度
F21i = T2t * Fn * T1;

// step5. 根据重投影误差进行卡方检验
currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma,goodfeatures,old_image,new_image);

// step6. 记录最优解
if (currentScore > score) {
F21 = F21i.clone();
vbMatchesInliers = vbCurrentInliers;
score = currentScore;
}
}
}
bool epipolarConstraintCheck(Mat CameraK, vector<Point2f>& p1s, vector<Point2f>& p2s, Mat R, Mat t)
 {
     for(int i = 0; i < p1s.size(); i++)
     {
         Mat y1 = (Mat_<double>(3,1)<<p1s[i].x, p1s[i].y, 1);
         Mat y2 = (Mat_<double>(3,1)<<p2s[i].x, p2s[i].y, 1);
         //T 的反对称矩阵
         Mat t_x = (Mat_<double>(3,3)<< 0, -t.at<double>(2,0), t.at<double>(1,0),
                     t.at<double>(2,0), 0, -t.at<double>(0,0),
                   -t.at<double>(1,0),t.at<double>(0,0),0);
         Mat d = y2.t() * CameraK.inv().t() * t_x * R * CameraK.inv()* y1;
         cout<<"epipolar constraint = "<<d<<endl;
     }
 }

void feature_points(Mat met,struct key_img &fe_point)
{ //met.resize(480,640);
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
  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
/*
  cv::namedWindow("图上的特征点", CV_WINDOW_NORMAL);
  imshow("图上的特征点",fe_point.image_key);
  waitKey();
*/

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


/*
  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  cv::imshow("匹配结果",result_img);
*/

  waitKey(0);
 // system("pause");

  return goodfeatur;
}

/*vector<Mat> calmatrix(vector<vector<cv::Point2f>> mpoint)
{ Mat R,t;
  Point2f pricipal(323.1992,240.1797);
  float focal=477.7987;
  Mat ess_matrix,Fundamental,homo_matrix;
  Fundamental= findFundamentalMat(mpoint[0],mpoint[1],CV_FM_8POINT);
  float SH,SF;


  ess_matrix =findEssentialMat(mpoint[0],mpoint[1],focal,pricipal);
  *//*cout<<"本质矩阵为"<<endl;
  cout<<ess_matrix<<endl;*//*

  homo_matrix= findHomography(mpoint[0],mpoint[1],RANSAC,3);


  recoverPose(ess_matrix,mpoint[0],mpoint[1],R,t,focal,pricipal);
  vector<Mat>Pose;
  Pose.push_back(R);
  Pose.push_back(t);

  return Pose;

}*/

void drawing(MatrixXf result,struct key_img img)
{
  MatrixXf test(4,4);
  test << 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;
  Mat resut;

  namedWindow("最终结果",WINDOW_NORMAL);
  resizeWindow("最终结果",500,500);
  line(img.image,Point(result(0,0)*100+img.image.cols/2,result(1,0)*100+img.image.rows/2),Point(result(0,1)*100+img.image.cols/2,result(1,1)*100+img.image.rows/2),Scalar(0, 255, 0),10);
  line(img.image,Point(result(0,1)*100+img.image.cols/2,result(1,1)*100+img.image.rows/2),Point(result(0,2)*100+img.image.cols/2,result(1,2)*100+img.image.rows/2),Scalar(0, 255, 0),10);
  line(img.image,Point(result(0,2)*100+img.image.cols/2,result(1,2)*100+img.image.rows/2),Point(result(0,3)*100+img.image.cols/2,result(1,3)*100+img.image.rows/2),Scalar(0, 255, 255),10);
  line(img.image,Point(result(0,3)*100+img.image.cols/2,result(1,3)*100+img.image.rows/2),Point(result(0,0)*100+img.image.cols/2,result(1,0)*100+img.image.rows/2),Scalar(0, 255, 255),10);
  //rectangle(img.image, Point(img.image.cols/2+100,img.image.rows/2+100), Point(img.image.cols/2,img.image.rows/2), Scalar(0, 0, 255),5);
  rectangle(img.image, Point(img.image.cols/2,img.image.rows/2), Point(img.image.cols/2+100,img.image.rows/2+100), Scalar(0, 0, 255),5);
  imshow("最终结果",img.image);


waitKey(0);
system("pause");

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
  //cv::Affine3f camPosition(Mat::eye(3,3,CV_32F),Vec3f(0,0,0));
  cv::Affine3f camPosition(Matx33f (1,0,0,0,1,0,0,0,1),Vec3f(0,0,0));
  //window.showWidget("Coordinate",viz::WCoordinateSystem(),cv::Affine3f::Identity());



  viz::WCameraPosition camparmn(mainCamera.getFov(),nimg.image,1.0,viz::Color::green());
  cv::Affine3d camPosition_2(Pose[0],Pose[1]);
  window.showWidget("oldimage",camparmo,camPosition);
  window.showWidget("newimage",camparmn,camPosition_2);
  window.spin();


}


int main() {
  int mMaxIterations=200;



  vector<DMatch>goodfeatures;





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
  //capture.open("http://admin:123456@192.168.1.105:8081");
    Mat K = (Mat_<double>(3, 3) << 477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1);
    Mat image_address_new = imread( "/home/hu/下载/IMG20230328182935.jpg");
    Mat image_address_old = imread( "/home/hu/下载/IMG20230328182918.jpg");
  // /home/hu/桌面/UPAN/CLionProjects/untitled/cmake-build-debug/F__downloads_001.png
    feature_points(image_address_new, nimage);
    feature_points(image_address_old, oimage);
  goodfeatures = matching(nimage, oimage);
  vector<vector<size_t>> ran_features;
  ran_features=RAN_calfeatures(goodfeatures,mMaxIterations);
  Mat H,F;
  float SH,SF,msigma;

  vector<bool>vbMatchesInliersH, vbMatchesInliersF;
  FindFundamental(vbMatchesInliersF,SF,F,oimage,nimage,mMaxIterations,ran_features,goodfeatures,msigma);
  FindHomography(vbMatchesInliersH,SH,H,oimage,nimage,mMaxIterations,ran_features,goodfeatures,msigma);
cout<<H<<endl;

  /*  Pose = calmatrix(mPoint);
    cout << "R 为" << endl;
    cout << Pose[0] << endl;
    cout <<"T 为" << Pose[1]<<endl;
  transfrom.block<3, 3>(0, 0)<< Pose[0].at<double>(0, 0), Pose[0].at<double>(0, 1), Pose[0].at<double>(0, 2),
      Pose[0].at<double>(1, 0), Pose[0].at<double>(1, 1), Pose[0].at<double>(1, 2),
      Pose[0].at<double>(2, 0), Pose[0].at<double>(2, 1), Pose[0].at<double>(2, 2);
  transfrom.col(3) << Pose[1].at<double>(0, 0), Pose[1].at<double>(1, 0), Pose[1].at<double>(2, 0), 1;
    cout << test * transfrom << endl;
    result = transfrom * test;
   // drawing(result, nimage);
    cout << result << endl;
    //epipolarConstraintCheck(K,mPoint[0],mPoint[1],Pose[0],Pose[1]);  // 验证对极约束
   drawing_rot(Pose,oimage,nimage);
*/
  }
