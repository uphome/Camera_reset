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


  //开始处理图片
  while (1)
  {


    capture>>capture_img;

    feature_points(capture_img,sou_image,"sou");
    feature_points(tar_img,tar_image,"tar");
    good_matches = matching(sou_image, tar_image);

    camera.set_number(K,good_matches,sou_image.key_point,tar_image.key_point);
    Mat R,T;
    vector<cv::Point3f> vP3D;
    vector<bool> vbTriangulated;
    //bool Hu;

//camera.Initialize(R,T,vP3D,vbTriangulated);

    const int N = camera.mvMatches12.size();
    // Indices for minimum set selection
    // 新建一个容器vAllIndices存储特征点索引，并预分配空间
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);

    //在RANSAC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引，所以这里起的名字叫做可用的索引
    vector<size_t> vAvailableIndices;
    default_random_engine   e;//随机数引擎对象
    e.seed(time(NULL)); //初始化种子
    //初始化所有特征点对的索引，索引值0到N-1
    for(int i=0; i<N; i++)
    {
      vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
    // 共选择 mMaxIterations (默认200) 组
    //mvSets用于保存每次迭代时所使用的向量
    camera.mvSets = vector< vector<size_t> >(camera.mMaxIterations,		//最大的RANSAC迭代次数
                                      vector<size_t>(8,0));	//这个则是第二维元素的初始值，也就是第一维。这里其实也是一个第一维的构造函数，第一维vector有8项，每项的初始值为0.

    //用于进行随机数据样本采样，设置随机数种子


    //开始每一次的迭代
    for(int it=0; it<camera.mMaxIterations; it++)
    {
      //迭代开始的时候，所有的点都是可用的
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
        camera.mvSets[it][j] = idx;

        // 由于这对点在本次迭代中已经被使用了,所以我们为了避免再次抽到这个点,就在"点的可选列表"中,
        // 将这个点原来所在的位置用vector最后一个元素的信息覆盖,并且删除尾部的元素
        // 这样就相当于将这个点的信息从"点的可用列表"中直接删除了
        vAvailableIndices[randi] = vAvailableIndices.back();
        vAvailableIndices.pop_back();
      }//依次提取出8个特征点对
    }//迭代mMaxIterations次，选取各自迭代时需要用到的最小数据集



    // Launch threads to compute in parallel a fundamental matrix and a homography
    // Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算

    //这两个变量用于标记在H和F的计算中哪些特征点对被认为是Inlier
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    //计算出来的单应矩阵和基础矩阵的RANSAC评分，这里其实是采用重投影误差来计算的
    float SH, SF; //score for H and F
    //这两个是经过RANSAC算法后计算出来的单应矩阵和基础矩阵
    cv::Mat H, F;

    // 构造线程来计算H矩阵及其得分
    // thread方法比较特殊，在传递引用的时候，外层需要用ref来进行引用传递，否则就是浅拷贝
    camera.FindHomography(vbMatchesInliersH,SH,H);
    //输出，计算的单应矩阵结果
    // 计算fundamental matrix并打分，参数定义和H是一样的，这里不再赘述
    camera.FindFundamental(vbMatchesInliersF,SF,F);

    // Wait until both threads have finished
    //等待两个计算线程结束

    // Compute ratio of scores
    // Step 4 计算得分比例来判断选取哪个模型来求位姿R,t
    //通过这个规则来判断谁的评分占比更多一些，注意不是简单的比较绝对评分大小，而是看评分的占比
    float RH = SH/(SH+SF);			//RH=Ratio of Homography

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 注意这里更倾向于用H矩阵恢复位姿。如果单应矩阵的评分占比达到了0.4以上,则从单应矩阵恢复运动,否则从基础矩阵恢复运动
    if(RH>0.40)
      //更偏向于平面，此时从单应矩阵恢复，函数ReconstructH返回bool型结果
      camera.ReconstructH(vbMatchesInliersH,	//输入，匹配成功的特征点对Inliers标记
                          H,					//输入，前面RANSAC计算后的单应矩阵
                          camera.mK,					//输入，相机的内参数矩阵
                          R,T,			//输出，计算出来的相机从参考帧1到当前帧2所发生的旋转和位移变换
                          vP3D,				//特征点对经过三角测量之后的空间坐标，也就是地图点
                          vbTriangulated,		//特征点对是否成功三角化的标记
                          1.0,				//这个对应的形参为minParallax，即认为某对特征点的三角化测量中，认为其测量有效时
          //需要满足的最小视差角（如果视差角过小则会引起非常大的观测误差）,单位是角度
                          50);				//为了进行运动恢复，所需要的最少的三角化测量成功的点个数
    else //if(pF_HF>0.6)
      // 更偏向于非平面，从基础矩阵恢复
      camera.ReconstructF(vbMatchesInliersF,F,camera.mK,R,T,vP3D,vbTriangulated,1.0,50);






    //watermark(capture_img,tar_img);
    //window.spinOnce();



  }





}
