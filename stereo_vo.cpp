#include "global.h"
#include "utility.h"
#include "thread"
#include <termio.h>
using namespace std;
using namespace cv;
void SaveCameraTime(const string &filename, double time);

VideoCapture cap;
//int width = 1280;
//int height = 480;
int width = 2560;
int height = 720;
//最快50Hz，再往上也不行了
int FPS = 50;
string dir="";

void createDir() {
    time_t now_time = time(NULL);
    tm *T_tm = localtime(&now_time);
    //转换为年月日星期时分秒结果，如图：
    string timeDetail = asctime(T_tm);
    timeDetail.pop_back();
    dir = "./dataset/" + timeDetail + "img/";
    createDirectory(dir + "right/");
    createDirectory(dir + "left/");
    createDirectory(dir + "space/right/");
    createDirectory(dir + "space/left/");
}

void InitCap(int id) {
    cap.open(id);                             //打开相机，电脑自带摄像头一般编号为0，外接摄像头编号为1，主要是在设备管理器中查看自己摄像头的编号。
    //--------------------------------------------------------------------------------------
    cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));//视频流格式
    cap.set(CV_CAP_PROP_FPS, FPS);//帧率
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);  //设置捕获视频的宽度
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);  //设置捕获视频的高度

    if (!cap.isOpened())                         //判断是否成功打开相机
    {
        cout << "摄像头打开失败!" << endl;
        return;
    }
}

bool showImages = false;
bool remapImage = false;

void GetKey(char& key)
{
    while(1)
    {
        struct termios new_settings;
        struct termios stored_settings;
        tcgetattr(STDIN_FILENO, &stored_settings); //获得stdin 输入
        new_settings = stored_settings;           //
        new_settings.c_lflag &= (~ICANON);        //
        new_settings.c_cc[VTIME] = 0;
        tcgetattr(STDIN_FILENO, &stored_settings); //获得stdin 输入
        new_settings.c_cc[VMIN] = 1;
        tcsetattr(STDIN_FILENO, TCSANOW, &new_settings); //
        key = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &stored_settings);

    }
}

//
int main(int argc, char **argv)            //程序主函数
{
    ros::init(argc, argv, "stereo_video_node");
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it(nh);
    //在camera/image话题上发布图像，这里第一个参数是话题的名称，第二个是缓冲区的大小
    image_transport::Publisher pub0 = it.advertise("/cam0/image_raw", 50);
    image_transport::Publisher pub1 = it.advertise("/cam1/image_raw", 50);
    ROS_INFO("autocar stereo_vo_node init.");
    string down_sample = "0";
    string cam_id = "0";
    string saveImages;
    string grey;
    //注意添加/，表示ros空间里的变量
    nh.param<std::string>("DOWN_SAMPLE", down_sample, "0");
    nh.param<std::string>("CAM_ID", cam_id, "0");
    nh.param<std::string>("SAVE", saveImages, "0");
    nh.param<std::string>("GREY", grey, "0");

    ROS_INFO("DOWN_SAMPLE %d, CAM_ID %d.",stoi(down_sample),stoi(cam_id));
    InitCap(stoi(cam_id));

    if(saveImages!="0")
        createDir();

    Mat frame = Mat::zeros(Size(width, height), CV_8UC3);
    Mat frameGrey = Mat::zeros(Size(width, height), CV_8UC1);
    Mat frame_L, frame_R;
    uint64_t count = 0;
    bool cameraFail = false;

    string strwriting = "/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/cali/result/robot_orb_stereo.yaml";

    cv::FileStorage fsSettings(strwriting.c_str(), cv::FileStorage::READ);

    if (remapImage && !fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (remapImage &&( K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
        D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)) {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l, M2l, M1r, M2r;
    if(remapImage)
    {
        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l,
                                    M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r,
                                    M2r);
    }

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    if (showImages) {
        namedWindow("Video_L");
        namedWindow("Video_R");
    }
    ros::Rate loop_rate(200);
    double tNow = ros::Time::now().toSec();
    ros::Time tImage = ros::Time::now();
    vector<double > timeStamp;
    timeStamp.reserve(20000);
    char key;
    thread getKey(GetKey,std::ref(key));
    int spaceCnt = 0;
    while (ros::ok()) {
        //好像固定50 fps
        if (cap.read(frame)) {
            count++;
            tImage = ros::Time::now();
            if(grey!="0")
            {
                //    变成灰度图
                if (frame.channels() == 3) {
                    cvtColor(frame, frameGrey, CV_BGR2GRAY);
                } else if (frame.channels() == 4) {
                    cvtColor(frame, frameGrey, CV_BGRA2GRAY);
                }
                //cout<<frameGrey.size()<<endl;
                frame_L = frameGrey(Rect(0, 0, width / 2, height));  //获取缩放后左Camera的图像
                frame_R = frameGrey(Rect(width / 2, 0, width / 2, height)); //获取缩放后右Camera的图像
            }else
            {
                frame_L = frame(Rect(0, 0, width / 2, height));  //获取缩放后左Camera的图像
                frame_R = frame(Rect(width / 2, 0, width / 2, height)); //获取缩放后右Camera的图像
            }

            if (remapImage) {
                cv::remap(frame_L, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(frame_R, imRightRect, M1r, M2r, cv::INTER_LINEAR);
            } else {
                imLeftRect = frame_L;
                imRightRect = frame_R;
            }
            if((down_sample=="1" && count%2==0) || down_sample=="0")
            {
                string encoding = "mono8";
                if(grey=="0")
                    encoding = "bgr8";
                sensor_msgs::ImagePtr msg0 = cv_bridge::CvImage(std_msgs::Header(), encoding, imLeftRect).toImageMsg();
                sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), encoding, imRightRect).toImageMsg();
                msg0->header.stamp = tImage;
                msg0->header.frame_id = "/sweep";
                msg1->header.stamp = tImage;
                msg1->header.frame_id = "/sweep";

                pub0.publish(msg0);
                pub1.publish(msg1);

                if (saveImages!="0") {
                    double now = tImage.toSec();
                    string image_L_path = dir + "left/" + to_string(uint64_t(now*1e9)) + ".jpg";
                    string image_R_path = dir + "right/" + to_string(uint64_t(now*1e9)) + ".jpg";
                    SaveCameraTime(dir + "cameraStamps.txt", now*1e9);
                    imwrite(image_L_path, imLeftRect);
                    imwrite(image_R_path, imRightRect);
                    if(count%10==0)
                        ROS_INFO("save %d", count);

                    //32对应空格
                    if (key == 32) {
                        key=0;
                        string image_L_path = dir + "space/left/" + to_string(uint64_t(now*1e9)) + ".jpg";
                        string image_R_path = dir + "space/right/" + to_string(uint64_t(now*1e9)) + ".jpg";
                        imwrite(image_L_path, frame_L);
                        imwrite(image_R_path, frame_L);
                        spaceCnt++;
                        ROS_INFO("save space img %d", spaceCnt);
                    }
                }
            }

            if (showImages) {
                imshow("Video_R", imLeftRect);
                imshow("Video_L", imRightRect);
                waitKey(1);
            }
        } else {
            if (!cameraFail) {
                ROS_INFO("fail");
                cameraFail = true;
            }
        }
        //waitKey(0);
        ros::spinOnce();
        //loop_rate.sleep();
    }

    return 0;
}


void SaveCameraTime(const string &filename, double time) {
    ofstream f;
    f.open(filename.c_str(), ios::out | ios::app);
    f << fixed;
        f << setprecision(9) << time  << endl;
    f.close();
}
