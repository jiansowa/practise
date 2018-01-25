# -*- coding: utf-8 -*-
import base64
import random
import traceback
import urllib2
import os
import time
import logging
import uuid
import cv2
import json
import redis
import re

import argparse
import shutil
import sys
import subprocess
import multiprocessing

from enum import Enum

# 摄像头提取的帧保存目录
frame_root_dir = '/data01/retail_frames_local'

# 摄像头提取的帧放入到redis等待后续异步处理, 每个摄像头一个redis, 这样做的目的是让每个摄像头的处理是单进程顺序处理,避免并行出现
# 前后帧处理时序不对的BUG
redis_frames_queue = 'retail:frame:urls:{}'

class FG_S(Enum):
    fg_state_init  = 1
    fg_state_still = 2
    fg_state_move  = 3

def process_camera_stream(camera_id, stream_url, redis_url, sample_per_frame=1):
    """
    从stream_url读取摄像头的视频流,按照sample_rate的采样率采样,让后将采样图片保存到文件夹,并将url等信息保存到redis做后续处理

    海康默认的视频编码格式是H265,要改成H264,否则抽帧会经常遇到灰屏
    cap = cv2.VideoCapture("rtsp://admin:gypsii2009@192.168.159.122:554/Streaming/Channels/101?transportmode=unicast")

    针孔摄像头400万, 默认分辨率是4M,这个会花屏,改为1080P就可以了
    cap = cv2.VideoCapture('rtsp://192.168.159.201:554/user=test&password=aaa111&channel=1&stream=0.sdp?')

    :param: camera_id: 摄像头编号
    :param stream_url: 摄像头rtsp url
    :param redis_url: redis server url
    :param sample_per_frame: 每秒采样几帧
    :return:
    """
    while True:
        try:
            redis_server = redis.Redis(host=redis_url)

            # 初始化队列
            redis_server.delete(redis_frames_queue.format(camera_id))

            cap = cv2.VideoCapture(stream_url)
            logging.info("stream_url:{}".format(stream_url))
            logging.info('camera id[{}] open status: {}'.format(camera_id, cap.isOpened()))

            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                cur_time = time.time()

                if ret:
                    duration = int((cur_time - start_time) * 1000)
                    if duration >= 100:
                        # frame裁剪, 如果frame超过2560, 则剧中裁剪2560处理
                        (f_h, f_w, _) = frame.shape

                        new_w = 2560
                        if f_w > new_w:
                            # frame = cv2.resize(frame, (new_w, f_h * new_w/f_w))
                            new_x1 = f_w / 2 - new_w / 2
                            new_y1 = 0
                            new_x2 = f_w / 2 + new_w / 2
                            new_y2 = f_h

                            frame = frame[new_y1: new_y2, new_x1: new_x2]

                        # 帧文件命名格式: <摄像头ID>_<抽帧的时间 unix timestamp>_uuid.jpg
                        frame_url = os.path.join(frame_root_dir,
                                                 '{}_{}_{}.jpg'.format(camera_id, int(cur_time), uuid.uuid1()))
                        cv2.imwrite(frame_url, frame)
                        # logging.debug('extract frame from camera: {} {}'.format(camera_id, frame_url))
                        try:
                            dict = {
                                'cid': camera_id,
                                'url': frame_url,
                                'size': {'height': frame.shape[0], 'width': frame.shape[1]},
                                'create': int(cur_time),
                            }

                            redis_server.lpush(redis_frames_queue.format(camera_id), json.dumps(dict))
                            start_time = time.time()
                        except Exception, e:
                            # 无法处理此帧, 删除帧的图片
                            logging.error(traceback.format_exc())
                            os.remove(frame_url)
                else:
                    logging.debug('extract frame False camera: {}'.format(camera_id))
                    cap.release()

            time.sleep(2)

        except Exception, e:
            logging.error(e)
            logging.error(traceback.format_exc())
            time.sleep(10)

bs = cv2.createBackgroundSubtractorMOG2(500,64,False)

def set_background(frame,frame_time,frame_url,bg_cnxt):
    logging.info("set_background")
    ret,blob = cv2.imencode('.jpg',frame)
    bg_cnxt['bg_frame'] = base64.b64encode(blob)
    bg_cnxt['bg_time'] = frame_time
    bg_cnxt['bg_url'] = frame_url
    bg_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
    bg_cnxt['bg_gray'] = bg_gray#.tobytes()

def if_check_rack(frame,frame_time,frame_url,bg_cnxt,ratio=0.01):
    need_check_rack = False
    fgmask = bs.apply(frame)
    fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
    th[th==255]=1 
    changes = th.sum()

    w = bg_cnxt['bg_w']
    h = bg_cnxt['bg_h']

    logging.debug("changes:{},total:{},ratio:{}".format(changes,w*h,changes/w/h))
    if(changes > w*h*ratio):
        bg_cnxt['bg_cnt'] = 0
        #bg_cnxt['bg_state'] = FG_S.fg_state_move
        #return False
    else:
        bg_cnxt['bg_cnt'] += 1

    count_to_still = 24 
    if(bg_cnxt['bg_cnt'] > count_to_still and bg_cnxt['bg_state'] == FG_S.fg_state_init):
        set_background(frame,frame_time,frame_url,bg_cnxt)
        bg_cnxt['bg_state'] = FG_S.fg_state_still
    if(bg_cnxt['bg_cnt'] > count_to_still and bg_cnxt['bg_state'] == FG_S.fg_state_move):
        bg_cnxt['bg_state'] = FG_S.fg_state_still
        need_check_rack = True
    if(bg_cnxt['bg_state'] == FG_S.fg_state_still):
        if(bg_cnxt['bg_cnt'] > count_to_still):
            bg_cnxt['bg_cnt'] = count_to_still + 88
            need_check_rack = True
        else:
            bg_cnxt['bg_state'] = FG_S.fg_state_move

    logging.info("bg_state:{}".format(bg_cnxt['bg_state']))
    return need_check_rack

def detect_rack_changes(frame,bg_cnxt):
    w = bg_cnxt['bg_w']
    h = bg_cnxt['bg_h']

    background = bg_cnxt['bg_gray']
    #background = numpy.array(background).reshape(h,w)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]

    diff[diff==255] = 1
    changes = diff.sum()

    ratio = changes/w/h

    return ratio

def dump_sku_diff_report(report):
    logging.info("prev_time:{}".format(report['prev_time']))
    logging.info("prev_url:{}".format(report['prev_url']))
    logging.info("cur_time:{}".format(report['cur_time']))
    logging.info("cur_url:{}".format(report['cur_url']))
    logging.info("diff_ratio:{}".format(report['diff_ratio']))

def process_one_frame(frame_info,bg_cnxt):
    frame = cv2.imread(frame_info['url'])
    frame_time = time.time()
     
    (h,w,_) = frame.shape
    bg_cnxt['bg_w'] = w
    bg_cnxt['bg_h'] = h

    need_check = if_check_rack(frame,frame_time,frame_info['url'],bg_cnxt)

    #need_check = False
    if(need_check):
        change_ratio = detect_rack_changes(frame,bg_cnxt)
        logging.info("change_ratio:{}".format(change_ratio))
        if(change_ratio > 0.05):
            report={}
            report['prev_frame'] = bg_cnxt['bg_frame']
            report['prev_time'] = bg_cnxt['bg_time']
            report['prev_url'] = bg_cnxt['bg_url']
            ret,blob = cv2.imencode('*.jpg',frame)
            report['cur_frame'] = base64.b64encode(blob)
            report['cur_time'] = frame_time
            report['cur_url'] = frame_info['url']
            report['diff_ratio'] = change_ratio
            set_background(frame,frame_time,frame_info['url'],bg_cnxt)
            dump_sku_diff_report(report)
            
    cv2.imshow('frame',frame)
    

def process_frame_sku_hot(redis_url, camera_id, port_start, port_end):
    """
    处理视频帧
    :param redis_url: redis服务器地址    :param camera_id: 摄像头ID
    :param port_start: facenet服务器集群的起始端口
    :param port_end: facenet服务器集群的结束端口
    :return:
    """
    logging.debug('frame processor for camera id[{}]'.format(camera_id))

    while True:
        try:

            logging.debug('connecting redis {}'.format(redis_url))
            redis_server = redis.Redis(redis_url)

            bg_cnxt = {}
            bg_cnxt['bg_cnt'] = 0
            bg_cnxt['bg_state'] = FG_S.fg_state_init
            while True:
                try:
                    # 从队列读取帧
                    (key, frame_info) = redis_server.brpop(redis_frames_queue.format(camera_id))

                    frame_info = json.loads(frame_info)
                    #logging.debug('get frame from camera: {} {}'.format(camera_id, frame_info['url']))

                    try:
                        process_one_frame(frame_info,bg_cnxt)
                    finally:
                        # 删除帧的图片
                        # TODO 最后放出来
                        os.remove(frame_info['url'])
                        #logging.debug("remove",frame_info['url'])
                        #pass

                    # needed by imshow
                    cv2.waitKey(1)
                except Exception, e:
                    logging.error(e)
                    #logging.error(traceback.format_exc())

        except Exception, e:
            #logging.error(e)
            #logging.error(traceback.format_exc())
            time.sleep(5)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='摄像头视频流抽帧程序')

    parser.add_argument('-r', '--redis_url', type=str, help="redis地址", default='127.0.0.1')
    parser.add_argument('-ps', '--port_start', dest='port_start', default=10001, type=int, help="facenet起始端口")
    parser.add_argument('-pe', '--port_end', dest='port_end', default=10002, type=int, help="facenet结束端口")
    parser.add_argument('-spf', '--sample_per_frame', dest='sample_per_frame', default=1, type=int, help="每秒采样几帧")
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='DEBUG', help="Set the logging level")
    parser.add_argument('-debug', dest='debug', default=False, action='store_true', help="调试模式")

    # 当有以下两个参数的时候, 表示启动捕获具体某个摄像头的视频流程序
    parser.add_argument('-cid', '--camera_id', type=str, help="camera ID")  # , default='1')
    parser.add_argument('-rtsp', '--rtsp_url', type=str,
                        help="camera rtsp URL")  # , default='rtsp://192.168.159.201:554/user=test&password=aaa111&channel=1&stream=0.sdp?')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=args.logLevel, filename='../logs/retail_stream_reader.log',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    if args.camera_id and args.rtsp_url:
        # 启动指定摄像头视频流捕获程序
        process_camera_stream(args.camera_id, args.rtsp_url, args.redis_url)
    else:
        # 程序入口
        # 初始化帧保存目录
        if not os.path.exists(frame_root_dir):
            os.makedirs(frame_root_dir)
        else:
            # 删除掉帧目录, 避免堆积过的的无用frame
            shutil.rmtree(frame_root_dir)
            os.makedirs(frame_root_dir)

        # test with record video
        camera_list = [
                        {
                            "camera_id":1,
                            "type":3,
                            # desktop
                            #"rtsp_url":'rtsp://192.168.159.135:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?'
                            # counter top
                            #"rtsp_url":'rtsp://192.168.159.117:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?'
                            #"rtsp_url":'rtsp://192.168.159.102:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?'
                            # counter side
                            #"rtsp_url":'rtsp://192.168.159.220:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?'
                            # front angle
                            "rtsp_url":'rtsp://192.168.159.213:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?'
                            # record
                            #"rtsp_url":'/home/jiansowa/Videos/vlc-record-2017-12-27-16h45m09s-rtsp___192.168.159.117_554_user=test&password=aaaa1111&channel=1&stream=0.sdp_-.mp4'
                            #"rtsp_url":'/home/jiansowa/Videos/vlc-record-2017-12-27-1700.mp4' 
                            #"rtsp_url":'/data01/vlc-record-2017-12-27-1700.mp4' 
                        }
                    ]


        for camera in camera_list:
            sample_per_frame = 20
            cmd = 'python {} -r {} -l {} -cid {} -rtsp {} -spf {}'.format(sys.argv[0], args.redis_url,
                                                                          args.logLevel,
                                                                          int(camera['camera_id']),
                                                                          camera['rtsp_url'],
                                                                          sample_per_frame)
            subprocess.Popen(cmd.split(' '))
            logging.info('camera id[{}] stream reader started. cmd: {}'.format(camera['camera_id'], cmd))

        # 启动视频帧分析程序, 每个摄像头一个视频帧分析进程, 这样做的目的是让每个摄像头的处理是单进程顺序处理,避免并行出现
        # 上传进程数量
        pool = multiprocessing.Pool(processes=len(camera_list))

        for camera in camera_list:
            if camera['type'] == 1:
                # 默认摄像头
                logging.info('启动默认摄像头程序')
            elif camera['type'] == 2:
                # 进门摄像头
                logging.info('启动进门摄像头程序')
            elif camera['type'] == 3:
                # 俯拍
                logging.info('启动俯拍摄像头程序')
                pool.apply_async(process_frame_sku_hot,
                                 (args.redis_url, int(camera['camera_id']), args.port_start, args.port_end))
            else:
                logging.info('未知的摄像头类型: {}'.format(camera))


        # 死循环, 不退出
        while (True):
            time.sleep(10)
 
