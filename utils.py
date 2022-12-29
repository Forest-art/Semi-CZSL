import os
import torch
import numpy as np
import random
import os
import yaml
import logging
from datetime import datetime

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


class Log():
    def __init__(self, log_level = logging.INFO) -> None:
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关

        # 第二步，创建一个handler，用于写入日志文件
        rq = datetime.now().strftime("%Y%m%d")[:None] #获取当天的日期，在logs文件夹下每天生成一个日志文件
        log_path = './logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        if not os.path.exists(logfile):
            os.mknod(logfile)
        fh = logging.FileHandler(logfile, encoding="utf-8", mode='a+') # mode的使用见以下Python文件读写
        fh.setLevel(log_level)  # 输出到file的log等级的开关

        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)


    def debug(self, msg):
        self.logger.debug(msg)


    def warning(self, msg):
        self.logger.warning(msg)


    def error(self, msg):
        self.logger.error(msg)


    def critical(self, msg):
        self.logger.critical(msg)