import logging
import os
from logging.handlers import RotatingFileHandler
import colorlog  # 控制台日志输入颜色

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


class Log:
    def __init__(self, logname: str = 'log.local.log'):
        self.logname = logname
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
            log_colors=log_colors_config)

        # # 创建一个 FileHandler，写到本地
        # fh = logging.handlers.TimedRotatingFileHandler(
        #     self.logname, when='MIDNIGHT', interval=1, encoding='utf-8')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(self.formatter)
        # self.logger.addHandler(fh)

        # 创建一个StreamHandler,写到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

    def console(self, level: str, message: str):
        LV = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warn': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }
        f = LV.get(level, self.debug)
        f(message)

    def debug(self, message):
        self.console('debug', message)

    def info(self, message):
        self.console('info', message)

    def warning(self, message):
        self.console('warning', message)

    def error(self, message):
        self.console('error', message)

    def critical(self, message):
        self.console('critical', message)


if __name__ == "__main__":
    log = Log()
    log.info("测试1")
    log.debug("测试2")
    log.warning("warning")
    log.error("error")
    log.critical("critical")