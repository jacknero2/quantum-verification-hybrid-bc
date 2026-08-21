# TODO: log settings
import logging
# Source: https://stackoverflow.com/a/66209331/19768075
class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self.buffer = []

    def write(self, msg : str):
        if msg.endswith('\n'):
            self.buffer.append(msg.removesuffix('\n'))
            self.logger_func(''.join(self.buffer))
            self.buffer = []
        else:
            self.buffer.append(msg)

    def flush(self):
        pass
    
def set_logger(log_file, log_level=logging.INFO):
    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = logging.FileHandler('logs/' + log_file, mode = 'w')
    file_handler.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    
    log_format='%(asctime)s - %(name)-13s - %(levelname)-7s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger