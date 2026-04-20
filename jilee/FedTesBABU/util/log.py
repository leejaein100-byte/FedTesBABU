import json
import os
import multiprocessing

def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if isinstance(text, dict):
            text = json.dumps(text)
            if display:
                print(text)
        elif isinstance(text, list):
            text = ', '.join(map(str, text))
            if display:
                print(text)
        else:
            text = str(text)  # Co
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close

#class Logger:
    #def __init__(self, log_filename, display=True):
        #self.f = open(log_filename, 'a')
        #self.counter = 0
        #self.display = display
        #self.lock = multiprocessing.Lock()  # Ensure that logging is safe for multiprocessing

    #def log(self, text):
        #with self.lock:
            #if self.display:
             #   print(text)
            #self.f.write(text + '\n')
            #self.counter += 1
            #if self.counter % 10 == 0:
                #self.f.flush()
                #os.fsync(self.f.fileno())

    #def close(self):
        #with self.lock:
            #self.f.close()

#def create_logger(log_filename, display=True):
    #logger = Logger(log_filename, display)
    #return logger.log, logger.close
