import sys
import datetime
import os.path
from shutil import copyfile
import subprocess
import atexit
from sys import platform

# check python version
assert sys.version_info[:2] == (3, 6), 'Python3.6 Needed!'

# check Python packages
import numpy
import sklearn
# import xgboost
import pandas
import scipy
import lightgbm





def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


class Logger(object):
  def __init__(self, output_dir):
    ensure_dir(output_dir)
    log_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self.log_dir = os.path.join(output_dir, log_name)
    ensure_dir(self.log_dir)
    file_path = os.path.join(self.log_dir, 'log.log')
    self.terminal = sys.stdout
    self.log = open(file_path, "a", encoding='utf-8')

  def write(self, message):
    if message == '\n':
        return
    time = datetime.datetime.now()
    message = f'{time}: {message}\n'
    self.terminal.write(message)
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.log.flush()

def is_linux():
  return platform == "linux" or platform == "linux2"

logger = Logger('../log')
sys.stdout = logger
sys.stderr = logger

log_path = logger.log_dir

# copy python files

for filename in os.listdir('.'):
    if filename.endswith(".py"):
      copyfile(filename, os.path.join(log_path, filename))


__all__ = [log_path, is_linux]


if is_linux():
  memory_logfile = log_path + '/system.log'
  memory_f = open(memory_logfile, 'w', encoding='utf-8')

  bash = 'bash ./memory.sh'
  child = subprocess.Popen(bash.split(), stdout=memory_f)
  def exit_handler():
    print('process will exit')
    child.terminate()
    memory_f.flush()
    memory_f.close()
  atexit.register(exit_handler)
