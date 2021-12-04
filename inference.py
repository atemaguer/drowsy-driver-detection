import argparse
import time

from PIL import Image

# import classify
import tflite_runtime.interpreter as tflite
import platform

import cv2
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.classify import get_classes
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    valid = False
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
            valid = True
    return left_eye, right_eye, valid

def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  inference_size = input_size(interpreter)

  cap = cv2.VideoCapture(args.camera_idx)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2_im = frame

    left_eye, right_eye, valid = detect_eyes(cv2_im, eye_cascade)

    if (valid):
        cv2_im_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        classes = get_classes(interpreter, args.top_k, args.threshold)

        for klass in classes:
            print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

    cv2.imshow('frame', cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()