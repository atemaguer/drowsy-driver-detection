import pathlib
import sys
import time

from draw_utils import *
from facemesh import *
from kalman import *

from cv2 import VideoCapture
from cv2 import waitKey

from periphery import GPIO
from time import sleep

green = GPIO("/dev/gpiochip2", 9, "out") # pin 16
red = GPIO("/dev/gpiochip4", 10, "out")  # pin 18
blue = GPIO("/dev/gpiochip4", 12, "out")  # pin 22

ENABLE_EDGETPU = True

MODEL_PATH = pathlib.Path("./models/")
if ENABLE_EDGETPU:
    DETECT_MODEL = "cocompile/face_detection_front_128_full_integer_quant_edgetpu.tflite"
    MESH_MODEL = "cocompile/face_landmark_192_full_integer_quant_edgetpu.tflite"
    CLASSIFIER_MODEL = "./mobilenet_v2_1.0_224_quant_edgetpu.tflite"
    LABELS_PATH = './eyes_labels.txt'
else:
    DETECT_MODEL = "face_detection_front.tflite"
    MESH_MODEL = "face_landmark.tflite"

# turn on camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret, init_image = cap.read()
if not ret:
    sys.exit(-1)

# instantiate face models
face_detector = FaceDetector(model_path=str(MODEL_PATH / DETECT_MODEL), edgetpu=ENABLE_EDGETPU)
face_mesher = FaceMesher(model_path=str((MODEL_PATH / MESH_MODEL)), edgetpu=ENABLE_EDGETPU)
face_aligner = FaceAligner(desiredLeftEye=(0.38, 0.38))
face_pose_decoder = FacePoseDecoder(init_image.shape)

classifier = Classifier(CLASSIFIER_MODEL,LABELS_PATH, edgetpu=ENABLE_EDGETPU)

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    initial_state=[0, 0, 0, 0],
    input_dim=2,
    cov_process=0.2,
    cov_measure=2) for _ in range(6)]


# detect single frame
def detect_single(image):
    # pad image
    h, w, _ = image.shape
    target_dim = max(w, h)
    padded_size = [(target_dim - h) // 2,
                   (target_dim - h + 1) // 2,
                   (target_dim - w) // 2,
                   (target_dim - w + 1) // 2]
    padded = cv2.copyMakeBorder(image.copy(),
                                *padded_size,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    padded = cv2.flip(padded, 3)

    # face detection
    bboxes_decoded, landmarks, scores = face_detector.inference(padded)

    mesh_landmarks_inverse = []
    r_vecs, t_vecs = [], []
    for i, (bbox, landmark) in enumerate(zip(bboxes_decoded, landmarks)):
        # landmark detection
        aligned_face, M = face_aligner.align(padded, landmark)
        mesh_landmark, _ = face_mesher.inference(aligned_face)
        mesh_landmark_inverse = face_aligner.inverse(mesh_landmark, M)
        mesh_landmarks_inverse.append(mesh_landmark_inverse)


        # tracking
        if i == 0:
            landmark_stable = []
            for mark, stb in zip(landmark.reshape(-1, 2), pose_stabilizers):
                stb.update(mark)
                landmark_stable.append(stb.get_results())
            landmark_stable = np.array(landmark_stable).flatten()
            landmarks[0] = landmark_stable
    # draw
    image_show, eye = draw_face(padded, bboxes_decoded, landmarks, scores, confidence=True)
        
    # remove pad
    image_show = image_show[padded_size[0]:target_dim - padded_size[1], padded_size[2]:target_dim - padded_size[3]]
    return image_show, eye

counter = 0

# endless loop
while True:
    start = time.time()
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # detect single
    image_show, eye = detect_single(image)


    if eye is not None and eye.shape[0] > 0 and eye.shape[1] > 0:
        # put fps
        prediction = classifier.inference(eye)
        
        if prediction == 0:
            counter+=1
        else:
            counter = 0
        
        if counter >= 40:
            red.write(True)
        else:
            red.write(False)
            
    image_show = put_fps(image_show, 1 / (time.time() - start))
    
    result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
    cv2.imshow('demo', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
