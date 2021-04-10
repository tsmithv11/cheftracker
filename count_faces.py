from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from matplotlib import pyplot
from os.path import isdir
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from numpy import save as np_save
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import cv2
import pickle
from multiprocessing import cpu_count, Process, Manager
from csv import DictWriter

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def extract_faces(image, required_size=(160, 160)):
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    
    faces_array = []
    for n in range(len(results)):
        # extract the bounding box from the first face
        x1, y1, width, height = results[n]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        faces_array.append(face_array)
    return faces_array


def process_video(face_count,file,frame_jump,group_number):
    print("Worker: ",group_number)
    model2 = load_model('facenet_keras.h5')
    cap = cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump * group_number)
    out_encoder = LabelEncoder()
    out_encoder.classes_ = load('classes.npy')
    model = pickle.load(open('svc_model.sav', 'rb'))


    count = 0
    while count < frame_jump:
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if count % 200 == 0:
            print("Group ",group_number,": ",count)
            print(face_count)

        faces = extract_faces(frame)

        if count % 15 == 0:
            for n in range(len(faces)):
                temp_face = faces[n]
                random_face_emb = get_embedding(model2, temp_face)
                # prediction for the face
                samples = expand_dims(random_face_emb, axis=0)
                yhat_class = model.predict(samples)
                yhat_prob = model.predict_proba(samples)
                # get name
                class_index = yhat_class[0]
                class_probability = yhat_prob[0,class_index] * 100
                predict_names = out_encoder.inverse_transform(yhat_class)

                if class_probability > 98:
                    if predict_names[0] in face_count:
                        face_count[predict_names[0]] += 1
                    else:
                        face_count[predict_names[0]] = 1
        count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_location = '{video/location.mp4}'
    num_processes = cpu_count()
    print("CPUs: ",num_processes)

    cap_temp = cv2.VideoCapture(video_location)
    total_frames = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total Frames: ",int(total_frames))
    frame_jump =  cap_temp.get(cv2.CAP_PROP_FRAME_COUNT) // num_processes
    print("Frame Jump: ",int(frame_jump))

    jobs = []
    face_count = Manager().dict()
    for i in range(num_processes):
        p = Process(target=process_video,args=(face_count,video_location,frame_jump,i))
        jobs.append(p)

    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print(face_count)

    face_count = dict(face_count)

    with open('output.csv', 'w') as f:
        w = DictWriter(f, face_count.keys())
        w.writeheader()
        w.writerow(face_count)
