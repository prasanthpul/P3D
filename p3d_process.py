from __future__ import unicode_literals
import youtube_dl
import os
import cv2
import sys, getopt
import numpy
import youtube_dl
import cntk as C
from cntk.ops import *
import time

def get_frameset(cap, clipLen):
    frameset = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while (len(frameset) < clipLen):
        cur_pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if cur_pos_frame >= frame_count:
            break
        flag, frame = cap.read()
        if flag:
            frameset.append(frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos_frame-1)
            print("frame %d is not ready" % cur_pos_frame)
            cv2.waitKey(1000)
    return frameset

def prep_frames(frameset):
    meanbgr = [104,117,123]
    frames = np.zeros((3, 24, 160, 160), dtype=np.float32)
    for i in range(0,len(frameset)-1):
        frame = frameset[i]
        frame = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_CUBIC) - meanbgr
        frame = frame.transpose([2,0,1])
        frames[:, i, :, :] = frame
    return np.ascontiguousarray(frames, dtype=np.float32)

def predict_on_frameset(frameset, p3d_model, labels):
    starttime = time.time()
    prob_node = combine([p3d_model.find_by_name('prob').owner])
    prob = np.squeeze(prob_node.eval({prob_node.arguments[0]: [frameset]}))
    pred = np.argmax(prob)
    predtime = time.time() - starttime
    return predtime, labels[pred], prob[pred]


def apply_label(img, label, origin, highlight):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255,255,255)
    lineType  = 2

    textsize, baseline = cv2.getTextSize(label, font, fontScale, lineType)
    color = (128, 128, 128)
    if (highlight):
        color = (0,150,0)
    cv2.rectangle(img, (origin[0], origin[1]+baseline), (origin[0]+textsize[0], origin[1]-textsize[1]-baseline), color, -1)
    cv2.putText(img, label, origin, font, fontScale, fontColor, lineType)
    return img

def process_video(inputvid, p3dmodel, labels, outputvid, realtime):
    clipLen = 24

    cap = cv2.VideoCapture(inputvid)
    while not cap.isOpened():
        cap = cv2.VideoCapture(inputvid)
        cv2.waitKey(1000)
        print("Wait for the header")

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG') 
    out = cv2.VideoWriter(outputvid, fourcc, 20.0, size)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Processing %d frames" % frame_count)
    while True:
        start_pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frameset = get_frameset(cap, clipLen)
        predtime, label, prob = predict_on_frameset(prep_frames(frameset), p3dmodel, labels)
        cur_pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(">>> %d,%0.2fs,%s,%f                     " % (cur_pos_frame, predtime, label, prob), end='\r', flush=True)

        topLeft   = (50,50)
        for frame in frameset:
            frame = apply_label(frame, '%d: %s (%f)' % (start_pos_frame, label, prob), topLeft, (prob > 0.75))
            if (realtime):
                cv2.imshow('video', frame)
                cv2.waitKey(2)
            out.write(frame)
            start_pos_frame += 1

        if cur_pos_frame >= frame_count:
            break

    out.release()
    cap.release()
    return cur_pos_frame

def load_labels(filepath):
    if not (os.path.isfile(filepath)):
        raise IOError("Cannot find file: %s" % (filepath))
    with open(filepath, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip('\n')
    return data

def init_p3d_model():
    # Load p3d model
    print('Initializing P3D model...')
    if not (os.path.isfile('p3d_resnet_kinetics.onnx')):
        raise IOError("Cannot find model: %s" % ('p3d_resnet_kinetics.onnx'))
    p3d_model = load_model('p3d_resnet_kinetics.onnx', format=C.ModelFormat.ONNX)
    # Load activity names
    if not (os.path.isfile("labels.txt")):
        raise IOError("Cannot find labels: %s" % ("labels.txt"))
    labels = load_labels("labels.txt")
    print('P3D model initialized')
    return p3d_model, labels

def download_youtube(youtubeurl, outdir):
    # TODO: find a way to get downloaded file name instead of using hardcoded name
    ydl_opts = {'outtmpl': outdir+'/video.mp4'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtubeurl])
    return outdir+'/video.mp4'

def main(argv):
    youtubeurl = ""
    outdir = "." 
    realtime = False
    saveframes = True
    inputvid = ""
    try:
        opts, args = getopt.getopt(argv, "ry:i:o:", ["realtime","youtubeurl=","inputvid=","outputfolder="])
    except getopt.GetoptError:
        print('p3d_process.py (-y <youtubeurl> | -i <inputvid>) -o <outputfolder> [-r]\n')
        print('This script generates a new version of the specified video with actions identified.')
        print('The input video can be a file on the local machine or a YouTube URL.')
        print('You can specify the -r realtime flag to view the frames in realtime. Otherwise you can view the generated video after it is done.')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--inputvid"):
            inputvid = arg
        elif opt in ("-y", "--youtubeurl"):
            youtubeurl = arg
        elif opt in ("-o", "--outputfolder"):
            outdir = arg
        elif opt in ("-r", "--realtime"):
            realtime = True

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if youtubeurl:
        inputvid = download_youtube(youtubeurl, outdir)
    outputvid = os.path.join(outdir, "p3d_video.mp4")
    p3dmodel, labels = init_p3d_model()
    frame_count = process_video(inputvid, p3dmodel, labels, outputvid, realtime)
    print("%s created" % outputvid)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
