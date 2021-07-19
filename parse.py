'''
bounding box parser
'''

import numpy as np
import os, cv2
import pandas as pd



def read_image(task_path, depth_path, frame):
    imgname = task_path + 'RGB_' + str(frame + 1) + '.png'
    dimgname = depth_path + 'Depth_' + str(frame + 1) + '.png'
    oriImg = cv2.imread(imgname)  # B,G,R order
    dimg = cv2.imread(dimgname, 0)
    boxsize = 480
    scale = boxsize / (oriImg.shape[0] * 1.0)
    img = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img, dimg


def objects(lines, num_obj, UL, LR):
    for line in lines:
        prsl = line.split(',')
        fr = int(prsl[0]) - 1  # index 0
        obj = prsl[1]
        ulx, uly = float(prsl[2]), float(prsl[3])
        lrx, lry = float(prsl[4]), float(prsl[5])
        # t1, t2, t3, t4, t5, t6 = float(prsl[6]),float(prsl[7]),float(prsl[8]),float(prsl[9]),float(prsl[10]),float(prsl[11])
        # Save orientations and positions of joints into ORI and P matrices
        UL[num_obj][fr] = ulx, uly
        LR[num_obj][fr] = lrx, lry


def parse_videos(video_info, DATASET_DIR):
    rel_table = pd.DataFrame()
    pd_id = 0
    VIDEOS = [DATASET_DIR + 'images/'+ data['subject'] + "/" + data['activity'] + "/" + data['task'] + "/" for data in video_info]
    ANNOTATIONS = [[DATASET_DIR + "enhanced_annotations/" + data['subject'][:9] + "annotations/" + data[
        'activity'] + "/" + object_file for object_file in os.listdir(
        DATASET_DIR + "enhanced_annotations/" + data['subject'][:9] + "annotations/" + data['activity'] + "/") if
                    object_file.startswith(data['task'] + "_obj")] for data in video_info]

    for v in range(len(VIDEOS)):
        print(str(v) + " out of " + str(len(VIDEOS)))
        rel_table = pd.DataFrame()
        video = VIDEOS[v]
        depth_video = VIDEOS[v]
        annotations = ANNOTATIONS[v]
        subject, activity, task = video_info[v]['subject'], video_info[v]['activity'], video_info[v]['task']
        # print("Subject " + str(subject))
        # print("task " + str(task))
        # print("activity " + str(activity))
        video_length = len(os.listdir(video)) / 2
        number_of_objects = len(annotations)
        UL = np.zeros(((number_of_objects, video_length, 2)))
        LR = np.zeros(((number_of_objects, video_length, 2)))
        for nmo in range(0, number_of_objects):
            pathobj = annotations[nmo]
            file = open(pathobj, 'r')
            frame_lines = file.readlines()
            if len(frame_lines) > video_length:
                frame_lines = frame_lines[:-1]
            objects(frame_lines, nmo, UL, LR)

        for frame in range(0, video_length):
            img, dimg = read_image(video, depth_video, frame)
            if frame == 150:
                print("mid")
            for o in range(number_of_objects):
                cv2.line(img, (int(UL[o][frame][0]), int(UL[o][frame][1])),
                         (int(LR[o][frame][0]), int(UL[o][frame][1])), (255, 255, 255), 2)
                cv2.line(img, (int(UL[o][frame][0]), int(UL[o][frame][1])),
                         (int(UL[o][frame][0]), int(LR[o][frame][1])), (255, 255, 255), 2)
                cv2.line(img, (int(LR[o][frame][0]), int(LR[o][frame][1])),
                         (int(LR[o][frame][0]), int(UL[o][frame][1])), (255, 255, 255), 2)
                cv2.line(img, (int(LR[o][frame][0]), int(LR[o][frame][1])),
                         (int(UL[o][frame][0]), int(LR[o][frame][1])), (255, 255, 255), 2)
                # print("----pos---")
                # print("x of "+ str(o) + " at " + str(frame) + ": " + str((int(UL[o][frame][0])+int(LR[o][frame][0]))/2))
                # print("y of "+ str(o) + " at " + str(frame) + ": " + str((int(LR[o][frame][1])+int(UL[o][frame][1]))/2))
                # print("----size---")
                # print("x of "+ str(o) + " at " + str(frame) + ": " + str(abs((int(UL[o][frame][0])-int(LR[o][frame][0]))/2)))
                # print("y of "+ str(o) + " at " + str(frame) + ": " + str(abs((int(LR[o][frame][1])-int(UL[o][frame][1]))/2)))
                #
                rel_table.append(pd.Series(name=str(pd_id)))
                x_pos = (int(UL[o][frame][0])+int(LR[o][frame][0]))/2
                y_pos = (int(LR[o][frame][1])+int(UL[o][frame][1]))/2
                x_size = abs((int(UL[o][frame][0])-int(LR[o][frame][0]))/2) #length?
                y_size = abs((int(LR[o][frame][1])-int(UL[o][frame][1]))/2) #width?
                rel_table.loc[str(pd_id),"x_pos"] = x_pos
                rel_table.loc[str(pd_id),"y_pos"] = y_pos
                rel_table.loc[str(pd_id),"x_size"] = x_size
                rel_table.loc[str(pd_id),"y_size"] = y_size
                rel_table.loc[str(pd_id),"object"] = o
                rel_table.loc[str(pd_id),"timestamp"] = frame
                rel_table.loc[str(pd_id),"subject"] = subject
                rel_table.loc[str(pd_id),"task"] = task
                rel_table.loc[str(pd_id),"activity"] = activity

                #print(rel_table.tail())
                pd_id +=1
                '''
                print("----first----")
                print(int(UL[o][frame][0]))
                print(int(UL[o][frame][1]))
                print(int(LR[o][frame][0]))
                print(int(UL[o][frame][1]))
                print("x: " + str((int(UL[o][frame][0])+int(LR[o][frame][0]))/2))
                print("y: " + str((int(UL[o][frame][1])+int(UL[o][frame][1]))/2))
                print("----second----")
                print(int(UL[o][frame][0]))
                print(int(UL[o][frame][1]))
                print(int(UL[o][frame][0]))
                print(int(LR[o][frame][1]))
                print("x: " + str((int(UL[o][frame][0])+int(UL[o][frame][0]))/2))
                print("y: " + str((int(LR[o][frame][1])+int(UL[o][frame][1]))/2))
                print("----third----")
                print(int(LR[o][frame][0]))
                print(int(LR[o][frame][1]))
                print(int(LR[o][frame][0]))
                print(int(UL[o][frame][1]))
                print("x: " + str((int(LR[o][frame][0])+int(LR[o][frame][0]))/2))
                print("y: " + str((int(LR[o][frame][1])+int(UL[o][frame][1]))/2))
                print("----fourth----")
                print(int(LR[o][frame][0]))
                print(int(LR[o][frame][1]))
                print(int(UL[o][frame][0]))
                print(int(LR[o][frame][1]))
                print("x: " + str((int(LR[o][frame][0])+int(UL[o][frame][0]))/2))
                print("y: " + str((int(LR[o][frame][1])+int(LR[o][frame][1]))/2))
                '''
            #cv2.imshow('Depth', dimg)
            #cv2.imshow('RGB frame', img)
            k = cv2.waitKey(30) & 0xff  # pres ESC to skip video
            rel_path = '/media/mangafas/OS/Users/manga/Downloads/load/'
            file = 'rel_table_video_' + str(v) + '.csv'
            rel_table.to_csv(rel_path + file)
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fold = 'fold_test_load'
    csvfile = '/media/mangafas/OS/Users/manga/Downloads/' + fold + '.csv'
    DATASET_DIR = '/media/mangafas/OS/Users/manga/Downloads/load/images/'

    FOLD_VIDEOS = pd.read_csv(csvfile, delimiter=',', index_col=False, names=('subject', 'activity', 'task'),
                              dtype={'task': str})
    video_info = [{'subject': row['subject'], 'activity': row['activity'], 'task': row['task']} for index, row in
                  FOLD_VIDEOS.iterrows()]
    parse_videos(video_info, DATASET_DIR)
