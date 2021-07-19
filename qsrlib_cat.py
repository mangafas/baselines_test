#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace
import argparse
import pandas as pd

def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message):
    #print("---")
    #print("Response is:")
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        foo = str(t) + ": "
        for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                        qsrlib_response_message.qsrs.trace[t].qsrs.values()):
            foo += str(k) + ":" + str(v.qsr) + "; "
       # print(foo)

def event_table(which_qsr, qsrlib_response_message,video):
    df = pd.DataFrame()
    conv = { "dc": 0,
             "po": 1,
             "pp": 2,
             "ppi":2}

    # t:- timestamp
    # k: pair of objects - format: o1,o2
    # v.qsr: dictionary containing the  type of qualitive spatial representation and the relation at timestamp t between objects k - format: {'rcc4':'oop'}

    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        df.append(pd.Series(name=int(t)))

        for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                        qsrlib_response_message.qsrs.trace[t].qsrs.values()):
            df.loc[int(t),str(k)] = conv[str(v.qsr['rcc4'])]
            #print(df)
            if t>0:
                if df.loc[int(t),str(k)] != df.loc[int(t)-1,str(k)]:
                    print("strk: " + str(k))
                    df.loc[int(t),"match"] = 1
    df.loc[0,"match"] = 1
    df = df[df['match'].notna()]
    df = df.fillna(9)
    df = df.drop(['match'],axis=1)
    df_t = df.T
    #print(df)
    #print(df_t)
    #print(video)
    save_str = "/media/mangafas/OS/Users/manga/Downloads/load/event_table/event_table_" + str(video) +".csv"
    df.to_csv(save_str)
# change below here
def create_objects(video,world):
    #vidobj = {1:11, 2:6, 10:8, 11:10, 18:8} -- only for load/watch-n-patch
    read_str_path = '/media/mangafas/OS/Users/manga/Downloads/load/mask_tracks/Subject1/'

    num_of_objects = vidobj[video]

    for obj in range(num_of_objects):
    	read_srt_file = 'vid' + str(video) + '_' + str(obj)  + '.csv'
    	vid = pd.read_csv(read_str_path+read_srt_file)
        rows = list(vid.index)
        o = []
        for i in range(rows[0]+obj,rows[-1]-num_of_objects+obj,num_of_objects):
            obj_str = "o" + str(obj)
            o.append(Object_State(name=obj_str,timestamp=vid.loc[i,"frame"],x=vid.loc[i,"x_min"],y=vid.loc[i,"y_min"],xsize=vid.loc[i,"x_max"]-vid.loc[i,"x_min"],ysize=vid.loc[i,"y_max"]-vid.loc[i,"y_min"]))
        world.add_object_state_series(o)
        print("obj " + str(obj))
    return world

def num_object(vid):
    rows = list(vid.index)
    for i in range(rows[0]+1,rows[-1]+1):
        if int(vid.loc[i,"object"])==0:
            return int(vid.loc[i-1,"object"])

if __name__ == "__main__":
    # ****************************************************************************************************
    # create a QSRlib object if there isn't one already
    qsrlib = QSRlib()

    # ****************************************************************************************************
    # parse command line arguments
    options = sorted(qsrlib.qsrs_registry.keys())
    print(options)
    parser = argparse.ArgumentParser()
    parser.add_argument("qsr", help="choose qsr: %s" % options, type=str)
    args = parser.parse_args()
    if args.qsr in options:
        which_qsr = args.qsr
    else:
        raise ValueError("qsr not found, keywords: %s" % options)

    world = World_Trace()

    # ****************************************************************************************************
    # make some inputdata
    '''

    o1 = [Object_State(name="o1", timestamp=0, x=0., y=0., xsize=1., ysize=1.),
          Object_State(name="o1", timestamp=1, x=1., y=1., xsize=1., ysize=1.),
          Object_State(name="o1", timestamp=2, x=1., y=1., xsize=2., ysize=2.)]
         # Object_State(name="o1", timestamp=3, x=0., y=2., xsize=5., ysize=8.),
         # Object_State(name="o1", timestamp=4, x=2., y=2., xsize=5., ysize=8.),
         # Object_State(name="o1", timestamp=5, x=2., y=0., xsize=5., ysize=8.),
         # Object_State(name="o1", timestamp=6, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o1", timestamp=7, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o1", timestamp=8, x=3., y=3., xsize=5.2, ysize=8.5)]


    o2 = [Object_State(name="o2", timestamp=0, x=0., y=2., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=1, x=2., y=2., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=2, x=6., y=6., xsize=3., ysize=3.)]
         # Object_State(name="o2", timestamp=3, x=3., y=3., xsize=5.2, ysize=8.5),
         #Object_State(name="o2", timestamp=4, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o2", timestamp=5, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o2", timestamp=6, x=1., y=1., xsize=5., ysize=8.),
         # Object_State(name="o2", timestamp=7, x=1., y=1., xsize=5., ysize=8.),
         # Object_State(name="o2", timestamp=8, x=1., y=1., xsize=5., ysize=8.)]

    o3 = [Object_State(name="o3", timestamp=0, x=3., y=3., xsize=1, ysize=1),
          Object_State(name="o3", timestamp=1, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o3", timestamp=2, x=3, y=3., xsize=2, ysize=2)]
         # Object_State(name="o3", timestamp=3, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o3", timestamp=4, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o3", timestamp=5, x=3., y=3., xsize=5.2, ysize=8.5),
         # Object_State(name="o3", timestamp=6, x=1., y=1., xsize=5., ysize=8.),
         # Object_State(name="o3", timestamp=7, x=1., y=1., xsize=5., ysize=8.),
         # Object_State(name="o3", timestamp=8, x=1., y=0., xsize=1., ysize=1.)]

    #       Object_State(name="o3", timestamp=3, x=1., y=4., xsize=5.2, ysize=8.5),
    #       Object_State(name="o3", timestamp=4, x=0., y=4., xsize=5.2, ysize=8.5)]

    world.add_object_state_series(o1)
    world.add_object_state_series(o2)
    world.add_object_state_series(o3)
    '''
    # ****************************************************************************************************
    # dynammic_args = {'argd': {"qsr_relations_and_values" : {"Touch": 0.5, "Near": 6, "Far": 10}}}
    # make a QSRlib request message
    #dynammic_args = {"qtcbs": {"no_collapse": True, "quantisation_factor":0.01, "validate":False, "qsrs_for":[("o1","o2")] }}
    dynammic_args={"tpcc":{"qsrs_for":[("o0","o1","o2")] }}



    videos = [1,2,10,11,18]

    for video in videos:
        world = World_Trace()
        world = create_objects(video,world)
        print("video: " + str(video))
        qsrlib_request_message = QSRlib_Request_Message(which_qsr, world, dynammic_args)
        # request your QSRs
        qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
        # ****************************************************************************************************
        # print out your QSRs
        # pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message)
        pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message)
        event_table(which_qsr, qsrlib_response_message,video)
