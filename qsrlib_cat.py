!/usr/bin/env python                                                                                                                                                                                                                         # -*- coding: utf-8 -*-                                                                                                                                                                                                                      from __future__ import print_function, division
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace                                                                                                                                                                                  import argparse                                                                                                                                                                                                                              import pandas as pd                                                                                                                                                                                                                                                                                                                                                                                                                                                                       def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message):                                                                                                                                                                            print(which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
          + " and received at " + str(qsrlib_response_message.req_received_at)                                                                                                                                                                         + " and finished at " + str(qsrlib_response_message.req_finished_at))                                                                                                                                                                  print("---")                                                                                                                                                                                                                                 print("Response is:")
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        foo = str(t) + ": "                                                                                                                                                                                                                          for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),                                                                                                                                                                                           qsrlib_response_message.qsrs.trace[t].qsrs.values()):                                                                                                                                                                            foo += str(k) + ":" + str(v.qsr) + "; "                                                                                                                                                                                                  print(foo)                                                                                                                                                                                                                           agg = []                                                                                                                                                                                                                                     ag = []                                                                                                                                                                                                                                      def create_at(which_qsr, qsrlib_response_message):                                                                                                                                                                                               df = pd.DataFrame()                                                                                                                                                                                                                          conv = { "NaN": -1,
             "dc": 0,                                                                                                                                                                                                                                     "po": 1,                                                                                                                                                                                                                                     "pp": 2,
             "ppi":2}                                                                                                                                                                                                                            df2 = pd.DataFrame(columns= ['o1,o2','o1,o3','o2,o3'])                                                                                                                                                                                       i=0                                                                                                                                                                                                                                          b = 1                                                                                                                                                                                                                                        for t in qsrlib_response_message.qsrs.get_sorted_timestamps():                                                                                                                                                                                   df.append(pd.Series(name=int(t)))                                                                                                                                                                                                            for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),                                                                                                                                                                                           qsrlib_response_message.qsrs.trace[t].qsrs.values()):                                                                                                                                                                            ag.append(str(k))                                                                                                                                                                                                                            df.loc[int(t),str(k)] = conv[str(v.qsr['rcc4'])]                                                                                                                                                                                             if t>0:                                                                                                                                                                                                                                          if df.loc[int(t),str(k)] != df.loc[int(t)-1,str(k)]:
                    print("here")                                                                                                                                                                                                                                df.loc[int(t),"match"] = 1
    df.loc[0,"match"] = 1                                                                                                                                                                                                                            #df.iloc[i] = agg
        #agg.clear()                                                                                                                                                                                                                            # for i in ag:                                                                                                                                                                                                                                   # print(i)                                                                                                                                                                                                                                                                                                                                                                                                                                                                             df = df[df['match'].notna()]                                                                                                                                                                                                                 df_t = df.T
    print(df)                                                                                                                                                                                                                                    print(df_t)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
if __name__ == "__main__":                                                                                                                                                                                                                       # ****************************************************************************************************
    # create a QSRlib object if there isn't one already                                                                                                                                                                                          qsrlib = QSRlib()
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

    # ****************************************************************************************************
    # make some input data
    world = World_Trace()
    o1 = [Object_State(name="o1", timestamp=0, x=0., y=0., xsize=1., ysize=1.),
          Object_State(name="o1", timestamp=1, x=1., y=1., xsize=5., ysize=8.),
          Object_State(name="o1", timestamp=2, x=1., y=1., xsize=5., ysize=8.),
          Object_State(name="o1", timestamp=3, x=0., y=2., xsize=5., ysize=8.),
          Object_State(name="o1", timestamp=4, x=2., y=2., xsize=5., ysize=8.),
          Object_State(name="o1", timestamp=5, x=2., y=0., xsize=5., ysize=8.),
          Object_State(name="o1", timestamp=6, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o1", timestamp=7, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o1", timestamp=8, x=3., y=3., xsize=5.2, ysize=8.5)
]

    o2 = [Object_State(name="o2", timestamp=0, x=0., y=2., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=1, x=2., y=2., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=2, x=2., y=0., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=3, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o2", timestamp=4, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o2", timestamp=5, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o2", timestamp=6, x=1., y=1., xsize=5., ysize=8.),
          Object_State(name="o2", timestamp=7, x=1., y=1., xsize=5., ysize=8.)]
         # Object_State(name="o2", timestamp=8, x=1., y=1., xsize=5., ysize=8.)]

        #   Object_State(name="o2", timestamp=3, x=1., y=1., xsize=5., ysize=8.),
        #   Object_State(name="o2", timestamp=4, x=1., y=2., xsize=5., ysize=8.)]

    o3 = [Object_State(name="o3", timestamp=0, x=3., y=3., xsize=1, ysize=1),
          Object_State(name="o3", timestamp=1, x=3., y=3., xsize=5.2, ysize=8.5),                                                                                                                                                                      Object_State(name="o3", timestamp=2, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o3", timestamp=3, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o3", timestamp=4, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o3", timestamp=5, x=3., y=3., xsize=5.2, ysize=8.5),
          Object_State(name="o3", timestamp=6, x=1., y=1., xsize=5., ysize=8.),
          Object_State(name="o3", timestamp=7, x=1., y=1., xsize=5., ysize=8.),
          Object_State(name="o3", timestamp=8, x=1., y=0., xsize=1., ysize=1.)]
    #       Object_State(name="o3", timestamp=3, x=1., y=4., xsize=5.2, ysize=8.5),
    #       Object_State(name="o3", timestamp=4, x=0., y=4., xsize=5.2, ysize=8.5)]

    world.add_object_state_series(o1)
    world.add_object_state_series(o2)
    world.add_object_state_series(o3)

    # ****************************************************************************************************
    # dynammic_args = {'argd': {"qsr_relations_and_values" : {"Touch": 0.5, "Near": 6, "Far": 10}}}
    # make a QSRlib request message
    #dynammic_args = {"qtcbs": {"no_collapse": True, "quantisation_factor":0.01, "validate":False, "qsrs_for":[("o1","o2")] }}
    dynammic_args={"tpcc":{"qsrs_for":[("o1","o2")] }}

    qsrlib_request_message = QSRlib_Request_Message(which_qsr, world, dynammic_args)
    print(qsrlib_request_message)
    # request your QSRs
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
    
     # ****************************************************************************************************
    # print out your QSRs
    # pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message)

    create_at(which_qsr, qsrlib_response_message)
