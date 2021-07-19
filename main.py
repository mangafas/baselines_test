import pandas as pd
import numpy as np


# get semantic event chain
def chain(xica):
    cols = list(xica.columns)
    rows = list(xica.index)
    xic = pd.DataFrame()
    for row in rows:
        xic.append(pd.Series(name=str(row), dtype=int))
        c = 1
        xic.loc[str(row), 0] = int(xica.loc[str(row), 0])
        for i in range(1, len(cols)):
            if i == len(cols) - 1:
                if xica.loc[str(row), i] != xica.loc[str(row), i - 1]:
                    xic.loc[str(row), c] = int(xica.loc[str(row), i])
                    c += 1
            else:
                if xica.loc[str(row), i] != xic.loc[str(row), c - 1]:
                    xic.loc[str(row), c] = int(xica.loc[str(row), i])
                    c += 1
    return xic


# compress semantic event chain
def compress(xida, video):
    cols = list(xida.columns)
    rows = list(xida.index)
    # print(cols)
    # print(rows)
    xid = pd.DataFrame()
    for row in rows:
        rowm = row + '_' + str(video)
        xs = pd.Series(name=str(rowm), dtype=object)
        xid.append(xs)
        for i in range(len(cols) - 1):
            if np.isnan(xida.loc[str(row), i + 1]) and i == 0:
                xid.loc[str(rowm), i] = str(int(xida.loc[str(row), i]))
            elif not (np.isnan(xida.loc[str(row), i]) or np.isnan(xida.loc[str(row), i + 1])):
                temp = str(int(xida.loc[str(row), i])) + str(int(xida.loc[str(row), i + 1]))
                xid.loc[str(rowm), i] = temp
    return xid


def del_dup(rows):
    for row in rows:
        rev_str = "o" + str(row[-1]) + ",o" + str(row[1])
        if rev_str in rows:
            rows.remove(row)
    return rows


# get video similarity
def summ(sim_table):
    rows = list(sim_table.index)
    num_rows = sim_table[rows[0]].count()
    lis = []
    count = 0
    for row in rows:
        for row2 in rows:
            lis.append(sim_table.loc[str(row), str(row2)])
        count += max(lis)
        lis = []
    return count / num_rows

def del_dup(rows, video):
    new_rows = []
    for row in rows:
        rev_str = "o" + str(row[4]) + ",o" + str(row[1]) + '_' + str(video)
        if rev_str in rows:
            rows.remove(row)
    for row in rows:
        if row[1] > row[4]:
            r = list(row)
            temp = r[4]
            r[4] = r[1]
            r[1] = temp
            new_row = ''.join(r)
            new_rows.append(new_row)
        else:
            new_rows.append(row)
    new_rows = list(set(new_rows))
    new_rows.sort()
    return new_rows


# rows - first, columns - second
# get similarity between objects on 2 videos
def sim(data1, data2, video1, video2):
    rows1 = del_dup(list(data1.index), video1)
    rows2 = del_dup(list(data2.index), video2)
    #     str_add1 = '_'+ str(video1)
    #     str_add2 = '_'+ str(video2)
    #     rows1 = [s + str_add1 for s in rows1]
    #     rows2 = [s + str_add2 for s in rows2]
    temp = []
    tempd = pd.DataFrame(dtype='string')
    if len(rows1) < len(rows2):
        temp = rows1
        rows1 = rows2
        rows2 = temp
        temp = data1
        data1 = data2
        data2 = temp
    # print(rows1,rows2)
    # sim_table = pd.DataFrame(dtype='float64', columns=rows1,index=rows2)
    sim_table = pd.DataFrame(dtype='float64')
    #     print('-------------------')
    #     print(rows1,rows2)
    #     print('-------------------')
    #     print(sim_table)
    #     print('-------------------')
    #     #print(sim_table)
    perc = 0
    cur_perc = []
    # seems correct till here as we get the bigger row - row1 is always the biggest or equal - bigger means more objects
    for row1 in rows1:
        for row2 in rows2:
            num_cols1 = data1.count(axis=1)[str(row1)]
            num_cols2 = data2.count(axis=1)[str(row2)]
            if num_cols1 > num_cols2:
                big = num_cols1
                small = num_cols2
                bigd = data1
                smalld = data2
                bigr = row1
                smallr = row2
            else:
                big = num_cols2
                small = num_cols1
                bigd = data2
                smalld = data1
                bigr = row2
                smallr = row1
            # print(big, small, bigr, smallr)
            # print(smalld)
            # print(bigd)
            for window in range(big - small + 1):
                for winlen in range(small):
                    #                     print(bigr,window,winlen,window+winlen)
                    #                     print(bigd)
                    #                     print(bigd.loc[str(bigr),str(window+winlen)])
                    #                     print(smalld.loc[str(smallr),str(winlen)])
                    # if bigd.loc[str(bigr),str(window+winlen)] == smalld.loc[str(smallr),str(winlen)] and len(str(bigd.loc[str(bigr),str(window+winlen)])) == len(str(smalld.loc[str(smallr),str(winlen)])):
                    if bigd.loc[str(bigr), str(window + winlen)] == smalld.loc[str(smallr), str(winlen)]:
                        # print(str(bigd.loc[str(bigr),str(window+winlen)]) + " " + str(bigr))
                        # print(str(smalld.loc[str(smallr),str(winlen)])+ " " +str(smallr))

                        perc += 1
                cur_perc.append(perc)
                perc = 0
            if int(row2[6:]) > int(row1[6:]):
                sim_table.loc[str(row1), str(row2)] = max(cur_perc) / small
            else:
                sim_table.loc[str(row2), str(row1)] = max(cur_perc) / small
            cur_perc.clear()
    return sim_table


# object similarity table
objs = []
def obj_sim(sim_table, video, videop):
    objs = []
    # print(sim_table)
    pairs1 = list(sim_table.index)  # video 1
    pairs2 = list(sim_table.columns)  # video 2
    ext_str1 = "_" + str(video)
    ext_str2 = "_" + str(videop)
    # print(pairs1,pairs2)
    vid_1_obj = []
    vid_2_obj = []
    for obj in pairs1:
        vid_1_obj.append(obj[1])
        vid_1_obj.append(obj[4])
    for obj in pairs2:
        vid_2_obj.append(obj[1])
        vid_2_obj.append(obj[4])
    vid_1_obj = list(set(vid_1_obj))
    vid_2_obj = list(set(vid_2_obj))
    # print(vid_1_obj,vid_2_obj)
    temp = []
    tempp = []
    vid_1_obj = [s + ext_str1 for s in vid_1_obj]
    vid_2_obj = [s + ext_str2 for s in vid_2_obj]
    if len(vid_1_obj) <= len(vid_2_obj):
        temp = vid_1_obj
        tempp = pairs1
        vid_1_obj = vid_2_obj
        pairs1 = pairs2
        vid_2_obj = temp
        pairs2 = tempp
    # print(vid_2_obj)
    obj_tab = pd.DataFrame(0, dtype='float64', columns=vid_1_obj, index=vid_2_obj)
    print(obj_tab)
    for obj in vid_2_obj:
        # get the lowest of the two
        count = 0
        for pair2 in pairs2:
            # print(pair2)
            # print(obj)
            for pair1 in pairs1:
                # print(pair1,pair2)
                # print(obj, obj[0],obj[-2:])
                # print(pair1,obj,pair2)
                if len(videop) == 60000:
                    # why the change from pair2 to pair 1 ???
                    if obj[0] in pair2[:5] and obj[-8:] in pair2[:5]:
                        temp_obj1 = str(pair1[1]) + str(pair1[-8:])
                        print(temp_obj0)
                        temp_obj4 = str(pair1[4]) + str(pair1[-8:])
                        print(temp_obj0)
                        obj_tab.loc[obj, temp_obj1] += sim_table.loc[pair2, pair1]
                        # print(sim_table.loc[pair2,pair1])
                        obj_tab.loc[obj, temp_obj4] += sim_table.loc[pair2, pair1]
                    elif obj[0] in pair1[:5] and obj[-8:] in pair1[:5]:
                        temp_obj1 = str(pair2[1]) + str(pair2[-8:])
                        # print(temp_obj1)
                        temp_obj4 = str(pair2[4]) + str(pair2[-8:])
                        # print(temp_obj1)
                        obj_tab.loc[obj, temp_obj1] += sim_table.loc[pair2, pair1]
                        # print(sim_table.loc[pair2,pair1])
                        obj_tab.loc[obj, temp_obj4] += sim_table.loc[pair2, pair1]
                #                 elif len(videop):
                #                     if obj[0] in pair2[:5] and obj[-2:] in pair2[:5]:
                #                         temp_obj1 = str(pair1[1]) + str(pair1[-3:])
                #                         #print(temp_obj2)
                #                         temp_obj4 = str(pair1[4]) + str(pair1[-3:])
                #                         #print(temp_obj2)
                #                         obj_tab.loc[obj,temp_obj1] += sim_table.loc[pair2,pair1]
                #                         #print(sim_table.loc[pair2,pair1])
                #                         obj_tab.loc[obj,temp_obj4] += sim_table.loc[pair2,pair1]
                #                     elif obj[0] in pair1[:5] and obj[-2:] in pair1[:5]:
                #                         temp_obj1 = str(pair2[1]) + str(pair2[-3:])
                #                         #print(temp_obj3)
                #                         temp_obj4 = str(pair2[4]) + str(pair2[-3:])
                #                         #print(temp_obj3)
                #                         #print('here')
                #                         obj_tab.loc[obj,temp_obj1] += sim_table.loc[pair2,pair1]
                #                         #print(sim_table.loc[pair2,pair1])
                #                         obj_tab.loc[obj,temp_obj4] += sim_table.loc[pair2,pair1]
                else:
                    # print(videop,obj)
                    print(obj[0], pair2[:5], obj[1:], pair2[5:])
                    if obj[0] in pair2[:5] and obj[1:] in pair2[5:]:
                        print('here')
                        print(pair1)
                        temp_obj1 = str(pair2[1]) + str(pair2[5:])
                        print(temp_obj1)
                        temp_obj4 = str(pair2[4]) + str(pair2[5:])
                        print(temp_obj4)
                        # print('here')
                        if pair2 not in sim_table.index and pair2 not in sim_table.columns:
                            print('works')
                            continue
                        obj_tab.loc[temp_obj1, obj] += sim_table.loc[pair2, pair1]
                        # print(sim_table.loc[pair2,pair1])
                        obj_tab.loc[temp_obj4, obj] += sim_table.loc[pair2, pair1]
                    elif obj[0] in pair1[:5] and obj[1:] in pair1[5:]:
                        # continue
                        temp_obj5 = str(pair1[1]) + str(pair1[5:])
                        print(temp_obj5)
                        temp_obj6 = str(pair1[4]) + str(pair1[5:])
                        print(temp_obj6)
                        # print('here')
                        if pair2 not in sim_table.index and pair2 not in sim_table.columns:
                            print('works')
                            continue
                        obj_tab.loc[obj, temp_obj5] += sim_table.loc[pair2, pair1]
                        # print(sim_table.loc[pair2,pair1])
                        obj_tab.loc[obj, temp_obj6] += sim_table.loc[pair2, pair1]

    objs = list(set(objs))
    big_tab = pd.DataFrame(columns=objs, index=objs)
    write_str = r"C:\Users\manga\Downloads\load\big_table.csv"
    big_tab.to_csv(write_str)
    # print(len(vid_2_obj))
    obj_tab = obj_tab / ((len(vid_2_obj) - 1) * (len(vid_1_obj) - 1))
    return obj_tab

'''
uncomment for running sim_table function
vis = [1,2,10,11,18]
for video in range(3):
    for video2 in vis:
        read_str1 = r"C:\Users\manga\Downloads\load\compressed_table\com_table_" + str(video+1) + "_"+ str(video2)+".csv"
        data1 = pd.read_csv(read_str1,dtype='string')
        data1 = data1.set_index('Unnamed: 0')
        data1.index.name = None
        for videop in range(video+1,3):
            for videop2 in vis:
                if videop2 != video2:
                    read_str2 = r"C:\Users\manga\Downloads\load\compressed_table\com_table_" + str(videop+1) + "_"+ str(videop2)+".csv"
                    data2 = pd.read_csv(read_str2,dtype='string')
                    data2 = data2.set_index('Unnamed: 0')
                    data2.index.name = None
                    #print(str(video) + " and " + str(videop)) 
                    vidstr1 = str(video+1) + '_'+ str(video2)
                    vidstr2 = str(videop+1) +'_'+ str(videop2)
                    sim_table = sim(data1,data2,vidstr1,vidstr2)
                    write_str = r"C:\Users\manga\Downloads\load\sim_tables\sim_table_" + str(video+1) + '_'+ str(video2) + '_' + str(videop+1) +'_'+ str(videop2)+ ".csv"
                    print(write_str)
                    sim_table.to_csv(write_str)
                    
'''


for video in range(3):
    for video2 in vis:
        for videop in range(video+1,3):
            for videop2 in vis:
                if videop2!=video2:
                    vidstr1 = str(video+1) + '_'+ str(video2)
                    vidstr2 = str(videop+1) +'_'+ str(videop2)
                    read_str = r"C:\Users\manga\Downloads\load\sim_tables\sim_table_" + vidstr1 + '_'+ vidstr2 +  ".csv"
                    obj_table = pd.read_csv(read_str)
                    obj_table = obj_table.set_index('Unnamed: 0')
                    obj_table.index.name = None
                    pairs1 = list(obj_table.index) #video 1
                    pairs2 = list(obj_table.columns) #video 2
                    #print(pairs1,pairs2)
                    objs.extend(pairs1)
