import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import itertools
import matplotlib.animation as animate

df = pd.read_csv('active_china.csv')
df_total = pd.read_csv('total_china.csv')
df_world = pd.read_csv('active_world.csv')
import seaborn as sns

sns.set()

# fig = plt.figure()
shanghai_list = list(df['Shanghai'])
print (shanghai_list)
bins  = len(shanghai_list)
x_bins = np.arange(0,bins,1)
# Writer = animate.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)



import matplotlib.pyplot as plt
from matplotlib import animation



def gen_keylist(dict):
    return [keys for keys in dict]

fig=plt.figure()

n=17 #Number of frames
x=range(1,6)
barcollection = plt.bar(x_bins,shanghai_list)
key_list = gen_keylist(df)
print (key_list)
y_tick = np.arange(0,1100,100)
def animate(i):
    y = list(df[key_list[i]])
    plt.xlabel("Days")
    plt.ylabel("Active Cases")
    plt.yticks(y_tick)
    plt.title(key_list[i])
    for k, b in enumerate(barcollection):
        b.set_height(y[k])

anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n,
                             interval=1000)
#anim.save('china_animation.mp4', writer=writer)
plt.show()

import plotly.express as px

df2 = px.data.gapminder()
print (df2)
#fig = px.bar(df, x = "Days", y="Acive Cases", color="Days", animation_frame=)

print (df)
def create_new_dict(df):
    dict={}
    key_list = []
    case_list=[]
    for keys in df:
        key_list.append(keys)
        case_list.append(list(df[keys]))
        print (keys)

    func = lambda x: list(itertools.chain.from_iterable(itertools.repeat(i,48) for i in x))
    new_key_list = func(key_list)
    #print (new_key_list)
    case_list = [item for sublist in case_list for item in sublist]

    dict['Province'] = new_key_list
    days = list(np.arange(1,49,1))
    #print (days)
    days = (days)*len(key_list)
    #print (len(case_list), len(new_key_list), len(days))
    dict['Days'] = days
    dict['Active_cases'] = case_list
    return dict, case_list, new_key_list, days

dict, case_list, new_key_list, days = create_new_dict(df)
new_df = pd.DataFrame.from_dict(dict)
print (new_df)

fig = px.bar(new_df, x="Days", y = "Active_cases", color="Days", animation_frame="Province", range_y= [0,1100])
fig.show()
