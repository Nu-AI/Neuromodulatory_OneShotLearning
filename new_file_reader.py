import numpy as np
import glob
import matplotlib.pyplot as plt
from Omniglot_plasticity.loss import *
import seaborn as sns


groupnames = glob.glob('./loss*rngseed_0.txt')
for numgroup, groupname in enumerate(groupnames):
    g = groupname[:-6] + "*"
#sns.set()
def mavg(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.ion()
# plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts
plt.figure()

maxl = 100

fnames = ['loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_1000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_1500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_2000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_2500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_3000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_3500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_4000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_4500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_0_5000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_1000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_1500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_2000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_2500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_3000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_3500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_4000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_4500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_1_5000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_1000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_1500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_2000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_2500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_3000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_3500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_4000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_4500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_10_5000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_1000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_1500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_2000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_2500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_3000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_3500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_4000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_4500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_3_5000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_1000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_1500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_2000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_2500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_3000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_3500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_4000000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_4500000.txt',
          'loss\loss_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_5_5000000.txt']

def file_split(fname):
    z = np.genfromtxt(fname,dtype='str')
    z = np.char.split(z,sep="(")


    #print (len(z[:,0]),z[:,0])
    temp = z[:,0]
    final_loss = np.zeros(len(z[:,0]))
    for i in range(len(z[:,0])):
        new_temp = np.char.split(temp[i],sep=",\s")
        new_temp2= np.char.split(new_temp[1],sep=",")
        final_loss[i] = float(new_temp2[0][0])
        #print(new_temp2[0][0],final_loss[i])

    #print (final_loss)
    #np.savetxt('sample_loss_1.txt',final_loss)
    return final_loss
    #print (new_temp2)

#new_temp = np.char.split(temp,sep=",")
#print(z)

for j in range(4):
    lgts = []
    fulllosses = []
    losses = []
    full_loss_array = np.array([])
    z = np.array([])
    for i in range(11):
        z = file_split(fnames[i+11*j])
        full_loss_array = np.append(full_loss_array,z)
        z = z[::10]  # Decimation
        # z = mavg(z, 100)
        # print(z,"the value of z")
        lgts.append(len(z))
        print(len(z))
        fulllosses.append(z)
        print (full_loss_array.size)
    minlen = min(lgts)
    print("the minlen is",minlen)
    for z in fulllosses:
        losses.append(z[:minlen])
    losses = np.array(losses)
    print ("the shape of losses is", np.shape(losses))
    print ("the losses are", losses)
    meanl = np.mean(losses, axis=0)
    print(np.shape(meanl),"the mean shape")
    stdl = np.std(losses, axis=0)

    medianl = np.median(losses, axis=0)
    print(np.shape(medianl),"the median shape")
    q1l = np.percentile(losses, 25, axis=0)
    print(np.shape(q1l),"the q1l shape")
    q3l = np.percentile(losses, 75, axis=0)
    print(np.shape(q3l),"the q3l shape")

    highl = np.max(losses, axis=0)
    lowl = np.min(losses, axis=0)
    # highl = meanl+stdl
    # lowl = meanl-stdl

    myls = '-'
    # if numgroup >= 8:
    #     myls = '--'
    xx = range(len(meanl))

    # xticks and labels
    if len(meanl) > maxl:
        maxl = len(meanl)
    print(maxl,"the max1 value")
    # plt.plot(mavg(meanl, 100), label=g) #, color='blue')
    # plt.fill_between(xx, lowl, highl,alpha=0.5)
    # plt.fill_between(xx, q1l, q3l,alpha=0.5)
    # plt.plot(meanl) #, color='blue')

    #plt.plot(mavg(medianl, 10), label=g,ls=myls)  # , color='blue')  # mavg changes the number of points !

    # plt.plot(mavg(q1l, 100), label=g,alpha=.3) #, color='blue')
    # plt.plot(mavg(q3l, 100), label=g, alpha=.3) #, color='blue')
    plt.fill_between(xx, q1l, q3l,  alpha=.5)
    plt.plot(medianl) #, color='blue')
    #plt.plot(full_loss_array)

newmax= int(maxl/2)
plt.xlabel('No. of Episodes/100')
plt.ylabel('Median Loss')
xt = range(0, maxl, 10)
xtl = [str(50* i) for i in xt]  # 5000 = 500 episode per loss saving, plus the decimation above
plt.xticks(xt, xtl)
#plt.axis([0,100,0,4])
plt.tight_layout()
plt.savefig('new_loss_plot.png')

np.savetxt('sample_full_loss.txt',full_loss_array)
print ("the save is done")