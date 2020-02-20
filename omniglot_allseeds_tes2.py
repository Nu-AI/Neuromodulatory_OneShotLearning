



import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import click
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle
import pdb
import time
import skimage
from skimage import transform
import os
import platform


# import qtorch
# from qtorch.quant import fixed_point_quantize

#from qtorch.optim import OptimL


import matplotlib.pyplot as plt
import glob

import omniglot

from omniglot import Network



np.set_printoptions(precision=4)


defaultParams = {

    'nbclasses': 5,
    'nbshots': 1,  # Number of 'shots' in the few-shots learning
    'prestime': 1,
    'nbf' : 64    ,
    'prestimetest': 1,
    'interpresdelay': 0,
    'imagesize': 31,    # 28*28
    'nbiter': 10000000,
    'learningrate': 1e-5,
    'print_every': 10,
    'rngseed':0,
    'quantization': 0
}
NBTESTCLASSES = 100




#ttype = torch.FloatTensor;
ttype = torch.cuda.FloatTensor;


# Generate the full list of inputs, labels, and the target label for an episode
def generateInputsLabelsAndTarget(params, imagedata, test=False):

    inputT = np.zeros((params['nbsteps'], 1, 1, params['imagesize'], params['imagesize']))    #inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w
    labelT = np.zeros((params['nbsteps'], 1, params['nbclasses']))      #labelTensor, initially in numpy format...

    patterns=[]
    if test:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES, len(imagedata)))[:params['nbclasses']]  # Which categories to use for this *testing* episode?
    else:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES))[:params['nbclasses']]  # Which categories to use for this *training* episode?

    cats = np.random.permutation(cats)
    #print(cats)


    rots = np.random.randint(4, size=len(imagedata))


    testcat = random.choice(cats) # select the class on which we'll test in this episode
    unpermcats = cats.copy()


    location = 0
    for nc in range(params['nbshots']):
        np.random.shuffle(cats)   # Presentations occur in random order
        for ii, catnum in enumerate(cats):
            #print(catnum)
            p = random.choice(imagedata[catnum])
            for nr in range(rots[catnum]):
                p = np.rot90(p)
            p = skimage.transform.resize(p, (31, 31))
            for nn in range(params['prestime']):


                inputT[location][0][0][:][:] = p[:][:]
                labelT[location][0][np.where(unpermcats == catnum)] = 1
                #if nn == 0:
                #    print(labelT[location][0])
                location += 1
            location += params['interpresdelay']

    # Inserting the test character
    p = random.choice(imagedata[testcat])
    for nr in range(rots[testcat]):
        p = np.rot90(p)
    p = skimage.transform.resize(p, (31, 31))
    for nn in range(params['prestimetest']):
        inputT[location][0][0][:][:] = p[:][:]
        location += 1

    # Generating the test label
    testlabel = np.zeros(params['nbclasses'])
    testlabel[np.where(unpermcats == testcat)] = 1


    assert(location == params['nbsteps'])

    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    labelT = torch.from_numpy(labelT).type(ttype)
    targetL = torch.from_numpy(testlabel).type(ttype)

    return inputT, labelT, targetL


def train(paramdict=None):
    print("Initializing random seeds")
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    print("Starting testing...")
    params = {}

    params.update(defaultParams)
    if paramdict:
        params.update(paramdict)

    #pdb.set_trace()


    print("Loading Omniglot data...")
    imagedata = []
    imagefilenames=[]
    #for basedir in ('./omniglot-master/python/images_background/',
    #                './omniglot-master/python/images_evaluation/'):
    #for basedir in ('C:\\Users\Anurag\PycharmProjects\Thesis_codes\omniglot-master\python\images_background',
    #                'C:\\Users\Anurag\PycharmProjects\Thesis_codes\omniglot-master\python\images_evaluation'):
    # for basedir in ('C:\\Users\Anurag\PycharmProjects\Thesis_codes\omniglot-master\python\images_background/',
    #                 'C:\\Users\Anurag\PycharmProjects\Thesis_codes\omniglot-master\python\images_evaluation/'):
    for basedir in ('./omniglot_dataset/omniglot/python/images_background/',
                    './omniglot_dataset/omniglot/python/images_evaluation/'):
        alphabetdirs = glob.glob(basedir+'*')
        print(alphabetdirs[:4],"meoww")
        for alphabetdir in alphabetdirs:
            chardirs = glob.glob(alphabetdir+"/*")
            #print(chardirs)
            for chardir in chardirs:
                chardata = []
                charfiles = glob.glob(chardir+'/*')
                #print (charfiles,"These are the charfiles")
                for fn in charfiles:
                    print(fn,"the file data")
                    filedata = plt.imread(fn)
                    print(len(filedata))
                    chardata.append(filedata)
                imagedata.append(chardata)
                imagefilenames.append(fn)

    # imagedata[CharactertNumber][FileNumber] -> numpy(105,105)
    np.random.shuffle(imagedata)
    print(len(imagedata))
    print(imagedata[1][2].shape)
    print("Data loaded!")


    successrates = []
    totaliter = 0
    totalmistakes = 0

    for myseed in range(8,9,1):


        #suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.75_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+str(myseed)
        suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+str(myseed)+"_5000000"
        with open('./results/results'+suffix+'.dat', 'rb') as fo:
            tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
            tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
            tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))

            tmplss = pickle.load(fo)
            paramdictLoadedFromFile = pickle.load(fo)
            # if (params['quantization']):
            #         qtmpw = fixed_point_quantize(tmpw, 16,14)
            #         qtmpalpha = fixed_point_quantize(tmpalpha, 16, 14)
            #         qtmpeta  = fixed_point_quantize(tmpeta, 16, 14)
            #         qparamdictLoadFromFile = fixed_point_quantize(paramdictLoadedFromFile, 16, 14)


        params.update(paramdictLoadedFromFile)
        # if (params['quantization']):
        #     params.update(qparamdictLoaddFromFile)

        print("Initializing network")
        net = Network(params)
        #net.cuda()
        params_conv = sum(p.numel() for p in net.parameters())
        print ("The number of params ", params_conv, "***********************************\n")
        print ("Size of all optimized parameters:", [x.size() for x in net.parameters()])
        allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
        print ("Size (numel) of all optimized elements:", allsizes)
        print ("Total size (numel) of all optimized elements:", sum(allsizes))


        print("Passed params: ", params)
        print(platform.uname())
        sys.stdout.flush()
        params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode

        net.load_state_dict(torch.load('./torchmodels/torchmodel'+suffix + '.txt'))


        #torch.quantization.quantize_dynamic(net.cpu(),dtype = torch.qint8 )

        params['nbiter'] = 100



        total_loss = 0.0
        #print("Initializing optimizer")
        ##optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=params['learningrate'])
        #optimizer = torch.optim.Adam(net.parameters(), lr=params['learningrate'])
        all_losses = []
        #print_every = 20
        nowtime = time.time()
        print("Starting episodes...")
        sys.stdout.flush()

        nbmistakes = 0

        for numiter in range(params['nbiter']):

            hebb = net.initialZeroHebb()
            #optimizer.zero_grad()


            is_test_step = 1

            inputs, labels, target = generateInputsLabelsAndTarget(params, imagedata, test=is_test_step)


            for numstep in range(params['nbsteps']):
                y, hebb = net(Variable(inputs[numstep], requires_grad=False), Variable(labels[numstep], requires_grad=False), hebb)

            #loss = (y[0] - Variable(target, requires_grad=False)).pow(2).sum()
            criterion = torch.nn.BCELoss()
            loss = criterion(y[0], Variable(target, requires_grad=False))

            if params['quantization']:
                qy = fixed_point_quantize(y, 16, 14)
                qhebb = fixed_point_quantize(hebb, 16, 14)
            #if is_test_step == False:
            #    loss.backward()
            #    optimizer.step()
            print (loss.item())
            lossnum = loss.item()       # loss.data() was not working ffor this pytorch patch
            #total_loss  += lossnum
            if is_test_step:
                total_loss = lossnum

            if is_test_step: # (numiter+1) % params['print_every'] == 0:

                print(numiter, "====")
                td = target.cpu().numpy()
                yd = y.data.cpu().numpy()[0]
                #print("y: ", yd[:10])
                #print("target: ", td[:10])
                if np.argmax(td) != np.argmax(yd):
                    print("Mistake!")
                    nbmistakes += 1
                #print("target: ", target.unsqueeze(0)[0][:10])
                absdiff = np.abs(td-yd)
                #print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
                #print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
                #print inputs[numstep]
                previoustime = nowtime
                nowtime = time.time()
                #print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)
                #total_loss /= params['print_every']
                #print("Mean loss over last", params['print_every'], "iters:", total_loss)
                #print("Loss on single withheld-data episode:", lossnum)
                all_losses.append(total_loss)
                #print ("Eta: ", net.eta.data.cpu().numpy())
                sys.stdout.flush()
                sys.stderr.flush()

                total_loss = 0

        all_losses = np.array(all_losses)
        print("Mean / std all losses :", np.mean(all_losses), np.std(all_losses))
        print("1st Quartile / median / 3rd Quartile all losses :", np.percentile(all_losses, 25), np.percentile(all_losses, 50), np.percentile(all_losses, 75))
        print("Max of all losses :", np.max(all_losses))
        print("Nb of mistakes :", nbmistakes, "over", numiter+1, "trials - (", 100.0 - 100.0 * nbmistakes / (numiter+1), " % correct )")
        successrates.append(100.0 - 100.0 * nbmistakes / (numiter+1))
        totalmistakes += nbmistakes
        totaliter += params['nbiter']

    print ("Mean / stdev success rate across runs: ", np.mean(successrates), np.std(successrates))
    totalsuccessrate = 1.0 - totalmistakes / totaliter
    pointestCI = 1.96 * np.sqrt(totalsuccessrate * (1.0 - totalsuccessrate) / totaliter)
    print ("Success % across all trials (95% CI point estimate):", 100.0 * totalsuccessrate, "+/-", 100.0 * pointestCI)
    print (totalmistakes, "mistakes out of ", totaliter, "trials")


    print ("Median success rate across runs: ", np.median(successrates))


@click.command()
@click.option('--nbclasses', default=defaultParams['nbclasses'])
@click.option('--nbshots', default=defaultParams['nbshots'])
@click.option('--prestime', default=defaultParams['prestime'])
@click.option('--prestimetest', default=defaultParams['prestimetest'])
@click.option('--interpresdelay', default=defaultParams['interpresdelay'])
@click.option('--nbiter', default=defaultParams['nbiter'])
@click.option('--learningrate', default=defaultParams['learningrate'])
@click.option('--print_every', default=defaultParams['print_every'])
@click.option('--rngseed', default=defaultParams['rngseed'])
def main(nbclasses, nbshots, prestime, prestimetest, interpresdelay, nbiter, learningrate, print_every, rngseed):
    train(paramdict=dict(click.get_current_context().params))
    #print(dict(click.get_current_context().params))

if __name__ == "__main__":
    #train()
    main()
