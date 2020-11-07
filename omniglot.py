
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import click
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import pdb
import time
import skimage
from skimage import transform
from skimage import io
import os
import platform

import numpy as np
import glob





np.set_printoptions(precision=4)
defaultParams = {
    'activ': 'tanh',    # 'tanh' or 'selu'
    #'plastsize': 200,
    'rule': 'hebb',     # 'hebb' or 'oja'
    'alpha': 'free',   # 'free' of 'yoked' (if the latter, alpha is a single scalar learned parameter, shared across all connection)
    'steplr': 1e6,  # How often should we change the learning rate?
    'nbclasses': 5,
    'gamma': .666,  #
    'flare': 0,     # Whether or not the ConvNet has more features in higher channels
    'nbshots': 1,  # Number of 'shots' in the few-shots learning
    'prestime': 1,
    'nbf' : 64,  # Number of features. 128 is better (unsurprisingly) but had to keep 64 for fair comparison with other reports :P
    'prestimetest': 1,
    'ipd': 0,  # Inter-presentation delay
    'imgsize': 31,
    'nbiter': 5000000,
    'lr': 3e-5,
    'test_every': 500,
    'save_every': 10000,
    'rngseed':0
}
NBTESTCLASSES = 100

#ttype = torch.FloatTensor;
ttype = torch.cuda.FloatTensor


# Generate the full list of inputs, labels, and the target label for an episode well this is a common code took from the previous works for maintaining consistency with the inputs
def generateInputsLabelsAndTarget(params, imagedata, test=False):
    #print(("Input Boost:", params['inputboost']))
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['ipd']) * params['nbclasses']) + params['prestimetest']
    inputT = np.zeros((params['nbsteps'], 1, 1, params['imgsize'], params['imgsize']))    #inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w
    labelT = np.zeros((params['nbsteps'], 1, params['nbclasses']))      #labelTensor, initially in numpy format...

    patterns=[]
    if test:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES, len(imagedata)))[:params['nbclasses']]  # Which categories to use for this *testing* episode?
    else:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES))[:params['nbclasses']]  # Which categories to use for this *training* episode?


    cats = np.random.permutation(cats)
    #print(cats)

    rots = np.random.randint(4, size=len(imagedata))

    #rots.fill(0)

    testcat = random.choice(cats) # select the class on which we'll test in this episode
    unpermcats = cats.copy()

    # Inserting the character images and labels in the input tensor at the proper places
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
                #numi =nc * (params['nbclasses'] * (params['prestime']+params['ipd'])) + ii * (params['prestime']+params['ipd']) + nn

                inputT[location][0][0][:][:] = p[:][:]
                labelT[location][0][np.where(unpermcats == catnum)] = 1 # The (one-hot) label is the position of the category number in the original (unpermuted) list
                #if nn == 0:
                #    print(labelT[location][0])
                location += 1
            location += params['ipd']

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
    #print(testcat, testlabel)

    #pdb.set_trace()


    assert(location == params['nbsteps'])

    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to pytorch Tensor
    labelT = torch.from_numpy(labelT).type(ttype)
    targetL = torch.from_numpy(testlabel).type(ttype)

    return inputT, labelT, targetL



class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        self.rule = params['rule']
        if params['flare'] == 1:
            self.cv1 = torch.nn.Conv2d(1, params['nbf'] //4 , 3, stride=2).cuda()
            self.cv2 = torch.nn.Conv2d(params['nbf'] //4 , params['nbf'] //4 , 3, stride=2).cuda()
            self.cv3 = torch.nn.Conv2d(params['nbf'] //4, params['nbf'] //2, 3, stride=2).cuda()
            self.cv4 = torch.nn.Conv2d(params['nbf'] //2,  params['nbf'], 3, stride=2).cuda()
        else:
            self.cv1 = torch.nn.Conv2d(1, params['nbf'] , 3, stride=2).cuda()
            self.cv2 = torch.nn.Conv2d(params['nbf'] , params['nbf'] , 3, stride=2).cuda()
            self.cv3 = torch.nn.Conv2d(params['nbf'] , params['nbf'] , 3, stride=2).cuda()
            self.cv4 = torch.nn.Conv2d(params['nbf'] ,  params['nbf'], 3, stride=2).cuda()



        self.w =  torch.nn.Parameter((.01 * torch.randn(params['nbf'], params['nbclasses'])).cuda(), requires_grad=True)
        #self.w =  torch.nn.Parameter((.01 * torch.rand(params['plastsize'], params['nbclasses'])).cuda(), requires_grad=True)
        if params['alpha'] == 'free':
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['nbf'], params['nbclasses'])).cuda(), requires_grad=True) # Note: rand rather than randn (all positive)
        elif params['alpha'] == 'yoked':
            self.alpha =  torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        else :
            raise ValueError("Must select a value for alpha ('free' or 'yoked')")
        self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
        self.params = params

    def forward(self, inputx, inputlabel, hebb):
        if self.params['activ'] == 'selu':
            activ = F.selu(self.cv1(inputx))
            activ = F.selu(self.cv2(activ))
            activ = F.selu(self.cv3(activ))
            activ = F.selu(self.cv4(activ))
        elif self.params['activ'] == 'relu':
            activ = F.relu(self.cv1(inputx))
            activ = F.relu(self.cv2(activ))
            activ = F.relu(self.cv3(activ))
            activ = F.relu(self.cv4(activ))
        elif self.params['activ'] == 'tanh':
            activ = F.tanh(self.cv1(inputx))
            activ = F.tanh(self.cv2(activ))
            activ = F.tanh(self.cv3(activ))
            activ = F.tanh(self.cv4(activ))
        else:
            raise ValueError("Parameter 'activ' is incorrect (must be tanh, relu or selu)")
        #activ = F.tanh(self.conv2plast(activ.view(1, self.params['nbf'])))
        #activin = activ.view(-1, self.params['plastsize'])
        activin = activ.view(-1, self.params['nbf'])

        if self.params['alpha'] == 'free':
            activ = activin.mm( self.w + torch.mul(self.alpha, hebb)) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will overwhelm the inputs and clamp the outputs
        elif self.params['alpha'] == 'yoked':
            activ = activin.mm( self.w + self.alpha * hebb) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will clip this and affect the other inputs
        activout = F.softmax( activ )

        if self.rule == 'hebb':
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(activin.unsqueeze(2), activout.unsqueeze(1))[0] # bmm used to implement outer product; remember activs have a leading singleton dimension
        elif self.rule == 'oja':
            hebb = hebb + self.eta * torch.mul((activin[0].unsqueeze(1) - torch.mul(hebb , activout[0].unsqueeze(0))) , activout[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
        else:
            raise ValueError("Must select one learning rule ('hebb' or 'oja')")

        return activout, hebb

    def initialZeroHebb(self):
        #return Variable(torch.zeros(self.params['plastsize'], self.params['nbclasses']).type(ttype))
        return Variable(torch.zeros(self.params['nbf'], self.params['nbclasses']).type(ttype))




def train(paramdict=None):
    #params = dict(click.get_current_context().params)
    print("Starting training...")
    params = {}
    params.update(defaultParams)
    if paramdict:
        params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    sys.stdout.flush()
    params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['ipd']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "W"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames
    print("Suffix: ", suffix, "length:", len(suffix))
    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)

    print("Loading Omniglot data...")
    imagedata = []
    imagefilenames=[]
    for basedir in ('./omniglot_dataset/omniglot/python/images_background/',
                    './omniglot_dataset/omniglot/python/images_evaluation/'):
        #print ("Entered the directory ********************************\n *****************")
        alphabetdirs = glob.glob(basedir+'*')
        print(alphabetdirs[:4], '##########################################\n')
        for alphabetdir in alphabetdirs:
            #print ("entering alphabet directory ******************* \n ###############")
            chardirs = glob.glob(alphabetdir+"/*")
            for chardir in chardirs:
                chardata = []
                charfiles = glob.glob(chardir+'/*')
                for fn in charfiles:
                    filedata = skimage.io.imread(fn) / 255.0 #plt.imread(fn)
                    chardata.append(filedata)
                imagedata.append(chardata)
                imagefilenames.append(fn)
    # imagedata is now a list of lists of numpy arrays
    # imagedata[CharactertNumber][FileNumber] -> numpy(105,105)
    np.random.shuffle(imagedata)  # Randomize order of characters
    print(len(imagedata))
    print(imagedata[1][2].shape)
    print("Data loaded!")



    print("Initializing network")
    net = Network(params)
    #net.cuda()
    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    #total_loss = 0.0
    print("Initializing optimizer")
    #optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=params['lr'])
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'])
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['gamma'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print (pytorch_total_params)
    all_losses = []
    all_losses_objective = []
    lossbetweensaves = 0.0
    lossbetweensavesprev = 1e+10
    #test_every = 20
    nowtime = time.time()

    print("Starting episodes...")
    sys.stdout.flush()

    for numiter in range(params['nbiter']):
        torch.cuda.empty_cache()
        hebb = net.initialZeroHebb()
        optimizer.zero_grad()

        is_test_step = ((numiter+1) % params['test_every'] == 0)
        inputs, labels, target = generateInputsLabelsAndTarget(params, imagedata, test=is_test_step)


        for numstep in range(params['nbsteps']):
            y, hebb = net(Variable(inputs[numstep], requires_grad=False), Variable(labels[numstep], requires_grad=False), hebb)

        # Compute the loss
        criterion = torch.nn.BCELoss()
        loss = criterion(y[0], Variable(target, requires_grad=False))

        # Compute the gradients
        if is_test_step == False:
            loss.backward()

            maxg = 0.0
            scheduler.step()
            optimizer.step()

        lossnum = loss.data
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)

        if is_test_step: # (numiter+1) % params['test_every'] == 0:

            print(numiter, "====")
            td = target.cpu().numpy()
            yd = y.data.cpu().numpy()[0]
            print("y: ", yd[:10])
            print("target: ", td[:10])
            #print("target: ", target.unsqueeze(0)[0][:10])
            absdiff = np.abs(td-yd)
            print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
            print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
            #print inputs[numstep]
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['test_every'], "iters: ", nowtime - previoustime)

            print("Loss on single withheld-data episode:", lossnum)
            all_losses.append(lossnum)
            print ("Eta: ", net.eta.data.cpu().numpy())
            sys.stdout.flush()
            #total_loss = 0


        if (numiter+1) % params['save_every'] == 0:
            print("Saving files...")
            lossbetweensaves /= params['save_every']
            print("Average loss over the last", params['save_every'], "episodes:", lossbetweensaves)
           # print("Alternative computation (should be equal):", np.mean(all_losses_objective[-params['save_every']:]))
            losslast100 = torch.mean(torch.stack(all_losses_objective[-100:]))
            print("Average loss over the last 100 episodes:", losslast100)


            #else: # to "print("Saved!")"
            print("Saving local files...")
            localsuffix = suffix
            if (numiter + 1) % 500000 == 0:
                localsuffix = localsuffix + "_"+str(numiter+1)
            with open('./results/results_'+localsuffix+'.dat', 'wb') as fo:
                pickle.dump(net.w.data.cpu().numpy(), fo)
                pickle.dump(net.alpha.data.cpu().numpy(), fo)
                pickle.dump(net.eta.data.cpu().numpy(), fo)
                pickle.dump(all_losses, fo)
                pickle.dump(params, fo)
            with open('./loss/loss_'+localsuffix+'.txt', 'w') as thefile:
                for item in all_losses:
                    #item = item.detach().cpu()
                    thefile.write("%s\n" % item)
            torch.save(net.state_dict(), './torchmodels/torchmodel_'+localsuffix+'.txt')
            lossbetweensavesprev = lossbetweensaves
            lossbetweensaves = 0
            sys.stdout.flush()
            sys.stderr.flush()



@click.command()
@click.option('--nbclasses', default=defaultParams['nbclasses'])
@click.option('--alpha', default=defaultParams['alpha'])
#@click.option('--plastsize', default=defaultParams['plastsize'])
@click.option('--rule', default=defaultParams['rule'])
@click.option('--gamma', default=defaultParams['gamma'])
@click.option('--steplr', default=defaultParams['steplr'])
@click.option('--activ', default=defaultParams['activ'])
@click.option('--flare', default=defaultParams['flare'])
@click.option('--nbshots', default=defaultParams['nbshots'])
@click.option('--nbf', default=defaultParams['nbf'])
@click.option('--prestime', default=defaultParams['prestime'])
@click.option('--prestimetest', default=defaultParams['prestimetest'])
@click.option('--ipd', default=defaultParams['ipd'])
@click.option('--nbiter', default=defaultParams['nbiter'])
@click.option('--lr', default=defaultParams['lr'])
@click.option('--test_every', default=defaultParams['test_every'])
@click.option('--save_every', default=defaultParams['save_every'])
@click.option('--rngseed', default=defaultParams['rngseed'])
def main(nbclasses, alpha, rule, gamma, steplr, activ, flare, nbshots, nbf, prestime, prestimetest, ipd, nbiter, lr, test_every, save_every, rngseed):
    train(paramdict=dict(click.get_current_context().params))
    #print(dict(click.get_current_context().params))

if __name__ == "__main__":
    #train()
    main()
