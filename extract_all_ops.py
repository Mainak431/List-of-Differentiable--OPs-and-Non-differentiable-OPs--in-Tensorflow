
import os


path1 = '/home/titanxpascal/Documents/cpuenv/lib/python2.7/site-packages/tensorflow/'

"""Path1 will be your tensorflow installation path
"""
listi = os.walk(path1)

dirpaths = []
for dirpath, dirname,file in listi:
    #print dirpath
    dirpaths.append(dirpath)


check_values = ['ops.NotDifferentiable', 'ops.RegisterGradient']

nondiff = []

diff = []

count = 0
for i in dirpaths:
    s = os.listdir(i)
    for j in s:
        li = (i+j).split('.')

        if li[-1] == 'py':
            file = open(i +'/'+ j, 'r')
            f = file.readlines()
            for data in f:
                if check_values[0] in data:
                    d = data.replace(')',' (').split('(')
                    nondiff.append(d[1])
                if check_values[1] in data:
                    d = data.replace(')', ' (').split('(')
                    diff.append(d[1])
           
file1 = open('NONDIFFERENTIABLE.TXT', 'w')

for i in nondiff:
    file1.write(i + '\n')

file1 = open('DIFFERENTIABLE.TXT', 'w')

for i in diff:
    file1.write(i + '\n')
