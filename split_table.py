
import os
import sys
import itertools
import numpy as np
import pandas as pd




def main():
    data = pd.read_csv(sys.argv[1], delimiter="|", header=None).values
    # data = np.genfromtxt(sys.argv[1], dtype=np.str, delimiter='|')

    params = []
    for dat in data:
        param = [[a.split('.') for a in e.split('-')] for e in os.path.splitext(dat[0])[0].split('/')]
        param = list(itertools.chain.from_iterable(param))
        param = list(itertools.chain.from_iterable(param))
        params += [[param[i] for i in range(8,17,2)]]

    params = np.array(params)
    baccs = np.array(data[:, -5], dtype=np.str)
    baccs = baccs[:, np.newaxis]

    data = np.concatenate((params, baccs), axis=1)
    
    rodadas = []
    for i in range(1,11):
        st = 'rodada%d'%i
        idx = np.where(data[:, 2]==st)[0]
        if len(idx) == 55:
            tab = data[idx][:,-1].reshape((5,-1)).T
            rodadas += [tab]
        else:
            print('Round %d did not finished yet!'%i, flush=True)

    f = open('tab.txt', 'w').close()
    f = open('tab.txt', 'a')
    for tab in rodadas:
        np.savetxt(f, tab.astype(np.float32), fmt='%.5f,%.5f,%.5f,%.5f,%.5f')
        f.write('\n\n\n')
    f.close()


if __name__ == '__main__':
    main()


