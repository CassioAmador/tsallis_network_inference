import copy
import numpy as np
import matplotlib.pyplot as plt


def eval_entropy(probs, q=1):
    probs = copy.copy(probs)
    res=0
    """evaluates entropy for a given probability or array of probabilities
    prob: the input probability
    q: non-extensive parameter. Defaults to 1 (boltzmann entropy)"""
    if q == 1:
        try:
            if isinstance(probs,np.ndarray):
                probs[np.where(probs == 0 )] = 1
            else:
                probs=[1 if prob==0 else prob for prob in probs]
            res=[-prob * np.log(prob) for prob in probs]
            soma=sum(res)
        except TypeError:
            if probs==0:
                probs = 1
            res=-probs * np.log(probs)
            soma=res
        return soma
    else:
        try:
            res=[pow(prob, q) for prob in probs]
            soma=sum(res)
        except TypeError:
            res=pow(probs, q)
            soma=res
        return (1 - soma) / (q - 1)

def sum_entropy(entro1, entro2, q=1):
    return entro1 + entro2 + (1 - q) * entro1 * entro2
    

if __name__ == '__main__':
    q=1.3
    entro=eval_entropy(0.5,q=q)
    print(entro)
    q=1
    probs=np.linspace(0,1,100)
    entro=[eval_entropy([p,1-p],q=q) for p in probs]
    plt.figure(figsize=(6,4))
    plt.plot(probs,entro)
    plt.xlabel("probability $p_0$")
    plt.ylabel("entropy/k")
    plt.title("Entropy x Probability")
    plt.tight_layout()
    plt.ylim(0,0.72)
    # plt.vlines(0.5,0,0.72,"r")
    plt.show()