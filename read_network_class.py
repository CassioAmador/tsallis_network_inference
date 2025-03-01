import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

import entropy as ent

offset = 1


class Network:
    """docstring for Network"""

    def __init__(self, agn_name):
        super().__init__()
        self.agn_name = agn_name
        self.alfa = 1
        self.read_network()
        self.read_adj_matrix()
        self.configurations = {}
        self.targene_table_base = {}
        self.grand_tar_tab_prof = {}
        self.probao,poptao=np.load("2estados.npy")                                                                                            
        self.func=interpolate.interp1d(self.probao,poptao)


    def read_network(self, verbose=False):
        """Read network, number of steps and of genes, from txt file"""
        self.filename_network = "{}_txt.agn".format(self.agn_name)
        self.steps = int(open(self.filename_network,
                              'r').readline().split(";")[-2]) + 1
        self.d = self.steps - 1
        self.network = np.loadtxt(self.filename_network, delimiter=";",
                                  usecols=range(self.steps), skiprows=1, dtype=int)
        self.genes_number = self.network.shape[0]
        if verbose:
            print("steps:", self.steps, "genes:", self.genes_number)

    def read_adj_matrix(self):
        """Read adjacency matrix"""
        self.filename_adj = "{}_adj.agn".format(self.agn_name)
        self.adj_matrix = np.loadtxt(self.filename_adj, delimiter=";",
                                     usecols=range(self.genes_number), skiprows=1, dtype=int)

    def eval_activity(self):
        dif = np.diff(self.network, 1)
        self.activity = np.zeros(self.genes_number, dtype=int)
        for d, di in enumerate(dif):
            self.activity[d] = np.count_nonzero(di)
        # most active genes:
        self.most_active = np.where(self.activity == self.activity.max())[0]
        # less active genes:
        self.less_active = np.where(self.activity == self.activity.min())[0]

    def eval_adj_matrix(self):
        self.targene_adj = self.adj_matrix[self.targene]
        self.conns = np.where(self.targene_adj == self.targene_adj.max())[0]
        if self.conns.size == (self.genes_number):
            self.conns = []

    def apply_targene(self, targene):
        self.targene = targene
        # find predictors for targene:
        self.predictors = np.where(self.adj_matrix[:, self.targene] == 1)[0]
        self.targets = np.where(self.adj_matrix[self.targene, :] == 1)[0]

    def eval_targene_entropy(self, verbose=False):
        """Evaluates entropy of target gene frequency"""
        p1 = np.count_nonzero(self.network[self.targene][
                              1:]) / self.d
        p0 = 1 - p1
        # if (p0 == 0) or (p1 == 0):
        #     p0 = 0.5
        #     p1 = 0.5
        entro = ent.eval_entropy([p0, p1], self.q)
        if verbose:
            print(p0, p1, p0 + p1, entro)
        self.targene_entro = entro

    def make_table_1D(self, gene, dif=False, verbose=False):
        self.configurations1 = (0, 1, 10, 11)
        self.targene_table_1D = self.targene_table_1D_base.copy()
        if gene != self.targene:
            if dif:
                diffs = np.diff(self.network[gene])
                self.confs_1D = diffs * 10 + self.network[self.targene, 1:]
            else:
                self.confs_1D = self.network[gene, :-1] * \
                    10 + self.network[self.targene, 1:]
            for c, conf in enumerate(self.configurations1):
                self.targene_table_1D[c] = np.count_nonzero(
                    self.confs_1D == conf)
            if verbose:
                print("\n  {} |  target: {} ".format(gene, self.targene))
                print("\u0332".join(" G1 |  {: <3}    {: <3} ".format(0, 1)))
                print("  0 |  {: <3}   {: <3}".format(
                    self.targene_table_1D[0], self.targene_table_1D[1]))
                print("  1 |  {: <3}   {: <3}\n".format(
                    self.targene_table_1D[2], self.targene_table_1D[3]))

            return True
        else:
            return False

    def make_table_2D(self, gene1, gene2, dif=False):
        self.configurations2 = (0, 10, 100, 110, 1, 11, 101, 111)
        """create frequency table for relation gene1-gene2-target"""
        self.targene_table_2D = self.targene_table_2D_base.copy()
        if (gene1 != gene2) and (gene1 != self.targene) and (gene2 != self.targene):
            if dif:
                diffs1 = np.diff(self.network[gene1])
                diffs2 = np.diff(self.network[gene2])
                self.confs_2D = diffs1 * 100 + diffs2 * \
                    10 + self.network[self.targene, 1:]
            else:
                self.confs_2D = self.network[
                    gene1, :-1] * 100 + self.network[gene2, :-1] * 10 + self.network[self.targene, 1:]
            for c, conf in enumerate(self.configurations2):
                self.targene_table_2D[c] = np.count_nonzero(
                    self.confs_2D == conf)
            return True
        else:
            return False

    def make_table(self, genes, verbose=False):
        """create frequency table for relation gene1-...-genen-target"""
        n = len(genes)
        if n not in self.configurations:
            self.configurations[n] = [
                int("".join(conf)) for conf in itertools.product(['0', '1'], repeat=n + 1)]
            self.targene_table_base[n] = np.zeros(
                len(self.configurations[n]), dtype=int)
        self.targene_table = self.targene_table_base[n].copy()
        if (self.targene not in genes) and (len(genes) == len(set(genes))):
            self.confs = sum([self.network[gene, :-1] * 10**(n - g)
                              for g, gene in enumerate(genes)]) + self.network[self.targene, 1:]
            for c, conf in enumerate(self.configurations[n]):
                self.targene_table[c] = np.count_nonzero(self.confs == conf)

            if verbose:
                print("\n  {} |  target: {} ".format(genes, self.targene))
                print("\u0332".join(" {} |  {: <3}    {: <3} ".format(
                    "genes".center(3 * n + 1), 0, 1)))
                for c in range(0, len(self.configurations[n]), 2):
                    print("  {} |  {: <3}   {: <3}".format(
                        str(self.configurations[n][c])[:-1].zfill(n).center(3 * n), self.targene_table[c], self.targene_table[1 + c]))
            return True
        else:
            return False

    def eval_table_entropy(self, dim=2, verbose=False):
        if dim == 1:
            ok = self.make_table_1D(self.gene1)
            targene_table = self.targene_table_1D
        elif dim == 2:
            ok = self.make_table_2D(self.gene1, self.gene2)
            targene_table = self.targene_table_2D
        if ok:
            targene_table_prob = targene_table / self.d
            self.targene_table_entro = ent.eval_entropy(
                targene_table_prob, self.q)
            entro = self.targene_table_entro * targene_table_prob
            entro[np.where(entro == 0)] = self.targene_entro
            if verbose:
                print(targene_table_prob)
                print(self.targene_table_entro)
                print(entro)
                print(entro.sum())
            return entro.sum()
        else:
            return offset

    def eval_cond_entropy(self, genes, verbose=False):
        comb_name = "_".join([str(gene)
                              for gene in genes]) + "_" + str(self.targene)
        if comb_name not in self.grand_tar_tab_prof:
            ok = self.make_table(genes, verbose=verbose)
        else:
            ok = True
        if ok:
            if comb_name not in self.grand_tar_tab_prof:
                # p0 = targene_table[conf_number - 1] / sum(targene_table)
                # p1 = targene_table[2 * len(genes) + conf_number - 1] / sum(targene_table)
                self.M = 2**len(genes)
                self.Py = self.targene_table.reshape(self.M, 2)
                self.fx = self.Py.sum(1)
                self.Px = self.fx / self.d
                self.N = np.count_nonzero(self.fx)
                self.targene_table_prob = np.zeros((self.M, 2))
                for f, fx in enumerate(self.fx):
                    if fx != 0:
                        self.targene_table_prob[f] = self.Py[f] / fx
                    else:
                        self.targene_table_prob[f] = 0
                self.grand_tar_tab_prof[comb_name] = [
                    self.targene_table_prob, self.Py, self.M, self.N]
            else:
                self.targene_table_prob, self.Py, self.M, self.N = self.grand_tar_tab_prof[
                    comb_name]
            self.part_entropy = 0

            for x in range(self.M):
                prob_yx = self.targene_table_prob[x]
                if 0<prob_yx.min()<=self.probao.max():
                    q_real=self.func(prob_yx.min())
                else:
                    q_real=q
                self.part_entropy += (self.Px[x]) * \
                    ent.eval_entropy(prob_yx, self.q)

            # for x in range(self.M):
            #     prob_yx = self.targene_table_prob[:, x]
            #     self.part_entropy += (self.fx[x] + self.alfa)*ent.eval_entropy(prob_yx, self.q)
            #     # self.part_entropy += (self.Px[x] + self.alfa)*ent.eval_entropy(prob_yx, self.q)
            # # print(self.part_entropy,self.fx,(self.fx+self.alfa)*self.part_entropy/(self.alfa * M + self.steps - 1))
            self.penalty = self.alfa * (self.M - self.N) * self.targene_entro
            cond_entropy = self.penalty + self.part_entropy
            # cond_entropy = (self.penalty + self.part_entropy) / (self.alfa * self.M + self.d)
            # # cond_entropy = (self.penalty + self.part_entropy) / (self.alfa * self.M + 1)
            if verbose:
                print("M, N:", self.M, self.N)
                print("Py:\n", self.Py)
                print("Px:", self.Px)
                print("fx:", self.fx)
                print("targene_table_prob:\n",
                      self.targene_table_prob.transpose())
                for x in range(self.M):
                    prob_yx = self.targene_table_prob[x]
                    print("xi entropy:", x, prob_yx,
                          ent.eval_entropy(prob_yx, self.q))
                print("part_entropy:", self.part_entropy)
                print("targene_entro:", self.targene_entro)
                print("penalty:", self.penalty)
                print("cond_entropy:", cond_entropy)
                print()
            return cond_entropy
        else:
            return offset

    def run_genes(self, dim=1):
        if dim <= 2:
            self.entroes1 = np.zeros(self.genes_number) + offset
            for gene1 in range(self.genes_number):
                if gene1 != self.targene:
                    self.gene1 = gene1
                    self.entroes1[gene1] = self.eval_cond_entropy(dim=1)
            self.gene1 = self.entroes1.argmin()
        if dim == 2:
            self.entroes2 = np.zeros(self.genes_number) + offset
            for gene2 in range(self.genes_number):
                self.gene2 = gene2
                entrao = self.eval_cond_entropy(dim=2)
                # entrao = self.eval_table_entropy(dim=2)
                self.entroes2[gene2] = entrao
            self.gene2 = np.where(self.entroes2 == self.entroes2.min())[0][0]
        if dim == 3:
            import multiprocessing as mp
            self.qs = np.linspace(0.5, 1.5, 10)
            self.entradas = np.zeros_like(self.qs)
            for q_ind, q in enumerate(self.qs):
                self.eval_q(q, q_ind)
            # # Setup a list of processes that we want to run
            # self.processes = [mp.Process(target=self.eval_q, args=(q,qq,)) for qq,q in enumerate(self.qs)]
            # # Run processes
            # for p in self.processes:
            #     p.start()

            # asd = np.where(self.entroes3 == self.entroes3.min())[0]
            # self.eval_adj_matrix()
            # for gene in asd:
            #     if gene in self.conns:
            #         print(self.q, gene)

            # for gene1 in range(self.genes_number):
            #     self.gene1=gene1
            #     for gene2 in range(self.genes_number):
            #         self.gene2=gene2
            #         if (self.gene1 in self.conns) and (self.gene2 in self.conns):
            #             plt.plot(self.gene1*self.genes_number+self.gene2,self.entroes3[gene1, gene2],'ko')

    def eval_q(self, q, q_ind):
        self.q = q
        self.entropia_q = np.zeros(self.genes_number)
        for targene in range(int(self.genes_number / 2)):
            self.eval_targene_entropy(targene)
            self.entroes3 = np.zeros(
                (self.genes_number, self.genes_number)) + offset
            # self.genes = np.mgrid[0:self.genes_number,0:self.genes_number]
            for gene1 in range(self.genes_number):
                self.gene1 = gene1
                for gene2 in range(gene1):
                    self.gene2 = gene2
                    self.entroes3[
                        gene1, gene2] = self.eval_table_entropy(dim=2)
            self.entropia_q[targene] = self.entroes3.sum()
        self.entradas[q_ind] = self.entropia_q.sum()

    def check_entropy(self):
        self.entropies = []
        for targene in range(self.genes_number):
            self.eval_targene_entropy(targene)
            self.eval_adj_matrix()
            if len(self.conns) != 0:
                entrao = 0
                for conn1 in self.conns:
                    self.gene1 = conn1
                    # for conn2 in self.conns:
                    #     self.gene2=conn2
                    #     if self.gene1!=self.gene2:
                    entro = self.eval_table_entropy(dim=1)
                    # if entro == offset:
                    #     print(self.targene, conn, self.conns)
                    entrao += entro
                self.entropies.append(entrao / (self.conns.size**2))

    def test(self):
        # prob=self.adj_matrix.sum(1)/(self.genes_number-1)
        prob = self.adj_matrix.sum(1)
        his, bin_edges = np.histogram(
            prob, bins=np.arange(prob.min(), prob.max(), 1))
        plt.plot(bin_edges[:-1] + np.diff(bin_edges), his, 'o')


def find_gene1(network, steps, genes_number, targene, targene_ent, proibidao, q=1):
    min_ent = 1e99
    for gene in range(self.genes_number):
        if (gene not in proibidao):
            targene_table_1D = make_table_1D(
                self.network, self.steps, gene, targene)
            entro = eval_cond_entropy(
                targene_table_1D, targene_ent, self.steps, dim=1, q=q)
            if entro < min_ent:
                min_ent = entro
                gene1 = gene
    # print(min_ent)
    return gene1, min_ent


def find_gene2(network, steps, genes_number, gene1, targene, targene_ent, min_ent1, q=1):
    right_one = True
    min_ent = 1e99
    for gene in range(self.genes_number):
        if (gene1 != gene) and (gene1 != targene) and (gene != targene):
            targene_table_2D = make_table_2D(
                self.network, self.steps, gene1, gene, targene)
            entro = eval_cond_entropy(
                targene_table_2D, targene_ent, self.steps, dim=2, q=q)
            if entro < min_ent:
                min_ent = entro
                gene2 = gene
    if min_ent > min_ent1:
        # print("rode de novo")
        right_one = False
        print(min_ent, min_ent1)
    # print(min_ent1,min_ent)
    return gene2, right_one


def teste(arq):
    verbose = True
    net = Network(arq)
    rede = range(net.genes_number)
    somas = []
    q = 1
    # qs=np.linspace(0.5,2,10)
    somao = 0
    # for q in qs:
    net.q = q
    for targene in rede:
        print("target gene:", targene)
        targene_ent = net.eval_targene_entropy(targene)
        net.run_genes(dim=1)
        print(net.gene1)
        # print(net.targene_ent)
        # proibidao = [targene]
        # right_one = False
        # while right_one == False:
        #     gene1, min_ent1 = find_gene1(
        #         network, steps, net.genes_number, targene, targene_ent, proibidao, q=q)
        #     gene2, right_one = find_gene2(
        #         network, steps, net.genes_number, gene1, targene, targene_ent, min_ent1, q=q)
        #     if right_one == False:
        #         proibidao.append(gene1)
        #         if len(proibidao) == net.genes_number:
        #             print("entropia furada")
        #             break
        #     print(gene1, adj_matrix[gene1][targene])
        #     print(gene2, adj_matrix[gene2][targene])
        #     print("")
        # targene_table_1D = make_table_1D(
        #     network, steps, gene1, targene)
        # targene_table_2D = make_table_2D(
        #     network, steps, gene1, gene2, targene)
        # print(targene_table_1D)
        # somao += adj_matrix[gene1][targene] + adj_matrix[gene2][targene]
    # print(somao)
    # print(q,somao)


def teste2():
    q = 1
    q_min = 1
    eita_grande = 1e99
    for q in np.linspace(0.1, 10, 20):
        # arq = "teste_grande_pb_txt.agn"
        # arq = "teste_pequeno_txt.agn"
        arq = "teste3"
        verbose = False
        network, steps, genes_number = read_network(arq, verbose)
        adj_matrix = read_adj_matrix(arq, genes_number)
        entropia = np.zeros(genes_number)
        # targene=
        entrão = 0
        for targene in range(5):
            targene_ent = eval_targene_entropy(network, targene, steps, q=q)
            for gene in range(genes_number):
                if gene != targene:
                    targene_table_1D = make_table_1D(
                        network, steps, gene, targene)
                    entro = eval_cond_entropy(
                        targene_table_1D, targene_ent, steps, dim=1, q=q)
                    entropia[gene] = entro
                else:
                    entropia[gene] = 1e99

            # print("q=",q,"menor entropia em:", entropia.argmin())
        entrão += entropia.min()
        # print(entrão)
        if entrão < eita_grande:
            eita_grande = entrão
            q_min = q
            # print(q)
    print(q_min, eita_grande)


def teste3():
    net = Network(arq)
    # import sys
    # sys.exit()

    targene = 0
    qs = np.linspace(0.5, 2, 50)
    entradas = []
    # for targene in (10,12):
    for targene in (0, 9):
        entras = []
        for q in qs:
            # q=1
            entro0, entro1, targene_ent = net.eval_targene_entropy(
                targene, q=q, verb=False)
            entroes = np.zeros(genes_number - 1)
            for gene in range(genes_number):
                if gene != targene:
                    targene_table = make_table_1D_dif(
                        network, steps, gene, targene)
                    # entrao=ent.eval_entropy(targene_table/(steps-1),q=q)
                    entrao = eval_cond_entropy(
                        targene_table, targene_ent, steps, dim=1, q=1, verb=False)
                    # for i in range(4):
                    #     if (entrao[i]==0) or (entrao[-1]==-0):
                    #         if i in (0,1):
                    #             entrao[i]=entro0
                    #         else:
                    #             entrao[i]=entro1
                    # entroes[gene]=sum(entrao)
                    if gene < targene:
                        entroes[gene] = entrao
                    elif gene > targene:
                        entroes[gene - 1] = entrao
            # entroes.sort()
            # plt.plot(normalize(entroes[50:]),label=q)
            entras.append(entroes.min())
        entradas.append(np.array(entras))
    plt.plot(qs, entradas[0] - entradas[1], '.', label=targene)
    # print(qs[(entradas[0]-entradas[1]).argmin()])
    # plt.legend(loc='best')
    plt.show()


def comp_q(net):
    rede = range(net.genes_number)
    qs = np.linspace(0.1, 8, 100)
    certoes = []
    entras = {}
    sais = {}
    # qs=[1]
    for q in qs:
        net.q = q
        entras[q] = []
        sais[q] = []
        certo = 0
        for targene in rede:
            # for targene in [36]:
            # print("target gene:", targene)
            # print(net.adj_matrix[targene]) #sai deste gene e vai para outro
            # print(net.adj_matrix[:,targene]) #genes que afetam este alvo
            net.apply_targene(targene)
            targene_ent = net.eval_targene_entropy(verbose)
            act = np.where(net.adj_matrix[:, targene] == 1)[0]
            nonact = np.where(net.adj_matrix[:, targene] == 0)[0]
            for gene in act:
                net.gene1 = gene
                entro = net.eval_cond_entropy(dim=1)
                if entro > 1:
                    print("{:.3}".format(q), gene, entro)
                entras[q].append(entro)
                # print("{:.3}".format(q),gene,entro)
            for gene in nonact[-10:]:
                net.gene1 = gene
                entro = net.eval_cond_entropy(dim=1)
                sais[q].append(entro)
                if entro > 1:
                    print("{:.3}".format(q), gene, entro)

                # print("{:.3}".format(q),gene,entro)

            # print(act)
    #         # print(net.targene_entro)
            # net.run_genes(dim=1)
            # if net.gene1 in act:
                # print(net.q,net.gene1)
                # certo+=1
            # else:
            #     print(net.entroes1.min())
        # certoes.append(certo)
    return qs, sais, entras


def compare_q_lines(net):
    net.targene = 10
    qs = np.linspace(0.5, 3, 100)
    for targene in (0, 10, 20, 30, 40, 50, 60, 70, 80):
        net.targene = targene
        act = np.where(net.adj_matrix[:, net.targene] == 1)[0]
        if len(act) != 0:
            entroes = []
            for gene in [act[0]]:
                print("gene:", gene)
                net.gene1 = gene
                for q in qs:
                    net.q = q
                    net.eval_targene_entropy(net.targene)
                    entro = net.eval_cond_entropy(dim=1, verbose=verbose)
                    entroes.append(entro)
            plt.plot(qs, entroes, label=targene)
    plt.legend(loc="best")


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())


if __name__ == '__main__':
    verbose = False
    # arq = "testes/rede2"  # rede com 30 starts
    # arq = "testes/rede_5"  # rede com 1 start, k=1,
    # arq = "testes/rede_10" # rede com 1 start, k=1,

    import sys

    try:
        arq="testes/"+sys.argv[1]

    except IndexError:
        print("specify gene network")
        sys.exit()

    

    net = Network(arq)
    # qs = np.linspace(0.1, 8, 100)
    # qs = np.linspace(1.5, 3, 500)
    # qs = np.linspace(2.35, 2.55, 500)
    qs = np.linspace(2, 3, 500)
    min_q = []
    for node in range(net.genes_number):
        node_entropy = np.zeros_like(qs)
        net.apply_targene(node)
        targets, predictors = net.targets, net.predictors
        print(node, targets, predictors)
        if len(predictors) != 0:
            somar = []
            for i, q in enumerate(qs):
                net.q = q
                som = 0
                max_entro = ent.eval_entropy([0.5, 0.5], q)
                net.eval_targene_entropy()
                entropy = net.eval_cond_entropy(predictors, verbose=False)
                if entropy < 0.4 * max_entro:
                    som += 1
                # print(entropy)
                node_entropy[i] = entropy / max_entro
                somar.append(som - len(predictors))
            if node_entropy.min() > 1e-3:
                min_q.append(qs[node_entropy.argmin()])
            else:
                min_q.append(None)
        else:
            min_q.append(None)
        # print(node_entropy)
    plt.plot(min_q,'.')
    plt.xlabel("gene")
    plt.ylabel("melhor q")
    plt.title(arq.split("/")[-1])
    plt.show()

    # for gene in (0, 10, 20, 30, 40, 50, 60, 70, 80):
    # for targene in range(net.genes_number):
    #     net.targene=targene
    #     for gene in range(net.genes_number):
    #         net.gene1 = gene
    #         entroes = []
    #         for q in qs:
    #             net.q = q
    #             net.eval_targene_entropy(net.targene)
    #             entro = net.eval_cond_entropy(dim=1, verbose=verbose)
    #             entroes.append(entro)
    #         plt.plot(qs, entroes, label=gene)
    # plt.legend(loc="best")
    # somas=[]
    # for q in qs:
    #     net.q=q
    #     max_entro=ent.eval_entropy([0.5,0.5],q)
    #     lim_entro=0.7*max_entro
    #     somar=0
    #     total=0
    #     for targene in range(net.genes_number):
    #         net.apply_targene(targene)
    #         net.eval_targene_entropy()
    #         for gene in range(net.genes_number):
    #             net.gene1=gene
    #             entro=net.eval_cond_entropy(dim=1)
    #             # entroes.append(entro)
    #             # genes.append(gene)
    #             if entro<lim_entro:
    #                 if gene in net.predictors:
    #                     somar+=1
    #                 # print(q,gene)
    #         total+=net.predictors.size
        # somas.append(somar)

    # pool = mp.Pool(processes=40)
    # results = [pool.apply_async(criar_BD_curso, args=(curso,)) for curso in cursos]
    # output = [p.get() for p in results]
