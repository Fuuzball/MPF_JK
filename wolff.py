import time
import numpy as np
import random
import matplotlib.pylab as plt
#np.random.seed(42)

def stack_X(X, ratio = 1.5, pad = 1):
    N, Dx, Dy = X.shape
    W = int(np.ceil(np.sqrt(ratio * N)))
    H = int(np.ceil(N / W))

    if H * W > N:
        X = np.concatenate((X, np.zeros((H * W - N, Dx, Dy))))

    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0.5)
    rows = []
    for i in range(H):
        rows.append(np.hstack((padX[i*W:(i+1)*W])))
    Xstack = np.vstack(rows)
    return Xstack

class WolffSampler(object):
    
    def __init__(self, D, J):
        self.D = D
        self.J = J
        self.make_nn_dict()

    def get_recommended_burnin_thin(self, burn_max=1000, thin_max=1000, plot=False):
        X = self.sample_X(burn_max, 0, 1)
        dist0 = ((X[0] != X).mean(axis=(1,2)))
        burn_in_rec = np.nonzero(dist0 > 0.5)[0][1]

        valid_start = np.arange(burn_in_rec, thin_max - burn_in_rec)
        dist = []
        for t in valid_start:
            dist.append(
                ((X[t] != X).mean(axis=(1,2)))[t: t + burn_in_rec] 
                    ) 
        dist = np.array(dist)
        dist_mean = dist.mean(axis=0)
        thin_rec = np.nonzero(dist_mean > 0.5)[0][1]
        
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(np.arange(burn_max), dist0)
            plt.subplot(212)
            plt.plot(dist_mean)

        return burn_in_rec, thin_rec

    def sample_X(self, N, burn_in=None, thin=None, new_J=None, display_time=False): 
        D = self.D
        t0 = time.process_time()
        if new_J:
            self.J = new_J

        if (burn_in is None) or (thin is None):
            print('Using recommended burn in / thin...')
            burn_in_rec, thin_rec = self.get_recommended_burnin_thin()
            if burn_in is None:
                burn_in = burn_in_rec
                print('Burn in : ', burn_in)
            if thin is None:
                thin = thin_rec
                print('Thin : ', thin)


        n_sample_steps = burn_in + N * thin
        X = np.zeros((N, D, D)) 
        self.x = np.random.randint(2, size=(D, D)) * 2 - 1

        for i in range(n_sample_steps):
            self.flip_random_cluster() 
            if i >= burn_in and (i - burn_in) % thin == 0:
                X[(i - burn_in) // thin] = self.x

        self.X = X

        if display_time:
            print('Sampling took {:.4f}s'.format(time.process_time() - t0))
        return X

    def flip_random_cluster(self):
        d_init = self.get_rand_site()
        visited = set([])
        # Initial site
        to_flip = {d_init}
        p = 1 - np.exp(-self.J)

        while to_flip:
            new_to_flip = set([])
            for d in to_flip:
                spin = self.x[d]
                visited.add(d)
                for n in self.nn_dict[d]:
                    if ( self.x[n] == spin) and ( n not in visited ) and (random.random() < p):
                        new_to_flip.add(n)
                self.x[d] *= -1
            visited.update(to_flip)
            to_flip = set(new_to_flip)

    def get_rand_site(self):
        return (random.randint(0, self.D - 1), random.randint(0, self.D - 1))

    def make_nn_dict(self):
        self.nn_dict = {}
        for i in range(self.D):
            for j in range(self.D):
                nn = []
                if i > 0:
                    nn.append((i - 1, j))
                if j > 0:
                    nn.append((i, j - 1)) 
                if i < self.D - 1:
                    nn.append((i + 1, j))
                if j < self.D - 1:
                    nn.append((i, j + 1))
                self.nn_dict[(i, j)] = nn

    def plot_sample(self):
        plt.imshow(stack_X(self.X))
        plt.show()

if __name__ == '__main__':
    N = 10
    D = 20

    J = np.log(1 + np.sqrt(2))
    print('J : ', J)

    #wolff = WolffSampler(J, D, N, burn_in, thin)
    wolff = WolffSampler(D, J)
    X = wolff.sample_X(N, display_time=True)
    mpf = MPF_Estimator(X)
    print(mpf.learn_jk())
    mpf_torch = glass_torch(X.reshape(N, -1))
    params = (mpf_torch.learn_jb())
    print(mpf_torch.unflatten_params(params))
    mpf_glass = glass(X.reshape(N, -1))
    print(mpf_glass.learn_jb())

    wolff.plot_sample()
