import numpy as np
# covariance, num_p,  
class PF:	
  def __init__(self, init_pose, num_p = 100, cov = 0.001, model = "velocity"):
    '''
    initialize the particle filter with num_p_ particles and velocity or acc model
    Inputs:
    1. init_pose: initial pose obtained from first frame
    2. num_p: number of particles
    3. cov: covariance for noise to be added in the predict step
    4. model: which motion model to use for the predict step. Currently, only supports constant velocity model
    '''
    self.num_p = num_p
    self.model = model
    if model == "velocity":
      self.state_dims = 6
    self.cov = cov * np.identity(self.state_dims)
    self.best_p = init_pose

    self.init_pose = init_pose
    self.particles = np.random.multivariate_normal(self.init_pose, self.cov, self.num_p)
    self.weights = np.ones((self.particles.shape[0]))/self.particles.shape[0] 

  def predict(self, dt = 0.1):
    """
    Move the particles as per the motion model and then add noise
    """
    self.propagate(dt)
    noise = np.random.multivariate_normal(np.zeros((self.state_dims)), self.cov, self.num_p)
    # print(noise)
    self.particles += noise

  def propagate(self,dt):
    """
    apply the motion model
    """
    F = np.identity((self.state_dims))
    if self.model == "velocity":
      F[0, -3] = dt
      F[1, -2] = dt
      F[2, -1] = dt
    # print(F)
    self.particles = np.matmul(F, self.particles[:,:,None])[:,:,0]

  def update(self, correlation):
    '''
    Reweight the particles as per the correlation score
    '''
    self.weights = correlation/np.sum(correlation)
    self.best_p = self.particles[np.argmax(self.weights),:]

  def restratified_sampling(self):
    '''
    Resample the particles as per the distribution governed by current weights
    '''
    print("resampling particles!")
    means = self.particles
    weights = self.weights
    N = self.weights.shape[0]
    c = weights[0]
    j = 0
    u = np.random.uniform(0,1.0/N)

    new_mean = np.zeros(means.shape)
    new_weights = np.zeros(weights.shape)
    
    for k in range(N):
        beta = u + float(k)/N

        while beta > c:
            j += 1
            c += weights[j]

        # add point
        new_mean[k] = means[j]
        new_weights[k] = 1.0/N

    self.particles = new_mean
    self.weights = new_weights


if __name__ == "__main__":
    init_pose = np.array([0, 0, 0, 0, 0, 0])
    pf = PF(init_pose)

    pf.predict()