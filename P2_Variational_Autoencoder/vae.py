# Implements auto-encoding variational Bayes.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
    
from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images

def save_images(images, file_name): return s_images(images, file_name, vmin = 0.0, vmax = 1.0)

# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)

# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale = 1e-2):

    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),   # weight matrix
             scale * npr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)         # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs

# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):

    # Params of a diagonal Gaussian.

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    #si cogemos la exponencia de log_std siempre va a ser positiva la desviacion
    # TODO use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # use npr.randn for that.
    # The output of this function is a matrix of size the batch x the number of latent dimensions
    #z = np.sqrt(np.exp(log_std))*npr.randn(*log_std.shape) + mean 
    z = np.exp(log_std)*npr.randn(*log_std.shape) + mean 
    
    return z

# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):

    # logits are in R
    # Targets must be between 0 and 1 

    # TODO compute the log probability of the targets given the generator output specified in logits
    # sum the probabilities across the dimensions of each image in the batch. The output of this function 
    # should be a vector of size the batch size

    loss = np.sum(np.log(targets*sigmoid(logits) +  (1 - targets)*(1 - sigmoid(logits))), axis=-1) 
    return loss

# This evaluates the KL between q and the prior

def compute_KL(q_means_and_log_stds):#ecuacion 12
    
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # TODO compute the KL divergence between q(z|x) and the prior (use a standard Gaussian for the prior)
    # Use the fact that the KL divervence is the sum of KL divergence of the marginals if q and p factorize
    # The output of this function should be a vector of size the batch size
    kl = np.sum((np.exp(log_std*2)+mean**2-1-2*log_std)/2, axis=-1)
    return kl


# This evaluates the lower bound

def vae_lower_bound(gen_params, rec_params, data):#evaluar la ecuacion 13

    # TODO compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:
    #compute_KL(neural_net_predict(sample_latent_variables_from_posterior(neural_net_predict(rec_params,data)))) #1,2
    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    encoder_output = neural_net_predict(rec_params,data)
    #print("1: ",encoder_output)
    # 2 - sample the latent variables associated to the batch in data 
    #     (use sample_latent_variables_from_posterior and the encoder output)
    sample_latent_variables_from_posterior_ = sample_latent_variables_from_posterior(encoder_output)
    #print("2: ",sample_latent_variables_from_posterior_)
    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    gen = bernoulli_log_prob(data,neural_net_predict(gen_params, sample_latent_variables_from_posterior_))
    #print("3bis: ",neural_net_predict(gen_params, sample_latent_variables_from_posterior_))
    #print("3: ",gen)
    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    kl = compute_KL(encoder_output) 
    #print("4: ",kl)
    # 5 - return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term
    #print(N/batch_size * np.sum(gen - kl))

    return (N/batch_size) * np.sum(gen - kl, axis=-1)


if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0) # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [ latent_dim ] + [ n_units for i in range(n_layers) ] + [ data_dim ]
    rec_layer_sizes = [ data_dim ]  + [ n_units for i in range(n_layers) ] + [ latent_dim * 2 ]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)

    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)

    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params) 

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)

    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params

    def objective(flattened_combined_params):#flattened_combined_params:son los parametros del modelo generativo y los de la red que se utiliza para q

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
    
        on = train_images[ data_idx ,: ] > npr.uniform(size = train_images[ data_idx ,: ].shape)
        images = train_images[ data_idx, : ] * 0.0
        images[ on ] = 1.0

        return vae_lower_bound(gen_params, rec_params, images) 

    # Get gradients of objective using autograd.

    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    
    t = 1

    # TODO write here the initial values for the ADAM parameters (including the m and v vectors)
    # you can use np.zeros_like(flattened_current_params) to initialize m and v
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)
    alpha = 0.001
    beta1= 0.9
    beta2= 0.999
    epsilon= 10**(-8)
    # We do the actual training
    elbo_values = []
    for epoch in range(num_epochs):

        elbo_est = 0.0
    
        for n_batch in range(int(np.ceil(N / batch_size))):

            batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))

            grad = objective_grad(flattened_current_params) #grad serian los gradientes ruidosos

            
            # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates
            #vamos completar este bucle
            
            t += 1
            m = beta1*m+(1-beta1)*grad
            v = beta2*v+(1-beta2)*grad**2
            m_est = m/(1-beta1**t)
            v_est = v/(1-beta2**t)
            #print(alpha*m_est*(np.sqrt(v_est)+epsilon))
            flattened_current_params += (alpha*m_est/(np.sqrt(v_est)+epsilon))

            elbo_est += objective(flattened_current_params)

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))
        elbo_values.append( elbo_est / np.ceil(N / batch_size))
    # We obtain the final trained parameters

    gen_params, rec_params = unflat_params(flattened_current_params)

    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images
    save_images(sigmoid(neural_net_predict(gen_params,npr.randn(25, latent_dim))), "Task_3_1")

    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model) 
    # and save them alongside with the original image using save_images
    post = neural_net_predict(rec_params, test_images[:10])
    sample_latent_variables_from_posterior_ = sample_latent_variables_from_posterior(post)
    gen = neural_net_predict(gen_params, sample_latent_variables_from_posterior_)
    save_images(np.r_[test_images[:10],sigmoid(gen)], "Task_3_2")


    num_interpolations = 5
    for i in range(num_interpolations):

        # TODO Generate 5 interpolations from the first test image to the second test image, 
        # for the third to the fourth and so on until 5 interpolations
        # are computed in latent space and save them using save images. 
        # Use a different file name to store the images of each iterpolation.
        # To interpolate from  image I to image G use a convex conbination. Namely,
        # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained by numpy.linspace
        # Use mean of the recognition model as the latent representation.

        post1 = neural_net_predict(rec_params, [test_images[2*i]])
        post2 = neural_net_predict(rec_params, [test_images[2*i+1]])

        D = np.shape(post1)[-1] // 2
        mean1 = post1[:, :D]
        mean2 = post2[:, :D]
        print(mean1.shape)

        s = np.linspace(0,1,25)
        #sample_latent_variables_from_posterior_1 = sample_latent_variables_from_posterior(mean1)
        #sample_latent_variables_from_posterior_2 = sample_latent_variables_from_posterior(mean2)

        z = np.repeat(mean1, 25,axis=0)*s[..., np.newaxis] + (1-s)[..., np.newaxis]*np.repeat(mean2, 25,axis=0)
                
        print(z.shape)
        gen = neural_net_predict(gen_params, z)
        save_images(sigmoid(gen), "Task_3_3_"+str(i))


    pass
