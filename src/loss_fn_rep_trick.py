from keras import backend as K


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch,1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#p stands for prior and q stands for posterior

def wasserstein_distance(mu_p, log_var_p, mu_q, log_var_q,wd_param):

    sigma_p = K.exp(log_var_p)            ## log variance to variance
    sigma_q = K.exp(log_var_q)
    term1 = K.mean(K.square(mu_q - mu_p))
    term2 = K.mean(sigma_q + sigma_p - 2.0*(sigma_p * sigma_q))
    return wd_param*(term1 + term2 )

def js_divergence(mu_post, log_var_post, inputs_k, js_param):

    var_post = K.exp(log_var_post)            ## log variance to variance
    term1 = K.mean(log_var_post)
    term2 = K.mean(K.square(mu_post - inputs_k) / var_post)

    return js_param*(term1 + term2)

def kl_divergence(mu_p, log_var_p, mu_q, log_var_q):
    
    sigma_p = K.exp(log_var_p)            ## log variance to variance
    sigma_q = K.exp(log_var_q)

    term1 = K.mean(log_var_p - log_var_q)
    term2 = K.mean(sigma_q/sigma_p)
    term3 = K.mean(K.square(mu_q - mu_p) / sigma_p)

    return (term1 + term2 + term3)

