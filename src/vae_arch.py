
from keras.layers import Input, Dense,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from loss_fn_rep_trick import sampling,js_divergence, wasserstein_distance,kl_divergence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mse
from keras import backend as K


batch_size = 32
early_stopping = EarlyStopping(monitor='val_loss', patience=70, verbose=1, mode='min', restore_best_weights=True)

def vae(intermediate_dims=(8, 5, 3), model_path = 'saved_decoder.h5'):

    input_shape_x = (10, )
    input_shape_k = (1, )
    latent_dim =1

    intermediate_dim_1, intermediate_dim_2, intermediate_dim_3 = intermediate_dims

    inputs_k = Input(shape=input_shape_k, name='ground_truth')
    inputs_k_pr = Input(shape=input_shape_k,name ='prior mean')
    inputs_x = Input(shape=input_shape_x, name='temp_input')
    inter_x1 = Dense(intermediate_dim_1, activation='tanh', name='encoder_intermediate_1')(inputs_x)
    inter_x2 = Dense(intermediate_dim_2, activation='tanh', name='encoder_intermediate_2')(inter_x1)
    inter_x3 = Dense(intermediate_dim_3, activation='tanh', name='encoder_intermediate_3')(inter_x2)


    # q(z|x)
    z_mean = Dense(latent_dim, name='z_mean')(inter_x3)
    z_log_var = Dense(latent_dim, name='z_log_var')(inter_x3)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    physics_based_decoder = load_model(model_path, compile=False)

    # Freeze the weights of the physics-based decoder
    for layer in physics_based_decoder.layers:
        layer.trainable = False

    outputs = physics_based_decoder(z)  # Connect encoder output 'z' to the physics-based decoder

    decoder = Model(z, outputs, name='decoder')
    encoder_model = Model(inputs_x, [z_mean, z_log_var, z], name='encoder2')
   
    vae_model = Model([inputs_x, inputs_k_pr], outputs, name='vae_mlp_kl')
    vae_model_2 = Model([inputs_x,inputs_k, inputs_k_pr], outputs, name='vae_mlp')

    return  encoder_model,vae_model,vae_model_2


def vae_kl(u_tr_no,u_te_no,k_tr_prior,k_te_prior):
    global batch_size, early_stopping 
    encoder,vae_model,vae_model_2 = vae()
    recons_loss = mse(vae_model.inputs[0],vae_model.outputs)
    kl_loss = K.mean(kl_divergence(vae_model.inputs[1], K.constant([0.0]), encoder.outputs[0], encoder.outputs[1]))
    vae_loss_kl = K.mean(recons_loss+kl_loss)

    vae_model.add_loss(vae_loss_kl)
    optimizer = Adam(learning_rate=0.001) 
    vae_model.compile(optimizer=optimizer)

   
    model_checkpoint = ModelCheckpoint('saved_vae_KL.h5', monitor='val_loss', save_best_only=True, mode='min')
          
    # Train the model with validation data and callbacks
    history_vae = vae_model.fit([u_tr_no,k_tr_prior],u_tr_no,
                  epochs=1000,
                  batch_size=batch_size,
                  validation_data=([u_te_no,k_te_prior], u_te_no),
                 callbacks=[early_stopping, model_checkpoint])
    
    [z_mean, z_log_var, z] = encoder.predict(u_tr_no,batch_size=batch_size)

    [z_mean_t, z_log_var_t, z_t] = encoder.predict(u_te_no,batch_size=batch_size)

    
    return history_vae,z_mean,z_log_var,z_mean_t,z_log_var_t

def vae_js(u_tr_no,u_te_no,k_tr,k_te,k_tr_prior,k_te_prior):
    global batch_size, early_stopping 
    alpha_js = 0.5
    js_param = (1.0-alpha_js)/alpha_js
    encoder,vae_model,vae_model_2 = vae()

    recons_loss = mse(vae_model_2.inputs[0],vae_model_2.outputs)
    kl_loss = K.mean(kl_divergence(vae_model_2.inputs[1], K.constant([0.0]), encoder.outputs[0], encoder.outputs[1]))
    js_loss = K.mean(js_divergence(encoder.outputs[0], encoder.outputs[1],vae_model_2.inputs[1],js_param))
    vae_loss_js = K.mean(js_loss +recons_loss+kl_loss)

    vae_model_2.add_loss(vae_loss_js)
    optimizer = Adam(learning_rate=0.001) 
    vae_model_2.compile(optimizer=optimizer)

    
    model_checkpoint = ModelCheckpoint('saved_vae_jsd.h5', monitor='val_loss', save_best_only=True, mode='min')
          
    #Train the model with validation data and callbacks
    history_vae = vae_model_2.fit([u_tr_no,k_tr,k_tr_prior],u_tr_no,
                  epochs=1000,
                  batch_size=batch_size,
                  validation_data=([u_te_no,k_te,k_te_prior], u_te_no),
                 callbacks=[early_stopping, model_checkpoint])
    
    [z_mean, z_log_var, z] = encoder.predict(u_tr_no,batch_size=batch_size)
    [z_mean_t, z_log_var_t, z_t] = encoder.predict(u_te_no,batch_size=batch_size)


    return history_vae,z_mean,z_log_var,z_mean_t,z_log_var_t

def vae_wd(u_tr_no,u_te_no,k_tr,k_te,k_tr_prior,k_te_prior):

    global batch_size, early_stopping 
    
    alpha_js = 0.5
    js_param = (1.0-alpha_js)/alpha_js
    wd_param = 1
    encoder,vae_model,vae_model_2 = vae()

    recons_loss = mse(vae_model_2.inputs[0],vae_model_2.outputs)
    wd_loss = K.mean(wasserstein_distance(vae_model_2.inputs[2], K.constant([0.0]), encoder.outputs[0], encoder.outputs[1],wd_param))
    js_loss = K.mean(js_divergence(encoder.outputs[0], encoder.outputs[1],vae_model_2.inputs[1],js_param))
    vae_loss_wd = K.mean(js_loss +recons_loss+wd_loss)

    vae_model_2.add_loss(vae_loss_wd)
    optimizer = Adam(learning_rate=0.001) 
    vae_model_2.compile(optimizer=optimizer)

    model_checkpoint = ModelCheckpoint('saved_vae_wd.h5', monitor='val_loss', save_best_only=True, mode='min')
          
    # Train the model with validation data and callbacks
    history_vae = vae_model_2.fit([u_tr_no,k_tr,k_tr_prior],u_tr_no,
                  epochs=1000,
                  batch_size=batch_size,
                  validation_data=([u_te_no,k_te,k_te_prior], u_te_no),
                 callbacks=[early_stopping, model_checkpoint])
    
    [z_mean, z_log_var, z] = encoder.predict(u_tr_no,batch_size=batch_size)
    [z_mean_t, z_log_var_t, z_t] = encoder.predict(u_te_no,batch_size=batch_size)
    
    return history_vae,z_mean,z_log_var,z_mean_t,z_log_var_t