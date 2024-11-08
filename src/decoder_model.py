from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def decoder(u_train, u_test, k_train, k_test, batch_size=32, latent_dim=1, intermediate_dims=(8, 5, 3), 
            learning_rate=0.001, epochs=100, checkpoint_path='saved_decoder.keras'):

    orginal_dim = u_train.shape[1]
    intermediate_dim_1, intermediate_dim_2, intermediate_dim_3 = intermediate_dims

    # Build physics-informed decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    inter_y1 = Dense(intermediate_dim_3, activation='tanh')(latent_inputs)
    inter_y2 = Dense(intermediate_dim_2, activation='tanh')(inter_y1)
    inter_y3 = Dense(intermediate_dim_1, activation='tanh')(inter_y2)
    outputs = Dense(orginal_dim, activation='linear')(inter_y3)

    # Instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='physics_informed_decoder')
    # Use Adam optimizer with the specified learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)
    decoder.compile(optimizer=adam_optimizer, loss='mean_squared_error')

    # Setup callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min',)
    # model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')


    # Fit the model
    history = decoder.fit(
        k_train, u_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(k_test, u_test),
        callbacks=[early_stopping, ModelCheckpoint],
        verbose=1
    )
    
    return history
