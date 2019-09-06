import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense,GRU

def create_model(num_tokens,pre_trained_weight=None,return_enc_out=False):
    latent_dim = 256
    # Encoder variables
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder = LSTM(latent_dim, return_state=True,activation="relu")
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder Variables
    decoder_inputs = Input(shape=(None, num_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Encoder - Decoder GAN Model (for training)
    model = Model(inputs=[encoder_inputs, decoder_inputs], 
                outputs=decoder_outputs)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])

    if pre_trained_weight is not None:
        model.load_weights(pre_trained_weight)

    # Encode model (for inference)
    encoder_model = Model(encoder_inputs,encoder_states)

    # Decoder model (for testing)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model