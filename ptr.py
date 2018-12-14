import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense,concatenate,dot,Activation
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

def seq2seq():
	encoder_input = Input(shape=(20,))
	decoder_input = Input(shape=(20,))
	encoder = Embedding(input_dict_size, 64, input_length=20, mask_zero=True)(encoder_input)
	encoder = LSTM(64, return_sequences=False)(encoder)
	decoder = Embedding(output_dict_size, 64, input_length=20, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True)(decoder, initial_state=[encoder, encoder])
	decoder = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)
	model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
	model.compile(optimizer='adam', loss='binary_crossentropy')
	model.summary()
	return model
	
def attention():
	encoder_input = Input(shape=(6,))
	decoder_input = Input(shape=(10,))
	encoder = Embedding(10, 64, input_length=6, mask_zero=True)(encoder_input)
	encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)
	encoder_last = encoder[:,-1,:]

	decoder = Embedding(7, 64, input_length=10, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

	attention= dot([decoder,encoder],axes=[2,2])
	attention = Activation('softmax')(attention)
	context=dot([attention,encoder],axes=[2,1])
	decoder_combined_context=concatenate([context, decoder])
	output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
	output = TimeDistributed(Dense(6, activation="softmax"))(output)

	model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model


def ptr():
	encoder_input = Input(shape=(6,))
	encoder = Embedding(10, 16, input_length=6, mask_zero=True)(encoder_input)
	encoder = LSTM(32, return_sequences=True, unroll=True)(encoder)
	encoder_last = encoder[:, -1, :]

	decoder_input = Input(shape=(10,))
	decoder = Embedding(7, 16, input_length=10, mask_zero=True)(decoder_input)
	decoder = LSTM(32, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

	attention = dot([decoder, encoder], axes=[2, 2])
	output = Activation('softmax')(attention)

	model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model

n_steps = 5
x_file = 'data/x_{}.csv'.format( n_steps )
y_file = 'data/y_{}.csv'.format( n_steps )

split_at = 9000
batch_size = 100
hidden_size = 64

x = np.loadtxt( x_file, delimiter = ',', dtype = int )
y = np.loadtxt( y_file, delimiter = ',', dtype = int )

x=np.insert(x, 0, values=-1, axis=1)
Y=np.insert(y, 0, values=-1, axis=1)
encoder_input=x[:split_at]
decoder_input=pad_sequences(Y[:split_at], maxlen=10,padding='post')
decoder_output=pad_sequences(y[:split_at], maxlen=10,padding='post')
decoder_output=to_categorical(np.asarray(decoder_output))

test_input=x[split_at:]
test_output=pad_sequences(y[split_at:], maxlen=10,padding='post')

model=ptr()
model.fit(x=[encoder_input, decoder_input], y=[decoder_output],verbose=1,batch_size=200, validation_split=0.2,epochs=150)
model.save_weights('ptrnet.hdf5')
model.load_weights('ptrnet.hdf5')

def generate(encoder_input):
    decoder_input = np.zeros(shape=(len(encoder_input), 10))
    decoder_input[:,0] = -1
    for i in range(1, 10):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]

def acc(a,b):
	count=a.shape[0]
	n=0
	for i in range(count):
		if all(a[i]==b[i,:-1]):
			n+=1
	return n/count
	
res=generate(test_input)
print (res)
print(acc(res,test_output))
