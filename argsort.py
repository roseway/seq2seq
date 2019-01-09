import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense,concatenate,dot,Activation
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

def seq2seq(inp_len, inp_size, out_size):
	encoder_input = Input(shape=(inp_len,))
	decoder_input = Input(shape=(inp_len,))
	
	encoder = Embedding(inp_size, 64, input_length=inp_len, mask_zero=True)(encoder_input)
	encoder_outputs, state_h, state_c = LSTM(64, return_state=True)(encoder)
	
	decoder = Embedding(out_size, 64, input_length=inp_len, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True)(decoder, initial_state=[state_h, state_c])
	decoder = TimeDistributed(Dense(out_size, activation="softmax"))(decoder)
	
	model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model

def attention(inp_len, inp_size, out_size):
	encoder_input = Input(shape=(inp_len,))
	decoder_input = Input(shape=(inp_len,))
	
	encoder = Embedding(inp_size, 64, input_length=inp_len, mask_zero=True)(encoder_input)
	encoder, state_h, state_c = LSTM(64, return_sequences=True, unroll=True, return_state = True)(encoder)

	decoder = Embedding(out_size, 64, input_length=inp_len, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[state_h, state_c])

	attention = dot([decoder, encoder], axes=[2, 2])
	attention = Activation('softmax')(attention)
	context = dot([attention, encoder], axes=[2, 1])
	decoder_combined_context = concatenate([context, decoder])
	output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
	output = TimeDistributed(Dense(out_size, activation="softmax"))(output)

	model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model

def ptrnet(inp_len, inp_size, out_size):
	encoder_input = Input(shape=(inp_len,))
	decoder_input = Input(shape=(inp_len,))
	
	encoder = Embedding(inp_size, 64, input_length=inp_len, mask_zero=True)(encoder_input)
	encoder, state_h, state_c = LSTM(64, return_sequences=True, unroll=True, return_state = True)(encoder)

	decoder = Embedding(out_size, 64, input_length=inp_len, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[state_h, state_c])
	
	attention = dot([decoder, encoder], axes=[2, 2])
	output = Activation('softmax')(attention)

	model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model

def onehot(seq, size):
    res = []
    for v in seq:
        temp = [[0] * size for i in range(len(v))]
        for i in range(len(v)):
            temp[i][v[i]] = 1
        res.append(temp)
    return np.asarray(res)
	
n_steps = 5
x_file = 'data/x_{}.csv'.format( n_steps )
y_file = 'data/y_{}.csv'.format( n_steps )

split_at = 9000
batch_size = 100
hidden_size = 64
embedding_dim = 16
SEQ_LEN = 8

x = np.loadtxt( x_file, delimiter = ',', dtype = int )
y = np.loadtxt( y_file, delimiter = ',', dtype = int )

x=np.insert(x, 0, values=10, axis=1)
Y=np.insert(y, 0, values=-1, axis=1)

encoder_input=pad_sequences(x[:split_at], maxlen=SEQ_LEN,padding='post')
decoder_input=pad_sequences(Y[:split_at], maxlen=SEQ_LEN,padding='post')
decoder_output=pad_sequences(y[:split_at], maxlen=SEQ_LEN,padding='post')
decoder_output=to_categorical(np.asarray(decoder_output))
# decoder_output=onehot(decoder_output, SEQ_LEN)

print(encoder_input.shape, decoder_input.shape, decoder_output.shape)

test_input=pad_sequences(x[split_at:], maxlen=SEQ_LEN,padding='post')
test_output=pad_sequences(y[split_at:], maxlen=SEQ_LEN,padding='post')

model=attention(SEQ_LEN, 11, 6)
model.fit(x=[encoder_input, decoder_input], y=[decoder_output],verbose=1,batch_size=batch_size, validation_split=0.2,epochs=150)
model.save_weights('attention.hdf5')
model.load_weights('attention.hdf5')

def generate(encoder_input):
    decoder_input = np.zeros(shape=(len(encoder_input), SEQ_LEN))
    decoder_input[:,0] = -1
    for i in range(1, SEQ_LEN):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]

def acc(a,b):
	count=a.shape[0]
	n=0
	for i in range(count):
		if all(a[i]==b[i,:-1]):
			n+=1
		else:
			print("a is", a[i])
			print("b is", b[i, :-1])
	return n/count

def acc2(a,b):
	count=a.shape[0]
	n=0
	for i in range(count):
		if all(a[i,1:]==b[i, 1:-1]):
			n+=1
		else:
			print("a is", a[i])
			print("b is", b[i, :-1])
	return n/count

res=generate(test_input)
print(acc(res,test_output))
print(acc2(res,test_output))
