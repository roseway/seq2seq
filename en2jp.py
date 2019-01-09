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
	encoder, state_h, state_c = LSTM(64, return_state = True)(encoder)
	
	decoder = Embedding(out_size, 64, input_length=inp_len, mask_zero=True)(decoder_input)
	decoder = LSTM(64, return_sequences=True)(decoder, initial_state=[state_h, state_c])
	decoder = TimeDistributed(Dense(out_size, activation="softmax"))(decoder)
	
	model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
	model.compile(optimizer='adam', loss='binary_crossentropy')
	model.summary()
	return model


def attention(inp_len, inp_size, out_size):
	encoder_input = Input(shape=(inp_len,))
	decoder_input = Input(shape=(inp_len,))
	
	encoder = Embedding(inp_size, 64, input_length=inp_len, mask_zero=True)(encoder_input)
	encoder, state_h, state_c = LSTM(64, return_sequences=True, unroll=True, return_state = True)(encoder)
	#encoder_last = encoder[:, -1, :]

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
	
def load_data(path):
	einp=[]
	dinp=[]
	out=[]
	with open(path) as f:
		for line in f:
			line=line.strip().split(',')
			einp.append(line[0])
			dinp.append('1'+line[1])
			out.append(line[1])
	return einp,dinp,out
	
train_path="data/train.csv"
test_path="data/test.csv"
seqlen=20

print("Loading data...")
train_einput,train_dinput,train_output=load_data(train_path)
test_input,__,test_output=load_data(test_path)

print("Processing data...")
# Input tokenizer
intk=Tokenizer(lower=False, char_level=True)
intk.fit_on_texts(train_einput)
input_dict_size=len(intk.word_index)

# Output tokenizer
ottk=Tokenizer(lower=False, char_level=True)
ottk.fit_on_texts(train_dinput)
output_dict_size=len(ottk.word_index)+1

train_einput=intk.texts_to_sequences(train_einput)
train_einput=pad_sequences(train_einput, maxlen=seqlen,padding='post')

train_dinput=ottk.texts_to_sequences(train_dinput)
train_dinput=pad_sequences(train_dinput, maxlen=20,padding='post')

train_output=ottk.texts_to_sequences(train_output)
train_output=pad_sequences(train_output, maxlen=seqlen,padding='post')
train_output=to_categorical(np.asarray(train_output))

test_input=intk.texts_to_sequences(test_input)
test_input=pad_sequences(test_input, maxlen=seqlen,padding='post')

test_output=ottk.texts_to_sequences(test_output)
test_output=pad_sequences(test_output, maxlen=seqlen,padding='post')

print(train_einput.shape, train_dinput.shape, train_output.shape)

model=attention(seqlen, input_dict_size, output_dict_size)
checkpoint = ModelCheckpoint('attention.h5', monitor='val_loss', verbose=1, save_best_only=True,
mode='auto')
callbacks_list = [checkpoint]
model.fit(x=[train_einput, train_dinput], y=[train_output],verbose=1,batch_size=200, validation_split=0.2,epochs=20,callbacks=callbacks_list)

model=load_model('attention.h5')

def generate(inp):
	dec=np.zeros(shape=(len(inp), seqlen))
	dec[:,0]=1
	for i in range(1,seqlen):
		output = model.predict([inp, dec]).argmax(axis=2)
		dec[:,i] = output[:,i]
	return dec[:,1:]

def acc(pre, gold):
	a=0
	for i in range(len(pre)):
		if all(pre[i]==gold[i][:-1]):
			a+=1
	return a/len(pre)
predictions = generate(test_input)
print(acc(predictions, test_output))





