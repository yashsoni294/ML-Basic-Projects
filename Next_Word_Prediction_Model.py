# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-14T05:27:10.658164Z","iopub.execute_input":"2021-08-14T05:27:10.658512Z","iopub.status.idle":"2021-08-14T05:27:10.663428Z","shell.execute_reply.started":"2021-08-14T05:27:10.658484Z","shell.execute_reply":"2021-08-14T05:27:10.662513Z"}}
import numpy as np 
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential , load_model
from keras.layers import LSTM
from keras.layers.core import Dense , Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt 
import pickle
import heapq

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-14T05:27:12.537178Z","iopub.execute_input":"2021-08-14T05:27:12.537734Z","iopub.status.idle":"2021-08-14T05:27:12.547858Z","shell.execute_reply.started":"2021-08-14T05:27:12.537700Z","shell.execute_reply":"2021-08-14T05:27:12.546674Z"}}
path = "../input/pdf-file/1661-0.txt"
text = open(path).read().lower()
print('corpus length:' , len(text))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-14T05:27:14.372149Z","iopub.execute_input":"2021-08-14T05:27:14.372673Z","iopub.status.idle":"2021-08-14T05:27:14.393203Z","shell.execute_reply.started":"2021-08-14T05:27:14.372624Z","shell.execute_reply":"2021-08-14T05:27:14.392481Z"}}
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-14T05:27:16.012122Z","iopub.execute_input":"2021-08-14T05:27:16.012615Z","iopub.status.idle":"2021-08-14T05:27:16.039349Z","shell.execute_reply.started":"2021-08-14T05:27:16.012546Z","shell.execute_reply":"2021-08-14T05:27:16.038642Z"}}
unique_words = np.unique(words)
unique_word_index = dict((c,i) for i ,c in enumerate(unique_words))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-14T05:29:30.052195Z","iopub.execute_input":"2021-08-14T05:29:30.052696Z","iopub.status.idle":"2021-08-14T05:29:30.570591Z","shell.execute_reply.started":"2021-08-14T05:29:30.052664Z","shell.execute_reply":"2021-08-14T05:29:30.569917Z"}}
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
print(prev_words[0])
print(next_words[0])

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:29:32.212200Z","iopub.execute_input":"2021-08-14T05:29:32.212538Z","iopub.status.idle":"2021-08-14T05:29:32.947263Z","shell.execute_reply.started":"2021-08-14T05:29:32.212508Z","shell.execute_reply":"2021-08-14T05:29:32.946343Z"}}
X = np.zeros((len(prev_words),WORD_LENGTH , len(unique_words)) , dtype = bool)
Y = np.zeros((len(next_words), len(unique_words)) , dtype = bool)
for i , each_words in enumerate(prev_words):
    for j , each_word in enumerate(each_words):
        X[i , j , unique_word_index[each_word]] = 1
    Y[i , unique_word_index[next_words[i]]] = 1
print(X[0][0])

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:29:34.372071Z","iopub.execute_input":"2021-08-14T05:29:34.372417Z","iopub.status.idle":"2021-08-14T05:29:34.627822Z","shell.execute_reply.started":"2021-08-14T05:29:34.372382Z","shell.execute_reply":"2021-08-14T05:29:34.627087Z"}}
model = Sequential()
model.add(LSTM(128 , input_shape = (WORD_LENGTH , len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:29:36.052067Z","iopub.execute_input":"2021-08-14T05:29:36.052407Z","iopub.status.idle":"2021-08-14T05:30:52.825607Z","shell.execute_reply.started":"2021-08-14T05:29:36.052373Z","shell.execute_reply":"2021-08-14T05:30:52.824641Z"}}
optimizer = RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy' , optimizer = optimizer , metrics = ['accuracy'])
history = model.fit(X , Y , validation_split = 0.05 , batch_size = 128 , epochs = 2 ,shuffle = True).history

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:30:55.850492Z","iopub.execute_input":"2021-08-14T05:30:55.850859Z","iopub.status.idle":"2021-08-14T05:30:56.215922Z","shell.execute_reply.started":"2021-08-14T05:30:55.850829Z","shell.execute_reply":"2021-08-14T05:30:56.214921Z"}}
model.save('keras_next_word_model.h5')
pickle.dump(history , open("history.p" , "wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p" , "rb"))

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:30:57.932118Z","iopub.execute_input":"2021-08-14T05:30:57.932446Z","iopub.status.idle":"2021-08-14T05:30:58.093881Z","shell.execute_reply.started":"2021-08-14T05:30:57.932419Z","shell.execute_reply":"2021-08-14T05:30:58.092815Z"}}
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'] , loc = "upper left")

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:06.293403Z","iopub.execute_input":"2021-08-14T05:31:06.293818Z","iopub.status.idle":"2021-08-14T05:31:06.457786Z","shell.execute_reply.started":"2021-08-14T05:31:06.293782Z","shell.execute_reply":"2021-08-14T05:31:06.456762Z"}}
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'] , loc = "upper left")

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:09.132533Z","iopub.execute_input":"2021-08-14T05:31:09.132951Z","iopub.status.idle":"2021-08-14T05:31:09.141787Z","shell.execute_reply.started":"2021-08-14T05:31:09.132919Z","shell.execute_reply":"2021-08-14T05:31:09.140862Z"}}
def prepare_input(text):
    x = np.zeros((1 , WORD_LENGTH , len(unique_words)))
    for t , word in enumerate(text.split()):
        print(word)
        x[0 , t ,unique_word_index[word]] = 1
    
    return x
prepare_input("This is not a lack".lower())

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:13.307723Z","iopub.execute_input":"2021-08-14T05:31:13.308051Z","iopub.status.idle":"2021-08-14T05:31:13.313042Z","shell.execute_reply.started":"2021-08-14T05:31:13.308021Z","shell.execute_reply":"2021-08-14T05:31:13.312058Z"}}
def sample(preds , top_n = 3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n , range(len(preds)) , preds.take)

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:14.732700Z","iopub.execute_input":"2021-08-14T05:31:14.733047Z","iopub.status.idle":"2021-08-14T05:31:14.738876Z","shell.execute_reply.started":"2021-08-14T05:31:14.733017Z","shell.execute_reply":"2021-08-14T05:31:14.737850Z"}}
def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x , verbose = 0)[0]
        next_index = sample(preds , top_n = 1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        if len(original_text + completion) + 2 > len(original_text) and next_char == '':
            return completion

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:17.332324Z","iopub.execute_input":"2021-08-14T05:31:17.332660Z","iopub.status.idle":"2021-08-14T05:31:17.338422Z","shell.execute_reply.started":"2021-08-14T05:31:17.332630Z","shell.execute_reply":"2021-08-14T05:31:17.337528Z"}}
def predict_completions(text , n = 3):
    x = prepare_input(text)
    preds = model.predict(x , verbose = 0)[0]
    next_indices = sample(preds , n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:19.452220Z","iopub.execute_input":"2021-08-14T05:31:19.452544Z","iopub.status.idle":"2021-08-14T05:31:19.457256Z","shell.execute_reply.started":"2021-08-14T05:31:19.452515Z","shell.execute_reply":"2021-08-14T05:31:19.456636Z"}}
quotes = [
    "I had seen little of Holmes lately. My marriage had drifted us away.",
"from each other My own complete happiness.", "and the home-centred interests which rise up around the man who,first finds himself master",
"of his own establishment, were sufficient to absorb all my attention",
"while Holmes, who loathed every form of society with his whole Bohemian"
]

# %% [code] {"execution":{"iopub.status.busy":"2021-08-14T05:31:21.572335Z","iopub.execute_input":"2021-08-14T05:31:21.572825Z","iopub.status.idle":"2021-08-14T05:31:21.631377Z","shell.execute_reply.started":"2021-08-14T05:31:21.572793Z","shell.execute_reply":"2021-08-14T05:31:21.630149Z"}}
for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq ,1))
    print()
