from keras.models import load_model
from music21 import converter, instrument, note, chord, stream
import numpy as np
import random
import ngram
import glob
from heapq import nlargest
import operator
from tqdm import tqdm
from music21.ext import joblib
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

def get_msg(file):
    notes = []
    n = 0
    try:
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if not parts: 
            print("NOT PARTS")
            notes_to_parse = midi.flat.notes
        prev_offset = 0
        for elem in parts:
            notes_to_parse = elem.recurse()
            for element in notes_to_parse:
                new_offset = 0.5
                if prev_offset == element.offset:
                    new_offset = 0
                else:
                    new_offset = element.offset - prev_offset
                    if 0 < new_offset <= 0.5:
                        new_offset = 0.5
                    if 0.5 < new_offset <= 1:
                        new_offset = 1

                    if 1 < new_offset <= 1.5:
                        new_offset = 1.5
                    if 1.5 < new_offset:
                        new_offset = 2
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch) + "|" + str(new_offset) + "|" + str(element.octave) 
                    + "|" + str(elem[0]) )
                    n += 1
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder) + "|" + str(new_offset)
                                 + "|" + str(elem[0]))
                    n += 1

                prev_offset = element.offset
    except Exception as e:
        print("Что - то не так: ", e)
    return notes


def create_dataset(dataset, look_back=1):
    """
    Создает последовательность вида:
    X = [n-look_back, n-look_back+1, ...., n-1] и Y = [n]
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)



def create_midi(prediction_output, name):
    output_notes = []
    offset = 0
    zero_counter = 0
    instruments = {}
    for pattern in tqdm(prediction_output):
        s = pattern.split("|")
        pattern = s[0]
        octave = s[2]
        try:
            offset += float(s[1])
            if float(s[1]) == 0:
                zero_counter += 1
            else:
                zero_counter = 0
        except:
            print("error", s[1])
            continue
            
        inst = ''
        if len(s) > 3:
            if s[3] != '':
                s[3] = ''.join(s[3].split(' '))
                inst = s[3]
                if s[3] == "Voice":
                    inst = "Vocalist"
                if s[3] == "Brass":
                    inst = "BrassInstrument"
                if s[3] == "Fretless Bass":
                    inst = "FretlessBass"
            else:
                inst = "Piano"
        else:
            inst = "Piano"
        method_to_call = getattr(instrument, inst)()
        
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.volume.velocity = 60
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            if inst in instruments.keys():
                instruments[inst].append(new_chord)
            else:
                instruments[inst] = stream.Part()
                instruments[inst].insert(0, method_to_call)
                instruments[inst].append(new_chord)
            
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.volume.velocity = 60
            new_note.octave = octave
            if inst in instruments.keys():
                instruments[inst].append(new_note)
            else:
                instruments[inst] = stream.Part()
                instruments[inst].insert(0, method_to_call)
                instruments[inst].append(new_note)
    
    for elem in instruments:
        output_notes.append(instruments[elem])

    midi_stream = stream.Stream(output_notes)
    midi_stream.show("text")
    midi_stream.write('midi', fp=name)


def extended_this(model, trainX, trainY, look_back):
    """
    Продолжает последовательность в зависимости от type
    extend - продолжает
    remake - изменяет уже существующею
    continue - продолжает, удаляя оригинал
    """
    # Create dataset in comfortable type
    X = []
    #Y = []
    new_Y = []
    for i in range(len(trainX)):
        X.append(trainX[i])
    #for i in range(len(trainY)):
    #    Y.append(trainY[i])
    #new_l = len(X) * multi
    for i in range(len(trainX)):
        #if i % 100 == 0:
        #    print(i, "/", new_l)
        last_x = np.array([trainX[i]])
        last_y = model.predict_proba(last_x)
        top = nlargest(1, enumerate(last_y[0]), operator.itemgetter(1))
        top = top[0][0]
        last_y[0] = [0] * len(last_y[0])
        last_y[0][top] = 1
        new_Y.append(last_y[0])
    return new_Y


def proc(midi):
    notes = []
    print("Start processing")
    for file in glob.glob(midi):
        notes.extend(get_msg(file))

    print("Загрузка ngram, для поиска похожих нот...")
    G = joblib.load("encoders/ngram_classic_main2_5.sav")
    for i in range(len(notes)):
        #print(notes[i], G.find(notes[i]))
        notes[i] = G.find(notes[i])

    print("Берем из словаря коды для каждой ноты...")
    encoder = LabelBinarizer()
    encoder.fit(notes)
    encoder = joblib.load("encoders/LabelBinarizer_classic_main2_5.sav")
    data = encoder.transform(notes)

    print("Создаем датасет...")
    look_back = 2
    trainX, trainY = create_dataset(data, look_back)

    print('Загружаем сеть...')
    model = load_model("models/Classic.h5")

    print("Генерируем...")
    Y = extended_this(model=model, trainX=trainX, trainY=trainY, look_back=look_back)

    print("Расшифруем полученые данные в мелодию...")
    new_notes = []
    text_labels = encoder.classes_
    for i in range(len(Y)):
        # Ищем индекс самой вероятной ноты
        pred = Y[i]
        top = nlargest(1, enumerate(pred), operator.itemgetter(1))
        top = top[0][0]
        # Загружаем из словаря по индексу ноту
        predicted_label = text_labels[top]
        #print(predicted_label)
        new_notes.append(predicted_label)

    sequence_length = 100
    midi_name = midi.split("/")
    create_midi(new_notes, midi_name[0] + "/proc_"+midi_name[1])
    print("Created: " + midi_name[0] + "proc_files/proc_"+midi_name[1])



