import sqlite3
import numpy as np
import torch
from lstm import lstm_seq2seq 
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.data import sampler

def get_data():
    con = sqlite3.connect("data/wjazzd.db")
    cur = con.cursor()
    
    x = cur.execute("SELECT melid, bass_pitch from beats")# WHERE (bass_pitch is not null)")
    bass = np.array(x.fetchall())
    inds = np.where(bass==None)
    nulls = np.unique(bass[inds, 0])
    r = cur.execute("SELECT melid, pitch, division, tatum, beat from melody WHERE division <= 4 AND tatum <= 4 AND beat <= 4")
    notes = np.array(r.fetchall()).astype('int32')
    mask = np.isin(notes[:, 0], nulls)
    notes = notes[~mask]

    maskb = np.isin(bass[:, 0], nulls)
    bass = bass[~maskb]
    return notes.astype('int32'), bass.astype('int32')

# takes in np array of notes and bass, returns the note_embs, bass_embs and all the encoders/decoders (bass_to_i, i_to_bass, notes_to_i, i_to_notes)
def preprocess(notes, bass):
    unique_notes = np.unique(notes[:, 1:], axis=0)
    unique_bass = np.unique(bass[:, 1])

    num_notes = unique_notes.shape[0]
    num_bass = unique_bass.shape[0]
    num_songs_notes = np.unique(notes[:, 0]).shape[0]
    num_songs_bass = np.unique(bass[:, 0]).shape[0]
    max_length_notes = np.max(np.bincount(notes[:, 0]))
    max_length_bass = np.max(np.bincount(bass[:, 0]))

    # encode and decode into indices
    
    bass_to_i = { b:i+1 for i, b in enumerate(unique_bass) }
    bass_to_i[0] = 0
    i_to_bass = { i+1:b for i, b in enumerate(unique_bass) }
    i_to_bass[0] = 0

    notes_to_i = { tuple(n):i+1 for i, n in enumerate(unique_notes) }
    notes_to_i[tuple(np.array([0, 0, 0, 0]))] = 0
    i_to_notes = { i+1:n for i, n in enumerate(unique_notes)}
    notes_to_i[0] = np.array([0, 0, 0, 0])
    
    bass_embed = np.zeros((num_bass+1, num_bass+1))
    indb = np.arange(num_bass+1)
    bass_embed[np.arange(num_bass+1), indb] = 1
    note_embed = np.zeros((num_notes+1, num_notes+1))
    indn = np.arange(num_notes+1)
    note_embed[np.arange(num_notes+1), indn] = 1

    rows = bass[:, 0]

    bass_reshape = np.zeros((num_songs_bass, max_length_bass+1))

    # reshape bass
    for i in range(num_songs_bass):
        x = np.squeeze(bass[np.where(rows==i+1), 1])
        x = np.pad(x, (0, max_length_bass+1 - len(x)))
        tmp=x
        bass_reshape[i, :] = tmp
    bass_ind = np.zeros_like(bass_reshape)

    # make bass into indices
    for i in range(bass_reshape.shape[0]):
        for j in range(bass_reshape.shape[1]):
            bass_ind[i, j] = bass_to_i[bass_reshape[i, j]]
    bass_ind = np.delete(bass_ind.astype(int), 0, axis=0)
    
    notes_reshape = np.zeros((num_songs_notes, max_length_notes+1, 4))
    for i in range(num_songs_notes):
        x = np.squeeze(notes[np.where(rows==i+1), 1:])
        desired_shape = (max_length_notes+1, 4)
        
        padding = np.subtract(desired_shape, x.shape)
        padding = np.where(padding < 0, 0, padding)

        x = np.pad(x, ((0, padding[0]), (0, padding[1])), mode='constant')
        tmp = x
        notes_reshape[i, :] = tmp

    notes_ind = np.zeros((notes_reshape.shape[0], notes_reshape.shape[1]))
    for i in range(notes_ind.shape[0]):
        for j in range(notes_ind.shape[1]):
            notes_ind[i, j] = notes_to_i[tuple(notes_reshape[i, j, :])]

    # get embeddings
    notes_ind = np.delete(notes_ind.astype(int), 0, axis=0)

    note_embs = note_embed[notes_ind]

    bass_embs = bass_embed[bass_ind]

    # turn to pytorch
    note_embs = torch.tensor(note_embs, dtype=torch.float32)
    bass_embs = torch.tensor(bass_embs, dtype=torch.float32)
    
    print(note_embs.shape, bass_embs.shape)
    
    torch.save(note_embs, 'note_embs.pt')
    torch.save(bass_embs, 'bass_embs.pt')
    coders = (bass_to_i, i_to_bass, notes_to_i, i_to_notes)
    return note_embs, bass_embs, coders

def main():
    notes, bass = get_data()
    note_features = 100
    bass_features = 100
    #_, _, coders = preprocess(notes, bass)
    note_embs = torch.load('note_embs.pt').permute(1,0,2)
    bass_embs = torch.load('bass_embs.pt').permute(1,0,2)

    # Define the desired shape after padding
    desired_shape = (100, 100, 70)

    # Calculate the amount of padding required for each dimension
    padding = [(0, 0), (0, 0), (0, note_embs.shape[2] - bass_embs.shape[2])]

    # Pad the array with zeros
    bass_embs = np.pad(bass_embs, padding, mode='constant')
    print(note_embs.shape, bass_embs.shape)
    print(note_embs.dtype, bass_embs.dtype)
    bass_embs = torch.tensor(bass_embs, dtype=torch.float32)
    model = lstm_seq2seq(input_size=bass_embs.shape[2], hidden_size=note_embs.shape[2])
    l = model.train_model(input_tensor=bass_embs, target_tensor=note_embs, n_epochs=5, target_len=note_embs.shape[0], batch_size=64)
    

main()