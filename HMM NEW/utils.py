import numpy as np

def save_hmm_model(model, filename):
    np.savez(filename, startprob=model.startprob_, transmat=model.transmat_)

def load_hmm_model(npz_path):
    data = np.load(npz_path)
    model = hmm.MultinomialHMM(n_components=len(data['startprob']), n_iter=100)
    model.startprob_ = data['startprob']
    model.transmat_ = data['transmat']
    return model
