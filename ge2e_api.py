import numpy as np
from encoder import inference as encoder
import sys
from pathlib import Path
import os

signup_dir = Path("encoder/saved_models/signed_up/")
encoder_model_fpath = "encoder/saved_models/1_backups/1_bak_090000.pt"

def signup(wav_fpath: Path, username, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    embed_fpath = signup_dir.joinpath(username+".npy")
    wav = encoder.preprocess_wav(str(wav_fpath))
    embed = encoder.embed_utterance(wav)
    if os.path.exists(embed_fpath):
        old_embed = np.load(embed_fpath)
        embed = old_embed + embed
        embed /= np.linalg.norm(embed, 2)
        os.remove(embed_fpath)
    np.save(embed_fpath, embed, allow_pickle=False)
    print(username+" signed up.")


def signin(wav_or_wavpath, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    wav = encoder.preprocess_wav(wav_or_wavpath)
    embed = encoder.embed_utterance(wav)
    embed = np.reshape(embed, [np.shape(embed)[0], 1]) # [emb_dim, 1]
    
    signed_spk_embs = list(signup_dir.glob("*.npy"))
    signed_spk_name = [_dir.stem for _dir in signed_spk_embs]
    signed_spk_embs = [np.load(str(_dir)) for _dir in signed_spk_embs]
    signed_spk_embs = np.array(signed_spk_embs) # [n, emb_dim]

    print(signed_spk_name)
    print(np.shape(signed_spk_embs), np.shape(embed))
    similar_score = np.matmul(signed_spk_embs, embed)
    similar_score = np.reshape(similar_score, [-1])
    sim_id = np.argmax(similar_score)
    sim_name = signed_spk_name[sim_id]
    for name, score in zip(signed_spk_name, similar_score):
        print(name, score)
    print("\nMatching name: ",sim_name)



if __name__ == "__main__":
    argv_len = len(sys.argv)
    if(argv_len<3):
        print("Usage:")
        print("    sign up: 'python3.5 ge2e_api.py signup xxx.wav username'")
        print("    sign in: 'python3.5 ge2e_api.py signin xxx.wav'")
        exit(-1)
    if sys.argv[1]=='signup':
        wav_path = Path(sys.argv[2])
        signup(wav_path, str(sys.argv[3]), encoder_model_fpath)
    if sys.argv[1]=='signin':
        wav_path = Path(sys.argv[2])
        signin(wav_path, encoder_model_fpath)
