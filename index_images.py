import jax
import jax.numpy as jnp
import flaxmodels as fm
import pickle
import os.path
import numpy as np
from PIL import Image, ImageOps
import pathlib
import hashlib
import os

DATA_FNAME = "/data/features.pkl"
ALLOWED_FILETYPES = ['.jpg', '.jpeg']

def get_id(fname):
    return hashlib.md5(fname.encode("utf-8")).hexdigest()

def load_data():
    if not os.path.isfile(DATA_FNAME):
        return [], np.zeros([0, 2048], dtype=np.float32), {}
    with open(DATA_FNAME, 'rb') as f:
        data = pickle.load(f)
    return data['ids'], data['features'], data['fnames']

def save_data(ids, features, fnames):
    with open(DATA_FNAME, 'wb') as f:
        pickle.dump(dict(ids=ids, features=features, fnames=fnames), f)

class FeatureCalculator:
    def __init__(self, jit_fwd=True):
        key = jax.random.PRNGKey(0)
        dummy_img = jnp.zeros([256, 256, 3])
        self.model = fm.ResNet101(output='activations', ckpt_dir="/data")
        self.params = self.model.init(key, dummy_img)
        self.TARGET_IMAGE_SIZE = 256
        if jit_fwd:
            self.calculate_features = jax.jit(self._calculate_features)
        else:
            self.calculate_features = self._calculate_features

    def process_image(self, fname):
        img = self._load_and_preprocess(fname)
        return self.calculate_features(img)

    def _calculate_features(self, img):
        activations = self.model.apply(self.params, img, train=False)
        features = activations['block4_2']
        features = features.mean(axis=[-3, -2])[0] # average over xy dims, squeeze batch-dim
        return features

    def _load_and_preprocess(self, fname):
        img = Image.open(fname)
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        scaling_factor = self.TARGET_IMAGE_SIZE / np.mean(img.size)
        img = img.resize([int(x * scaling_factor) for x in img.size])
        img = jnp.array(img, dtype=float) / 255.0
        return img


def index_directory(directory, checkpoint_every=100, n_max=None):
    known_ids, known_features, known_fnames = load_data()

    all_fnames = []
    for fname in pathlib.Path(directory).rglob("*.*"):
        if fname.suffix.lower() in ALLOWED_FILETYPES:
            all_fnames.append(str(fname))

    feature_calc = FeatureCalculator()
    new_ids = []
    new_features = []
    skipped = 0
    processed = 0
    for i,fname in enumerate(all_fnames):
        fname_id = get_id(fname)
        if i % 10 == 0:
            print(f"{i} / {len(all_fnames)}")
        if fname_id in known_fnames:
            continue
            skipped += 1

        x = feature_calc.process_image(fname)
        new_ids.append(fname_id)
        new_features.append(x)
        known_fnames[fname_id] = fname
        processed += 1

        if (processed > 0) and (processed % checkpoint_every) == 0:
            known_ids = known_ids + new_ids
            known_features = np.concatenate([known_features, np.stack(new_features, axis=0)], axis=0)
            save_data(known_ids, known_features, known_fnames)
            new_ids = []
            new_features = []
        if n_max and (processed >= n_max):
            break

    if len(new_features) > 0:
        known_ids = known_ids + new_ids
        known_features = np.concatenate([known_features, np.stack(new_features, axis=0)], axis=0)
    save_data(known_ids, known_features, known_fnames)
    print(f"Processed {processed} images, skipped {skipped} images.")
    return known_ids, known_features, known_fnames


if __name__ == '__main__':
    import os
    ids, features, fnames = index_directory(os.environ["PHOTO_DIR"], checkpoint_every=100)

