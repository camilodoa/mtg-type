# mtg-type

A Magic the Gathering card type classifier made w/ Keras and TensorFlow. Trained with a cross-entropy loss function for 25 epochs. Final model (`mtg-types.h5`) plateaus at around 70% accuracy with a validation accuracy of 50%.

## Getting started

Install dependencies!

```
pip3 install -r requirements.txt
```


## Running

```
python3 app.py
```

Running will:

1. Create `train/` and `test/` directories
2. Download a bunch of magic cards from scryfall into those directories
3. Train a model with Keras / Tensorflow
4. Test the model on images in `/trials`

If you want to skip any of these steps, go to `class Config()` in `util.py`
and edit the steps to your liking!
