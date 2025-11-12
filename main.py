import sys
from pathlib import Path
from fastai.vision.all import *

DATA_PATH = Path('xis_or_not')
MODEL_PATH = Path('model.pkl')


def train_model():
    if not DATA_PATH.exists():
        print(f"Error: Data directory '{DATA_PATH}' not found.")
        print("Please run 'python fetch_data.py' first.")
        return

    print("Loading data...")
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms()
    ).dataloaders(DATA_PATH, bs=32)

    print("Data loaded. Starting training...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    print(f"Training complete. Saving model to {MODEL_PATH}")
    learn.export(MODEL_PATH)
    print("Model saved.")


def test_model(image_path: str):
    if not MODEL_PATH.exists():
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python main.py train' first.")
        return

    if not Path(image_path).exists():
        print(f"Error: Test image '{image_path}' not found.")
        return

    print(f"Loading model '{MODEL_PATH}' to test image '{image_path}'...")
    learn = load_learner(MODEL_PATH)

    predicted, _, probs = learn.predict(PILImage.create(image_path))

    print(f"--- Prediction Results ---")
    print(f"Vocabulary: {learn.dls.vocab}")
    print(f"Probabilities: {probs}")
    if (float(probs[1]) > 0.7):
        print("\nIt is the famous Xis.")
    else:
        print("\nUnfortunately, it is not a Xis.")



def print_usage():
    print("Usage:")
    print("  python main.py train           # Train the model and save it")
    print("  python main.py test <image.jpg> # Load the model and test an image")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train_model()
    elif len(sys.argv) == 3 and sys.argv[1] == 'test':
        test_model(sys.argv[2])
    else:
        print_usage()