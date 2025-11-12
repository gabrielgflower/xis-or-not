import time
from ddgs import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *


def search_images(keywords, max_images=300):
    print(f"Searching for: {keywords}")
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')


def fetch_and_process_images():
    searches = 'other', 'xis gaúcho'
    path = Path('xis_or_not')

    print(f"Starting data fetch. Saving to: {path.absolute()}")
    for o in searches:
        dest = (path / o)
        dest.mkdir(exist_ok=True, parents=True)

        print(f"\nDownloading images for: {o}")
        download_images(dest, urls=search_images(f'{o} photo', max_images=100))

        time.sleep(3)

        print(f"Resizing images in {dest}")
        resize_images(path / o, max_size=400, dest=path / o)

    print("\n Verifying all downloaded images")
    failed = verify_images(get_image_files(path))
    if failed:
        print(f"Removing {len(failed)} failed images.")
        failed.map(Path.unlink)
    else:
        print("All images verified successfully.")

    # folder 'other' should be populated with diverse images! Make sure to also clean up images that don't make sense.
    print(f"\n'other' folder should now be populated with diverse images! Make sure to also clean up images that don't make sense.")
    print(f"\nData fetching complete. Data is ready in '{path}' folder.")


if __name__ == "__main__":
    fetch_and_process_images()