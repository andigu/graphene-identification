# An Automated Monolayer Graphene Detector
We present a zero-configuration image analysis algorithm that can identify and extract monolayer graphene features from a single RGB image.
## Setup
Install `pipenv` with `python3 -m pip install pipenv`. Then a simple `pipenv shell` and `pipenv install` should suffice. 
## Usage
The bulk of the algorithm is held in `identify.py`. The main function, `monolayers`, takes as input an RGB image of graphene and returns an integer mask representing each individual piece of graphene.

For a demo of the algorithm in action, one can call `main.py`, which takes a list of files and annotates each of them with a perimeter around each identified piece of graphene as well as its physical dimensions. For example, to run the algorithm on `imgs/gr1.png` and `imgs/gr2.png`, run `python main.py imgs/gr1.png imgs/gr2.png`. Also supports regex so `python main.py imgs/*.png` is valid. The output is in the `annotated/` folder.
## Algorithm
For a more detailed description of the algorithm, read [here](https://andigu.github.io/Graphene_Identification.pdf).
