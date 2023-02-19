import string
from csv import DictReader
from aux_functions import create_one_hot_encoding


def read_captions():
    with open('clotho_captions_subset.csv', newline='') as csv:
        reader = DictReader(csv, delimiter=';')
        captions = []
        for row in reader:
            row.pop('file_name')
            captions += list(row.values())
    return captions


def preprocess_captions(captions):
    # Remove punctuations
    exclude = string.punctuation
    captions = [''.join(c for c in s if c not in exclude) for s in captions]

    # Add start of sentence
    add = '<sos> '
    captions = [add + s for s in captions]

    # Remove possible empty strings
    captions = [s for s in captions if s]

    unique_words = set()
    for idx, caption in enumerate(captions):
        words = caption.split()
        length = len(words)
        if length == 14:
            pass
        elif length < 14:
            n_add = 14 - length
            captions[idx] += n_add * ' <eos>'
        else:
            word_list = words[:14]
            captions[idx] = ' '.join(word_list)

        captions[idx] += ' <eos>'
        caption = captions[idx]
        words = caption.split()
        for word in words:
            unique_words.add(word)

    unique_words = list(unique_words)

    oh_captions = [[create_one_hot_encoding(w, unique_words) for w in cap.split()] for cap in captions]

    return oh_captions, unique_words


def main():
    caps = read_captions()
    preprocess_captions(caps)


if __name__ == '__main__':
    main()
