import csv


# To read in our vocabulary
def get_vocabulary(path):
    vocab = {}

    with open(path) as infile:  # path
        for line in infile:
            word = ""
            num = 0
            if ',' in line:
                word = line.split(',', 1)[0]
                num = int(line.split(',', 1)[1])
            #else:
            #    word = line
            #    print("Error: word ", word, " does not have index")
            #    return

            vocab[num] = word

    return vocab


def get_truth_labels(path):
    labels = []
    with open(path) as infile:  # path
        for line in infile:
            labels.append(int(line))

    return labels


def get_features(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = tuple(reader)
        data = tuple(tuple(map(int, i)) for i in data)

    return data

