import sys
import pickle
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, basename
import requests, io, tarfile


def create_dictionary(files_list):

    lexicons_dict = {'<PAD>': 0, '.': 1, '?': 2, '-': 3}

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        lexicons_dict[word.lower()] = len(lexicons_dict)
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=800):

    files = {}
    story_outputs = None
    masking = None
    check_mask = None
    stories_lengths = []
    answers_flag = False
    limit = length_limit

    for indx, filename in enumerate(files_list):

        test_flag = ('test' in filename)
        files[filename] = []
        story_inputs = None

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            stories_lengths.append(len(story_inputs))
                            if (len(story_inputs) <= limit) or test_flag:
                                files[filename].append({
                                    'inputs':story_inputs,
                                    'outputs': story_outputs,
                                    'question_mask': masking
                                })
                        story_inputs = []
                        story_outputs = []
                        masking = []
                        check_mask = 0

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                            check_mask += 1
                            if word == '.':
                                masking += [0]*check_mask
                                check_mask = 0
                            elif word == '?':
                                masking += [1]*check_mask
                                check_mask = 0
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])
                            masking.append(0)

                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')
            if not story_inputs is None:
                stories_lengths.append(len(story_inputs))
                if (len(story_inputs) <= limit) or test_flag:
                    files[filename].append({
                        'inputs': story_inputs,
                        'outputs': story_outputs,
                        'question_mask': masking
                    })
                story_inputs = []
                story_outputs = []
                masking = []
                check_mask = 0

    return files, stories_lengths


def bAbI_preprocessing(data_dir):

    download_url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
    download_dir = 'tasks_1-20_v1-2'

    r = requests.get(download_url)
    t = tarfile.open(fileobj=io.BytesIO(r.content), mode='r|gz')
    t.extractall('.')

    files_list = []
    task_dir = join(download_dir, 'en-10k')
    for entryname in listdir(task_dir):
        entry_path = join(task_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary)

    rmtree(download_dir)

    train_data_dir = join(data_dir, 'train')
    test_data_dir = join(data_dir, 'test')

    mkdir(data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    pickle.dump(lexicon_dictionary, open(join(data_dir, 'lexicon-dict.pkl'), 'wb'))

    train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            train_data.extend(encoded_files[filename])

    pickle.dump(train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    return None
