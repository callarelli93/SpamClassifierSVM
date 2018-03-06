#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import string
import re
import os
import csv
from nltk.corpus import stopwords
from random import shuffle
from sklearn import svm
from email.parser import Parser
from collections import Counter
from nltk import PorterStemmer
ps = PorterStemmer()


# List of stop words from nltk corpus
stoplist = str(stopwords.words('english'))

# The corpus of email has been extracted from "http://spamassassin.apache.org/old/publiccorpus"

PATH_TO_PUBLICCOURPUS = '/Users/andrea/Desktop/PublicCorpus/'  # FILL IN CUSTOM PATH

# PATHS TO FOLDERS WITH MESSAGES
spam = PATH_TO_PUBLICCOURPUS + 'spam/'  # 500 spam messages
spam_2 = PATH_TO_PUBLICCOURPUS + 'spam_2/'  # 500 spam messages
easy_ham = PATH_TO_PUBLICCOURPUS + 'easy_ham/'  # 2551 non-spam messages
easy_ham_2 = PATH_TO_PUBLICCOURPUS + 'easy_ham_2/'  # 1400 non-spam messages
hard_ham = PATH_TO_PUBLICCOURPUS + 'hard_ham/'  # 250 non-spam messages with 'spammish' characteristics


def accuracy(pred, y):
    misclass_vector = []
    for i in range(len(y)):
        if pred[i] == y[i]:
            misclass_vector.append(0)
        else:
            misclass_vector.append(1)

    return 1.0 - float(sum(misclass_vector)) / float(len(y))


def htmltrans(raw_html):

    # Function to convert markup tags to 'html'
    trans = re.compile(r'<[^>]+>')
    transtxt = re.sub(trans, 'html ', raw_html)
    return transtxt


def process_raw_text(raw_text):
    """
    This function processes a raw email string, divides it in single words.
    It also converts redundant features typical of spam email into words for cleaner ML data.

    :param raw_email_text: raw email string
    :return: is_multipart = Bool
    :return: (is_multipart = False) list of processed words from email
             (is_multipart = True) concatenated list of lists containing processed words from each email part
    """


    parser = Parser()
    text = parser.parsestr(raw_text)

    # Get body of mail
    text = text.get_payload()

    # Translate markup language to 'html' as spammers can leave markup language in mail
    text = htmltrans(text)

    # Lowcase
    text = text.lower()

    # Remove numbers
    text = text.translate(None, '1234567890')

    # Translate some common currency signs in 'dollar' (emails sample from american repository)
    # text = text.replace('£', 'dollar')
    # text = text.replace('$', 'dollar')
    # text = text.replace('€', 'dollar')

    # Split string into list of strings
    text = text.split()

    # Translate http references to 'httpref'
    for idx, word in enumerate(text):
        if len(word) >= 4:
            if word[0:4] == 'http':
                text[idx] = 'httpref'
        else:
            pass

    # Translate mail addresses to 'mailaddr'
    for idx, word in enumerate(text):
        if '@' in word:
            text[idx] = 'mailaddr'
        else:
            pass

    # Remove punctuation
    text = [word.translate(None, string.punctuation) for word in text]

    # Remove redundant white spaces elements from the list
    text = [word for word in text if word != '']

    # Stem words
    text = [str(ps.stem(word)) for word in text]

    # Remove stop words
    text = [word for word in text if word not in stoplist]

    # Remove non alphanumeric chars
    text = [word for word in text if word.isalpha()]

    # Remove words which are single chars
    text = [word for word in text if len(word) != 1]

    return text


def converter(list):
    # Convert list of strings to list of values
    converted = []
    for item in list:
        l = []
        if item[0] == 'True':
            l.append(True)
        else:
            l.append(False)
        for i in item[1:len(item)]:
            if i == '1':
                l.append(1)
            else:
                l.append(0)
        converted.append(l)

    return converted


def write_csv(data, filename):
    with open(filename, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        if type(data[0]) == list:
            for item in data:
                wr.writerow(item)
        else:
            wr.writerow(data)


def open_csv(file):
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def count_spam_ham(mail):
    count_spam = 0
    count_ham = 0
    for item in mail:
        if item[0]:
            count_spam += 1
        else:
            count_ham += 1
    return [count_spam, count_ham]


class SpamFilter:

    def __init__(self):

        self.mail = []
        self.dictionary = None
        self.model = None

    # Load mail to the self.mail attribute
    def load_mail(self, PATH_TO_FOLDER, is_spam):
        print 'Loading mail...'
        # Loop though folder and converts each message into a list of words

        mail_list = []

        # Append 2-element list [is_spam, 'text_to_be_processed']]
        for root, dirs, files in os.walk(PATH_TO_FOLDER):
            for filename in files:
                if filename != '.DS_Store':
                    str = PATH_TO_FOLDER + filename
                    f = open(str, 'r')
                    mail_list.append(f.read())

        # Process raw text into a list of words
        for idx, element in enumerate(mail_list):
            try:
                mail_list[idx] = process_raw_text(element)
            except (UnicodeDecodeError, TypeError):
                pass

        # For this time let's remove the non-processed messages, need to find a better way to parse them...
        mail_list = [element for element in mail_list if type(element) == list]

        # Create list for each message where the first entry corresponds to 'is_spam'
        lst = []
        for item in mail_list:
            l = []
            l.append(is_spam)
            l.extend(item)
            lst.append(l)
        mail_list = lst

        print '\n{} messages successfully loaded'.format(len(mail_list))

        self.mail.extend(mail_list)

        count_spam = count_spam_ham(self.mail)[0]
        count_ham = count_spam_ham(self.mail)[1]

        print '\n{} messages loaded in total, {} are spam and {} are ham'.format(len(self.mail), count_spam, count_ham)

    # Create dictionary of the most_common words in the loaded set of messages
    def create_dictionary(self, most_common):

        spam = count_spam_ham(self.mail)[0]
        ham = count_spam_ham(self.mail)[1]

        if spam < ham:
            count = ham
            while count != spam:
                for idx, item in enumerate(self.mail):
                    if item[0]:
                        pass
                    else:
                        self.mail.remove(self.mail[idx])
                        count -= 1
                        break

        elif spam == ham: pass

        else:
            count = spam
            while count != ham:
                for idx, item in enumerate(self.mail):
                    if item[0]:
                        self.mail.remove(self.mail[idx])
                        count -= 1
                        break
                    else:
                        pass

        if spam > ham:
            print '\nToo much spam, the set has been reduced to equal amounts of {} spam and ham'.format(ham)
        elif spam == ham:
            pass
        else:
            print '\nToo much ham, the set has been reduced to equal amounts of {} spam and ham'.format(spam)

        all_words = []
        for element in self.mail:
            all_words.extend(element[1:len(element)])

        dictionary = Counter(all_words)
        dictionary = dictionary.most_common(most_common)
        dictionary = [item[0] for item in dictionary]

        self.dictionary = dictionary
        print '\nCreated dictionary with {} most common words.'.format(most_common)

    # Process mail into a feature matrix and train the SVM classifier,
    # storing the parameters in the attribute self.model
    def train_filter(self, split, C, cache_size):

        print '\nProcessing mail...'
        shuffle(self.mail)
        feature_matrix = []

        for item in self.mail:
            feature_vector = [item[0]]
            for word in self.dictionary:
                if word in item:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            feature_matrix.append(feature_vector)

        # Separating dataset into training and test set
        sep = int(math.floor(len(feature_matrix) * split))
        training_set = feature_matrix[0:sep]
        test_set = feature_matrix[sep:len(feature_matrix)]

        # Extract class labels vectors
        y_train = [x[0] for x in training_set]
        y_test = [x[0] for x in test_set]
        print '\nSize of training set = {}, size of test set = {}'.format(len(y_train), len(y_test))

        # Extract matrix of features and train the SVM
        X_train = [x[1:len(x)] for x in training_set]
        X_test = [x[1:len(x)] for x in test_set]

        print '\nTraining SVM classifiers...'
        self.model = svm.SVC(C=C, kernel='linear', cache_size=cache_size)  # If possible set cache_size to high value
        self.model.fit(X_train, y_train)

        print '\nTesting accuracy...'
        train_pred = self.model.predict(X_train)
        train_acc = accuracy(train_pred, y_train)
        print 'Accuracy on training set: ', train_acc * 100.0, '%'

        test_pred = self.model.predict(X_test)
        test_acc = accuracy(test_pred, y_test)
        print 'Accuracy on test set: ', test_acc * 100.0, '%'

        print '\n Ready to make predictions!!'

    # Make prediction over a sample mail
    def predict(self, PATH_TO_SAMPLE_MAIL):

        f = open(PATH_TO_SAMPLE_MAIL, 'r')
        sample_mail = f.read()
        sample_mail = process_raw_text(sample_mail)
        prediction = self.model.predict(sample_mail)

        if prediction:
            print 'The mail is SPAM'
        else:
            print 'The mail is HAM'




if __name__ == '__main__':

    sc = SpamFilter()
    sc.load_mail(spam, True)
    sc.load_mail(spam_2, True)
    sc.load_mail(easy_ham_2, False)
    sc.create_dictionary(3000)
    sc.train_filter(0.75, 1.0, 1000)



    
