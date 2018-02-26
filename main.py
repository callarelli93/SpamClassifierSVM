#!/usr/bin/python
# -*- coding: utf-8 -*-
from email.parser import Parser
from collections import Counter
from email.message import Message
from nltk import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
import string
import re
import os
import csv
from random import shuffle
from pandas import DataFrame, read_csv
import pandas as pd
from sklearn import svm
import math

# List of stop words from nltk corpus
stoplist = str(stopwords.words('english'))


# The corpus of email has been extracted from "http://spamassassin.apache.org/old/publiccorpus"

PATH_TO_PUBLICCOURPUS = '/Users/andrea/Desktop/emailCorpus/' #FILL IN CUSTOM PATH

# PATHS TO FOLDERS WITH MESSAGES
spam = PATH_TO_PUBLICCOURPUS + 'spam/' # 500 spam messages
spam_2 = PATH_TO_PUBLICCOURPUS +'spam_2/' # 500 spam messages
easy_ham = PATH_TO_PUBLICCOURPUS +'easy_ham/' # 2551 non-spam messages
easy_ham_2 = PATH_TO_PUBLICCOURPUS +'easy_ham_2/' # 1400 non-spam messages
hard_ham = PATH_TO_PUBLICCOURPUS +'hard_ham/' # 250 non-spam messages with 'spammish' characteristics


def process_messages(PATH_TO_FOLDER, is_spam):

    # Loop though folder and process all text messages
    # returns list of 2-elements lists where:
    # - first entry = is_spam
    # - second entry = processed text

    email_list = []

    # Append 2-elements list [is_spam, ['text_to_be_process']]
    for root, dirs, files in os.walk(PATH_TO_FOLDER):
        for filename in files:
            if filename != '.DS_Store':
                str = PATH_TO_FOLDER + filename
                f = open(str,'r')
                email_list.append([is_spam, f.read()])


    # Process messages
    for idx, element in enumerate(email_list):
        try:
            email_list[idx][1] = process_raw_text(element[1])

        except (UnicodeDecodeError, TypeError):
                pass

    # For this time let's remove the non-processed messages, need to find a better way to parse them...
    email_list = [element for element in email_list if type(element[1]) == list]


    # Make words appear only once
    # email_list = [[element[0], list(set(element[1]))] for element in email_list]

    simple_list = []
    for item in email_list:
        l = []
        l.append(item[0])
        l.extend(item[1])
        simple_list.append(l)

    email_list = simple_list



    return email_list


def htmltrans(raw_html):

    # Function to convert markup tags to 'html'
    trans = re.compile(r'<[^>]+>')
    transtxt = re.sub(trans, 'html ', raw_html)
    return transtxt

def process_raw_text(raw_email_text):

    """
    This function processes a raw email string, divides it in single words.
    It also converts redundant features typical of spam email into words for cleaner ML data.

    :param raw_email_text: raw email string
    :return: is_multipart = Bool
    :return: (is_multipart = False) list of processed words from email
             (is_multipart = True) concatenated list of lists containing processed words from each email part

    """

    parser = Parser()
    text = parser.parsestr(raw_email_text)

    """
    IMPLEMENT MULTIPART (MIME) MESSAGES AT A LATER TIME
    
    #Get body of the email
    if email.is_multipart():
        print 'Email is Multipart, var email will be a concatenated list of word lists'
        is_multipart = True
        #Get body of email if it is multipart
        email = [part.get_payload() for part in email.get_payload()]
        #For loop to operate on the list
        for i, part in enumerate(email):
            #Remove html
            email[i] = cleanhtml(email[i])

            # Lowcase
            email[i] = email[i].lower()

            # Remove numbers
            email[i] = email[i].translate(None, '1234567890')

            # translate some common currency signs in 'crrnc'
            email[i] = email[i].replace('£', 'crrnc')
            email[i] = email[i].replace('$', 'crrnc')
            email[i] = email[i].replace('€', 'crrnc')

            # split string into list of strings
            email[i] = email[i].split()

            # translate http references to 'httpref'
            for idx, word in enumerate(part):
                if len(word) >= 4:
                    if word[0:4] == 'http':
                        part[idx] = 'httpref'
                else:
                    pass

            # translate mail addresses to 'mailaddr'
            for idx, word in enumerate(part):
                if '@' in word:
                    part[idx] = 'mailaddr'
                else:
                    pass

            # remove punctuation
            email[i] = [word.translate(None, string.punctuation) for word in email[i]]

            # remove white spaces
            email[i] = [word for word in email[i] if word != '']

            # Stem words
            email[i] = [str(ps.stem(word)) for word in email[i]]


    else:"""

    # Get body of mail
    text = text.get_payload()

    # Translate markup language to 'html' as spammers can leave markup language in mail
    text = htmltrans(text)

    # Lowcase
    text = text.lower()

    # Remove numbers
    text = text.translate(None, '1234567890')

    # Translate some common currency signs in 'dollar' (emails sample from american repository)
    #text = text.replace('£', 'dollar')
    #text = text.replace('$', 'dollar')
    #text = text.replace('€', 'dollar')

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

    #Stem words
    text = [str(ps.stem(word)) for word in text]

    # Remove stop words
    text = [word for word in text if word not in stoplist]

    # Remove non alphanumeric chars
    text = [word for word in text if word.isalpha() == True]

    # Remove words which are single chars
    text = [word for word in text if len(word) != 1]

    return text


def create_dictionary(all_mail, most_common):

    # Create dictionary of the most_common words in all_mail
    all_words = []
    for element in all_mail:
        all_words.extend(element[1:len(element)])

    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(most_common)
    dictionary = [item[0] for item in dictionary]

    return dictionary


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


def extract_features(all_mail, dictionary):

    shuffle(all_mail) # Shuffle mail to avoid high-variance model

    feature_matrix = []
    for item in all_mail:
        feature_vector = [item[0]]
        for word in dictionary:
            if word in item:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        feature_matrix.append(feature_vector)

    return feature_matrix


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


def extract_class_labels(features): return [x[0] for x in features]


def extract_X_matrix(features): return [x[1:len(x)] for x in features]


def accuracy(pred, y):

    misclass_vector = []
    for i in range(len(y)):
        if pred[i] == y[i]:
            misclass_vector.append(0)
        else:
            misclass_vector.append(1)

    return 1.0 - float(sum(misclass_vector))/float(len(y))


if __name__ == '__main__':

    """
    Simple SVM spam classifier. 
    To add mail uncomment appropriate lines.
    To write the processed data to csv uncomment appropriate lines.
    If reading features from a previously saved csv uncomment the converter! Viceversa make sure the converter is commented...
    
    FIDDLE WITH SVM PARAMETERS, MAIL AND HAVE FUN!!
    """

    print 'Processing mail, this may take a while...'

    # Process mail
    spam = process_messages(spam, True) # Length of 349
    spam_2 = process_messages(spam_2, True) # Length of 349
    #easy_ham = process_messages(easy_ham, False) #Length of 2338
    easy_ham_2 = process_messages(easy_ham_2, False) # Length of 1226
    hard_ham = process_messages(hard_ham, False) # Length of 188

    # Create mail list
    all_mail = [] # When selecting mail try to create equal amounts of spam and non-spam
    all_mail.extend(spam)
    all_mail.extend(spam_2)
    #all_mail.extend(easy_ham)
    all_mail.extend(easy_ham_2[0:500])
    all_mail.extend(hard_ham)

    # WRITE PROCESSED MAIL TO CSV
    #write_csv(all_mail, 'all_mail.csv')

    # Create dictionary with 3000 most common words
    print '{} messages processed, creating dictionary...'.format(len(all_mail))
    dictionary = create_dictionary(all_mail, 3000)
    print 'Dictionary created with {} words'.format(len(dictionary))

    # WRITE PROCESSED MAIL TO CSV
    #write_dictionary_csv(dictionary_list, 'dictionary.csv')
    #dictionary = open_csv('dictionary.csv')[0]
    #all_mail = open_csv('all_mail.csv')

    print 'Creating feature matrix...'
    feature_matrix = extract_features(all_mail, dictionary) # Every time the mail is randomly shuffled to guarantee more truthful predictions
    print 'Feature matrix created, with {} samples'.format(len(feature_matrix))

    # WRITE PROCESSED MAIL TO CSV
    #write_csv(feature_matrix, 'feature_matrix.csv')
    #feature_matrix = open_csv('feature_matrix.csv')
    #feature_matrix = converter(feature_matrix) # Uncomment line only if reading features from csv!!


    # Separate training and test sets in a 3 to 1 proportion
    sep = int(math.floor(len(feature_matrix) * 0.75))
    training_set = feature_matrix[0:sep]
    test_set = feature_matrix[sep:len(feature_matrix)]

    # Extract class labels vectors
    y_train = extract_class_labels(training_set)
    y_test = extract_class_labels(test_set)
    print 'Size of training set = {}, size of test set = {}'.format(len(y_train), len(y_test))

    # Extract matrix of features and train the SVM
    X_train = extract_X_matrix(training_set)
    X_test = extract_X_matrix(test_set)
    print 'Training SVM classifiers...'
    clf = svm.SVC(C=1.0, kernel='linear', cache_size=1000) # If possible set cache_size to high value
    clf.fit(X_train, y_train)
    print 'SVM ready to make predictions!'

    # Predict accuracy on traning set
    train_pred = clf.predict(X_train)
    train_acc = accuracy(train_pred, y_train)
    print 'Accuracy on training set: ', train_acc*100.0, '%'

    # Predict accuracy on test set
    test_pred = clf.predict(X_test)
    test_acc = accuracy(test_pred, y_test)
    print 'Accuracy on test set: ', test_acc*100.0, '%'









