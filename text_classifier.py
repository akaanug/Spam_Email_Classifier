import math
import read_files

import numpy as np


class NaiveBayesTextClassifier:
    def __init__(self):
        print("Fetching data...")
        self.vocabulary = read_files.get_vocabulary("data/vocabulary.txt")
        self.x_train = np.asarray(read_files.get_features("data/x_train.csv"))
        self.x_test = np.asarray(read_files.get_features("data/x_test.csv"))
        self.y_train = read_files.get_truth_labels("data/y_train.csv")
        self.y_test = read_files.get_truth_labels("data/y_test.csv")
        self.s, self.n = get_class_counts(self.y_train)
        print("Completed fetching data.")

        self.spam_ftr_probs = []  # Feature probabilities of spam mails
        self.ns_ftr_probs = []  # Feature probabilities of non-spam mails

        self.spam_mails, self.ns_mails = self.divide_mails()

    def test_model(self, bernoulli=False):
        print("Testing the model...")
        predictions = []

        # Predict each mail
        for test_mail in self.x_test:
            predictions.append(self.guess_class(test_mail, bernoulli))

        true_amt = 0  # Correct prediction amount
        false_amt = 0  # Incorrect prediction amount

        tp, fp, fn, tn = 0, 0, 0, 0  # True positive, false positive, false negative, true negative

        # Compare the y_test and our predictions to see the accuracy
        for i, class_prediction in enumerate(predictions):
            correct_class = self.y_test[i]
            if class_prediction == correct_class:
                true_amt = true_amt + 1

                if class_prediction == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                false_amt = false_amt + 1

                if class_prediction == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1

        print("True Positive (TP) = {}".format(tp))
        print("False Positive (FP) = {}".format(fp))
        print("False Negative (FN) = {}".format(fn))
        print("True Negative (TN) = {}".format(tn))

        print("Accuracy of the model: ", (true_amt / (true_amt + false_amt)) * 100, "%")
        print("Correct prediction amount: ", true_amt)
        print("Incorrect prediction amount: ", false_amt)

    # Guesses if the mail is spam or not
    def guess_class(self, mail, bernoulli=False):
        spam_prior = self.s / (self.s + self.n)
        ns_prior = 1 - spam_prior

        spam_prior = math.log(spam_prior)
        ns_prior = math.log(ns_prior)

        mail_wc = [0] * len(self.vocabulary)  # Word counts inside mail

        for i, word_count in enumerate(mail):
            if word_count > 0:
                mail_wc[i] = word_count if not bernoulli else 1  # Count is 1 if the bernoulli model is used

        mail_wc = np.asarray(mail_wc)

        #  Convert -inf's
        self.spam_ftr_probs = np.nan_to_num(self.spam_ftr_probs)
        self.ns_ftr_probs = np.nan_to_num(self.ns_ftr_probs)

        with np.errstate(over='ignore'):
            if bernoulli:  # Bernoulli Naive Bayes Model
                with np.errstate(divide='ignore'):
                    p_spam = spam_prior + np.sum(
                        np.log(2 * np.multiply(self.spam_ftr_probs, mail_wc) - mail_wc - self.spam_ftr_probs + 1))
                    p_ns = ns_prior + np.sum(
                        np.log(2 * np.multiply(self.ns_ftr_probs, mail_wc) - mail_wc - self.ns_ftr_probs + 1))
            else:  # Multinomial Naive Bayes Model
                p_spam = spam_prior + np.sum(np.multiply(self.spam_ftr_probs, mail_wc))
                p_ns = ns_prior + np.sum(np.multiply(self.ns_ftr_probs, mail_wc))

        return 1 if p_spam > p_ns else 0

    # Divide mails by their classes
    def divide_mails(self):
        print("Dividing mails by class...")
        spam_mails = []
        ns_mails = []

        # Divide mails by their classes
        for i, mail in enumerate(self.x_train):
            if self.y_train[i] == 0:
                ns_mails.append(self.x_train[i])
            else:
                spam_mails.append(self.x_train[i])

        spam_mails = np.asarray(spam_mails)
        ns_mails = np.asarray(ns_mails)

        return spam_mails, ns_mails

    #  Calculates and returns conditional probabilities for each feature (Multinomial Naive Bayes)
    #  MAP -> P(w | c) = ( count(w,c) + 1 ) / ( count(c) + |V| )
    #  MLE -> P(w | c) = ( count(w,c) ) / ( count(c) )
    def calc_cond_prob(self, MLE=False, bernoulli=False):
        print("Calculating conditional probabilities ({} Estimate)...".format("MLE" if MLE else "MAP"))
        vocab_len = len(self.vocabulary)

        spam_word_counts = [0] * vocab_len  # Word counts inside spam mails
        ns_word_counts = [0] * vocab_len  # Word counts inside non-spam mails

        # Feature probabilities of spams
        spam_ftr_probs = []

        # Feature probabilities of non-spams
        ns_ftr_probs = []

        # Sum columns (Occurrences of each word)
        for i, word in enumerate(self.vocabulary):
            spam_word_counts[i] = self.spam_mails[:, i].sum(axis=0)
            ns_word_counts[i] = self.ns_mails[:, i].sum(axis=0)

        spam_word_sum = sum(spam_word_counts)
        ns_word_sum = sum(ns_word_counts)

        # Now find conditional probabilities for each word
        for i in range(vocab_len):

            #  Find P(word | spam)
            if MLE:
                p_spam = ((spam_word_counts[i]) / (spam_word_sum))
            else:
                p_spam = ((spam_word_counts[i] + 1) / (spam_word_sum + vocab_len))

            if bernoulli:
                spam_ftr_probs.append(p_spam)
            else:
                if p_spam == 0:
                    spam_ftr_probs.append(float('-inf'))  # Avoid base exception
                else:
                    spam_ftr_probs.append(math.log(p_spam))

            #  Finding P(word | not spam)
            if MLE:
                p_ns = ((ns_word_counts[i]) / (ns_word_sum))
            else:
                p_ns = ((ns_word_counts[i] + 1) / (ns_word_sum + vocab_len))

            if bernoulli:
                ns_ftr_probs.append(p_ns)
            else:
                if p_ns == 0:
                    ns_ftr_probs.append(float('-inf'))  # Avoid base exception
                else:
                    ns_ftr_probs.append(math.log(p_ns))

        ns_ftr_probs = np.asarray(ns_ftr_probs)
        spam_ftr_probs = np.asarray(spam_ftr_probs)

        print("Finished calculating conditional probabilities.")

        self.spam_ftr_probs, self.ns_ftr_probs = spam_ftr_probs, ns_ftr_probs


# Returns P(Y=0) and P(y=1)
def get_class_counts(lst):
    one_c = lst.count(1)
    zero_c = lst.count(0)

    print("spams: ", one_c)
    print("hams: ", zero_c)

    pspam = one_c / (one_c + zero_c)
    pnormal = zero_c / (one_c + zero_c)

    return pspam, pnormal
