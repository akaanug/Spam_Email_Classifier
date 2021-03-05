from text_classifier import NaiveBayesTextClassifier

if __name__ == '__main__':
    tc = NaiveBayesTextClassifier()

    # Train Multinomial Naive Bayes model with MLE estimator:
    print("\n---- Training Multinomial Naive Bayes model with MLE estimator: ----")
    tc.calc_cond_prob(MLE=True)
    tc.test_model()

    # Train Multinomial Naive Bayes model with MAP estimator:
    print("\n---- Training Multinomial Naive Bayes model with MAP estimator: ----")
    tc.calc_cond_prob()
    tc.test_model()

    # Train Bernoulli Naive Bayes model with MLE estimator:
    print("\n---- Training Bernoulli Naive Bayes model with MLE estimator: ----")
    tc.calc_cond_prob(MLE=True, bernoulli=True)
    tc.test_model(bernoulli=True)
