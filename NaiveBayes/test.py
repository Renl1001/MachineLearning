# -*- coding:utf-8 -*-
from bayes import NaiveBayes


def loadDataSet():
    train_samples = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', ' and', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    test_samples = [['love', 'my', 'girl', 'friend'], ['stupid', 'garbage'],
                    ['Haha', 'I', 'really', "Love", "You"],
                    ['This', 'is', "my", "dog"]]
    train_classes = [0, 1, 0, 1, 0, 1]  # 0：good; 1:bad
    return train_samples, train_classes, test_samples


if __name__ == "__main__":
    train_samples, train_classes, test_samples = loadDataSet()

    clf = NaiveBayes()
    clf.train(train_samples, train_classes)
    # test:
    for item in test_samples:
        clf.classify(item)
