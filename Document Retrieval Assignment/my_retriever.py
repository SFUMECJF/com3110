"""
The retriever of Document Retrieval System
"""

import math


class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting

        self.total_doc = max([doc for value in self.index.values() for doc in value])
        self.doc_freq = {term: len([doc_freq for doc_freq in self.index[term]]) for term in self.index}
        self.inv_doc_freq = {term: math.log10(self.total_doc / self.doc_freq[term]) for term in self.index}
        self.doc_vec_size = {doc: (([self.calculate_tf_idf(doc, term) ** 2
                                                 for term in self.index
                                                 if doc in [term_doc for term_doc in self.index[term]]]))
                             for doc in range(1, self.total_doc + 1)}

        print(self.total_doc)
        print(self.doc_vec_size)

    # Method performing retrieval for specified query
    def forQuery(self, query):
        return range(1, 11)

    def calculate_tf_idf(self, doc, term):
        """
        Get the term frequency (tf) of a specific term / word

        :param term: The term / word to calculate for
        :param doc: The document ID
        :return: The term frequency (tf) of the term / word
        """

        return self.index[term][doc] * self.inv_doc_freq[term]

    def jaccard(self, d1, d2):
        """
        Compute similarity score for document pair

        :param d1: First document
        :param d2: Second document
        :return: Similarity score
        """

        wds1 = set(d1)
        wds2 = set(d2)

        over = under = 0

        for w in (wds1 | wds2):

            if w in d1 and w in d2:
                over += min(d1[w], d2[w])

            wmax = 0

            if w in d1:
                wmax = d1[w]
            if w in d2:
                wmax = max(d2[w], wmax)

            under += wmax

        if under > 0:
            return over / under
        else:
            return 0.0
