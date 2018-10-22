"""
The retriever of Document Retrieval System
"""

from collections import Counter
from operator import itemgetter
import math


class Retrieve:
    """
    The class for Retriever
    """

    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting

        # The total number of documents in the collection |D|
        self.total_doc = max([doc for value in index.values()
                              for doc in value])

        # The number of documents containing a term
        self.doc_freq = {term: len([doc_freq for doc_freq in index[term]])
                         for term in index}

        # The inverse document frequency log(|D| / doc_freq) of each term
        self.inv_doc_freq = {term: math.log10(self.total_doc / self.doc_freq[term])
                             for term in index}

        # The size of each document vector
        self.doc_vec_size = {doc: math.sqrt(sum([(index[term][doc] * self.inv_doc_freq[term]) ** 2
                                                 for term in [terms for terms, docs in index.items()
                                                              if doc in docs]]))
                             for doc in range(1, self.total_doc + 1)}

        # All the terms that appear in a document
        self.terms_in_doc = {doc: {term for term in [terms for terms, docs in self.index.items()
                                                     if doc in docs]}
                             for doc in range(1, self.total_doc + 1)}

        if term_weighting in ['tf', 'tfidf']:
            self.term_freq = {term: sum([freq for freq in self.index[term].values()])
                              for term in self.index}

            # self.paice_model = {doc: 1
            #                     for doc in range(1, self.total_doc + 1)}

    def forQuery(self, query):
        """
        Method performing retrieval for specified query

        :param query: The query to process
        :return: The top 10 most relevant documents to the query
        """

        if self.term_weighting == 'binary':
            return self.binary_scheme(query)
        elif self.term_weighting == 'tf':
            return self.tf_scheme(query)
        else:
            return self.tfidf_scheme(query)

    def binary_scheme(self, query):
        """

        :param query:
        :return:
        """

        candidate_doc = {doc: len(set(query) & self.terms_in_doc[doc])
                         for doc in self.terms_in_doc}

        candidate_doc = sorted(candidate_doc.items(), key=itemgetter(1), reverse=True)

        return [doc for doc, _ in candidate_doc][:10]

    def tf_scheme(self, query):
        """
        The term frequency weighting scheme

        :param query: The query to search
        :return: The 10 most relevant document to the query
        """

        # Get all words that appear in a document, doc
        # AND or OR the query and the all words
        # Calculate Paice model

        candidate_doc = []

        sorted_query = dict(sorted(query.items(), key=itemgetter(1), reverse=True))

        for term, freq in sorted_query.items():
            if term in self.index:
                doc_list = sorted(self.index[term].items(), key=itemgetter(1), reverse=True)
                candidate_doc += [doc for doc, count in doc_list]

        return [doc for doc, _ in Counter(candidate_doc).most_common(10)]

    def tfidf_scheme(self, query):
        """
        The TFIDF weighting scheme

        :param query: The query to search
        :return: The 10 most relevant document to the query
        """

        # Vector space model
        vec_space_model = {doc: sum([self.inv_doc_freq[term] * (self.term_freq[term] * self.index[term][doc])
                                     for term in [terms for terms in query if terms in self.term_freq and terms in self.terms_in_doc[doc]]]) / self.doc_vec_size[doc]
                           for doc in range(1, self.total_doc + 1)}

        vec_space_model = dict(sorted(vec_space_model.items(), key=itemgetter(1), reverse=False))

        print(vec_space_model)

        return [doc for doc in vec_space_model.keys()][:10]
