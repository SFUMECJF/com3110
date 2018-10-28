"""
The retriever of Document Retrieval System
"""

from operator import itemgetter
import math


class Retrieve:
    """
    The class for Retriever
    """

    def __init__(self, index, term_weighting):
        """
        Create new Retrieve object storing index and term weighting scheme

        :param index: The index dictionary
        :param term_weighting: The term weighting scheme, binary, tf or tfidf
        """

        self.index = index
        self.term_weighting = term_weighting

        # The total number of documents in the collection |D|
        self.total_doc = max(doc for value in index.values() for doc in value)

        # All the terms that appear in a document
        self.terms_in_doc = {doc: {term for term in [terms for terms, docs in index.items()
                                                     if doc in docs]}
                             for doc in range(1, self.total_doc + 1)}

        if term_weighting == 'tf':
            # The size of each document vector for TF, 1.95 gives better results
            self.doc_vec_size = {doc: math.sqrt(sum(index[term][doc] ** 1.95
                                                    for term in self.terms_in_doc[doc]))
                                 for doc in range(1, self.total_doc + 1)}

        elif term_weighting == 'tfidf':
            # The number of documents containing a term
            self.doc_freq = {term: len(index[term]) for term in index}

            # The TFIDF value of each term in each document
            # Created for performance boost to prevent calculating the same values in the loop
            self.term_tfidf_in_doc = \
                {doc: {term: math.log(self.total_doc / self.doc_freq[term]) * index[term][doc]
                       for term in terms}
                 for doc, terms in self.terms_in_doc.items()}

    def forQuery(self, query):
        """
        Method performing retrieval for specified query

        :param query: The query to process
        :return: The top 10 most relevant documents to the query
        """

        candidate = self.get_candidate(query)

        similarity = {}

        if self.term_weighting == 'binary':
            """
            Binary weighting scheme

            Term weight is either 0 or 1, easy implementation and weak result
            """

            for doc in candidate:
                query_doc_product = sum(1 for term in query if term in self.terms_in_doc[doc])

                doc_vec_size = len(self.terms_in_doc[doc])

                similarity[doc] = query_doc_product / math.sqrt(doc_vec_size)

        elif self.term_weighting == 'tf':
            """
            Term frequency weighting scheme

            Term weight depends on its frequency in the specific document, for query,
            the term weight depends on its frequency in the query as well
            """

            for doc in candidate:
                same_terms = query.keys() & self.terms_in_doc[doc]

                query_doc_product = sum(self.index[term][doc] * query[term]
                                        for term in same_terms)

                similarity[doc] = query_doc_product / self.doc_vec_size[doc]

        else:
            """
            TFIDF weighting scheme

            Term weight depends on the inverse document frequency (idf), and the
            TFIDF (term frequency * inverse document frequency)
            """

            query_tfidf = {term: query[term] * math.log(self.total_doc / self.doc_freq[term])
                           for term in query
                           if term in self.index}

            for doc in candidate:
                same_terms = query.keys() & self.terms_in_doc[doc]

                # query[term] * tfidf * idf
                query_doc_product = sum(query_tfidf[term] * self.term_tfidf_in_doc[doc][term]
                                        for term in same_terms)

                doc_vec_size = math.sqrt(sum(tfidf ** 2
                                             for tfidf in self.term_tfidf_in_doc[doc].values()))

                similarity[doc] = query_doc_product / doc_vec_size

        ranked_doc = sorted(similarity.items(), key=itemgetter(1), reverse=True)

        return [x[0] for x in ranked_doc[:10]]

    def get_candidate(self, query):
        """
        Get a list of candidate documents that are related to the query

        :param query: The query to search for
        :return: The list of documents that contain at least one same word with query
        """

        candidate_docs = set()

        # Get all documents that contain more than one term from query
        for term in query:
            if term in self.index:
                candidate_docs.update(set(self.index[term].keys()))

        return candidate_docs
