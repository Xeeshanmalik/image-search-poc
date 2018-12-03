try:

    from sklearn.metrics.pairwise import cosine_similarity

    class Similarity:

        def __find_similarity_for_model_1__(self):

            print(self.features)

        def __init__(self, features):
           self.features = features
           self.__find_similarity_for_model_1__()

except ImportError as E:
    raise E