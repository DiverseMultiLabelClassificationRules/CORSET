from logzero import logger

class InvertedIndex:
    def __init__(self, S):
        self.S = S 
        self.Index = self.build_index()
        logger.debug('done')
        
    def build_index(self):
        # here S is the original dataset (original format)
        logger.debug('building inverted index...')
        return [set(self.S[:, i].nonzero()[0]) for i in range(self.S.shape[1])]
        
        

   
