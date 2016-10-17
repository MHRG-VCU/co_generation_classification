#************************************************************************
#      __   __  _    _  _____   _____
#     /  | /  || |  | ||     \ /  ___|
#    /   |/   || |__| ||    _||  |  _
#   / /|   /| ||  __  || |\ \ |  |_| |
#  /_/ |_ / |_||_|  |_||_| \_\|______|
#    
# 
#   Written by < Daniel L. Marino (marinodl@vcu.edu) > (2016)
#
#   Copyright (2016) Modern Heuristics Research Group (MHRG)
#   Virginia Commonwealth University (VCU), Richmond, VA
#   http://www.people.vcu.edu/~mmanic/
#   
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#  
#   Any opinions, findings, and conclusions or recommendations expressed 
#   in this material are those of the author's(s') and do not necessarily 
#   reflect the views of any other entity.
#  
#   ***********************************************************************
#
#   Description: simple vocabulary for coding text
#
#   ***********************************************************************


from nltk.stem.porter import PorterStemmer
import numpy as np
import pickle
import operator

class Vocabulary(object):     
    def __init__(self, text_list, max_size= None):
        ''' Builds a dictionary with the stemmed words extracted from a given list of words
        inputs:
            text_list: text with the words splitted in a list
        outputs:
            key_dict: dictionary with stemmed words as keys, and an index as values. 
        '''
        self.stemmer = PorterStemmer()
        # extracting the set of words in the string     
        self.key_dict = dict()
        key_hist = dict()
        
        key= 1
        for word in text_list:
            word = self.stemmer.stem(word) # this is the porter stemmer
            if word not in self.key_dict:
                self.key_dict[word] = key
                key_hist[word] = 1
                key += 1
            else:
                key_hist[word] += 1
        
        # Delete words with low appareances
        if max_size is not None:
            sorted_hist = sorted(key_hist.items(), key=operator.itemgetter(1), reverse= True)
            
            self.key_dict= dict((d[0], i) for d, i in zip(sorted_hist[:max_size], range(1,max_size+1)))
            
        self.key_dict['N/A'] = 0        
        
        self.vocabulary_size= len(self.key_dict)
        self.key_list = sorted(self.key_dict.items(), key=operator.itemgetter(1))
        
    def text2keys(self, text_list, ignore_unknown= False):
        text_id= list()

        for word in text_list:
            word = self.stemmer.stem(word)
            if word not in self.key_dict:
                if not ignore_unknown:
                    text_id.append(0)
            else:
                text_id.append(self.key_dict[word])

        return text_id

    def keys2text(self, text_id):
        text= ''
        for i in text_id:
            try:
                text= text + self.key_list[i][0]+' '
            except IndexError:
                text= text + 'N/A '
            
            #else:
            #    raise ValueError('Dictionary should be constructed with increasing values')
        return text
    
    def prob2char(self, probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation.
        It returns the character corresponding to the higest probability """
        return self.keys2text([c for c in np.argmax(probabilities, 1)])
    
    
                
class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings, vc):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        self.vc= vc
        
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()
        
    
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self.vc.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, self._text[self._cursor[b]]] = 1.0    # the letter pointed by the cursor is converted to 1-hot encoding
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size  # Here, the cursor is increased
        return batch
    
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch] # include last batch from previous array
        #batches = list()
        for step in range(self._num_unrollings):
            # each call of _next_batch() increases the cursor by one
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches