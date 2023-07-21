"""
Author: Jack Mango
Contact: jackmango@berkeley.edu
Date: 2023-07-21
Description: This class is used to warn users of unexpected behavior when using
the neural network analysis.
"""

import warnings

class AnalysisWarning(Warning):

    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return repr(self.message)