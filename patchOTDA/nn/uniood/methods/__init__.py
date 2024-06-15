from .source_only import SourceOnly

from .uniot import UniOT


"""
Note that each method class should be a subclass of SourceOnly.
"""
method_classes = {'SO': SourceOnly,
                 
                  'UniOT': UniOT,

                  }

