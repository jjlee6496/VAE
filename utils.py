import pytorch_lightning as pl

def data_loader(fn):
    """
    decorator to handle the deprecation of data_loader from 0.7
    
    Args:
        fn (function): User defined data loader function
        return: A wrapper for the data_loader function
    """
    
    def func_wrapper(self):
        try:
            return pl.data_loader(fn)(self)
        
        except:
            return fn(self)
    
    return func_wrapper