import gc
import functools

import keras



def manage_memory(func):
    """
    Function decorator to exiplicitly end TensorFlow session and invoke garbage collection.
    IMPORTANT: Should be used at the end of the session to clear everything in the computational graph
               and start fresh, i.e. training models ones after another, to redfuce high memory consumption.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function
        result = func(*args, **kwargs)
        
        # Explicitly invoke garbage collection
        keras.backend.clear_session()
        gc.collect()
        
        return result
    return wrapper


def out_in_learning_phase(func):
    """
    Function decorator to exiplicitly det out and in of the learning mode in keras while performing inference,
    e.g. for validation or testing. THis reduces memory consumption.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        
        # Get out of the leaning phase
        keras.backend.set_learning_phase(0)
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Go back to the leaning phase
        keras.backend.set_learning_phase(1)
        
        return result
    return wrapper