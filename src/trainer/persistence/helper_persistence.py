import os

current_file_path = __file__[:__file__.rfind(os.path.sep)]
predictions_path = current_file_path[:len(current_file_path) - len("persistentce")] + os.path.sep + 'predicted' 

def get_prediction_filenames():
    """ Retrieves filenames of prediction results stored in the 'predicted' directory

    Creation Date: 10th of January, 2021
    
    Returns:
        list: A list of file paths to the prediction result files.
    """
    if os.path.exists(predictions_path):
        contents = os.listdir(predictions_path)
        files = []
        for content in contents:
            filepath = os.path.join(predictions_path, content)
            if os.path.isfile(filepath):
                files.append(filepath)
        return files
    else:
        os.mkdir(predictions_path)
        return get_prediction_filenames()