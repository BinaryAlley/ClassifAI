from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from history.history_item import HistoryItem
from persistence.helper_persistence import get_prediction_filenames

class HistoryScreen(Screen):
    """ Screen for displaying history of predictions

    Manages a list of history prediction items and displays them

    Creation Date: 02nd of January, 2021

    Attributes:
        history_list (ObjectProperty): Holds the list of HistoryItem instances

    Methods:
        on_kv_post: Initializes the history list with prediction items
        share: Method for sharing a prediction (currently not implemented)
    """
    history_list = ObjectProperty(None)
  
    def on_kv_post(self, base_widget):
        """ Initializes the history list with prediction items. """
        files = get_prediction_filenames()
        for file in files:
            item = HistoryItem()
            item.img_path = file 
            self.history_list.add_widget(item)
        

    def share(self, path):
        """ Method for sharing a prediction.

        Currently not implemented.

        Args:
            path (str): Path of the item to be shared
        """
        pass
