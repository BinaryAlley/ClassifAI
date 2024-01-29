from kivymd.uix.boxlayout import MDBoxLayout
from kivy.properties import StringProperty

class HistoryItem(MDBoxLayout):
    """ A layout component representing a history item

    Used to display an item with its associated image path

    Creation Date: 02nd of January, 2021

    Attributes:
        img_path (StringProperty): Path to the image associated with the item
    """
    img_path = StringProperty()
