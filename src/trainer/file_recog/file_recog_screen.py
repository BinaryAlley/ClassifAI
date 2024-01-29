from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty
from plyer import filechooser

class FileRecogScreen(Screen):
    """ Screen for file recognition tasks

    Provides functionality to select a file using a file chooser dialog
    Stores the selected file's path in the `path` property

    Creation Date: 02nd of January, 2021

    Attributes:
        path (StringProperty): Stores the path of the selected file

    Methods:
        open_gallery: Opens the file chooser dialog
        handle_selection: Callback for file selection, updates `path`
    """
    path = StringProperty()

    def open_gallery(self):
        """ Opens a file chooser dialog for selecting a file """
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, path):
        """ Callback for file selection

        Args:
            path (list): Path of the selected file
        """
        self.path = str(path[0]) if path else ''
