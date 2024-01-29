from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp
from kivy.properties import ObjectProperty, StringProperty
from kivy.core.window import Window
from history.history_item import HistoryItem
from history.history_screen import HistoryScreen 
from menu.menu_screen import MenuScreen
from rt_recog.rt_recog_screen import RtRecogScreen
from file_recog.file_recog_screen import FileRecogScreen
from about.about_screen import AboutScreen
from recog_summary.recog_summary import RecogSummary

Window.size = (300, 500)

class RootLayout(MDBoxLayout):
    """ Root layout of the app

    Creation Date: 10th of January, 2021
    """
    manager = ObjectProperty(None)

class InfoLabel(MDBoxLayout):
    """ Information label component, displays a property and its value

    Creation Date: 10th of January, 2021

    Attributes:
        property (StringProperty): Name of the property to display
        value (StringProperty): Value of the property to display
    """
    property = StringProperty('property')
    value = StringProperty('value')

class WindowManager(ScreenManager):
    """ Manages different screens in the app

    Creation Date: 10th of January, 2021
    """
    pass

class AiApp(MDApp):
    """ Main application class

    Creation Date: 10th of January, 2021

    Methods:
        build: Initializes the root layout of the app
        change_screen: Changes the current screen of the app
    """

    def build(self):
        """ Initializes the root layout of the app. """
        root = RootLayout()
        self.manager = root.manager
        return root
    
    def change_screen(self, screen_key, direction='right'):
        """ Changes the current screen of the app.

        Args:
            screen_key (str): Key identifier of the screen to be displayed
            direction (str, optional): Direction of the screen transition. Defaults to 'right'.
        """
        self.manager.transition.direction = direction
        self.manager.current = screen_key