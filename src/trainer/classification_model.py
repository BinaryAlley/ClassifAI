class ClassificationM:
    """ Model for the Flowers classifications

    Creation Date: 30th of December, 2020
    
    Attributes:
        name (str): Common name of the flower
        kingdom (str): Biological kingdom classification
        clade1 (str): First clade classification
        clade2 (str): Second clade classification
        clade3 (str): Third clade classification
        order (str): Biological order classification
        family (str): Biological family classification
        subfamily (str): Biological subfamily classification
        tribe (str): Biological tribe classification
        genus (str): Biological genus classification

    Methods:
        __str__: Returns the common name of the flower
        __repr__: Returns the official string representation, which is the common name
    """
        
# region ==================================================================== CTOR =====================================================================================
    def __init__(self, _name, _kingdom, _clade1, _clade2, _clade3, _order, _family, _subfamily, _tribe, _genus):
        """ Initializes a new instance of the ClassificationM class

        Args:
            _name (str): Common name of the flower
            _kingdom (str): Biological kingdom classification
            _clade1 (str): First clade classification
            _clade2 (str): Second clade classification
            _clade3 (str): Third clade classification
            _order (str): Biological order classification
            _family (str): Biological family classification
            _subfamily (str): Biological subfamily classification
            _tribe (str): Biological tribe classification
            _genus (str): Biological genus classification
        """
        self.name = _name
        self.kingdom = _kingdom
        self.clade1 = _clade1
        self.clade2 = _clade2
        self.clade3 = _clade3
        self.order = _order
        self.family = _family
        self.subfamily = _subfamily
        self.tribe = _tribe
        self.genus = _genus
# endregion    

# region ================================================================== METHODS ====================================================================================
    def __str__(self):
        """ Called by str(object) and the built-in functions format() and print() to compute the 'informal' or nicely printable string representation of an object
        
        Args:
            self: The current instance of the ClassificationM class
        """
        return self.name

    def __repr__(self):
        """ Called by the repr() built-in function to compute the 'official' string representation of an object
        
        Args:
            self: The current instance of the ClassificationM class
        """
        return self.name
# endregion