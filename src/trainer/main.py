# Creation Date: 19th of December, 2020
# Purpose: Startup window

# region ================================================================== IMPORTS ====================================================================================
from ai_app import AiApp
from deep_neural_network_classifier import DeepNeuralNetwork
# endregion

# application entry point when ran from interpreter
if __name__ == '__main__':
    dnn = DeepNeuralNetwork()
    # for training a full fledged kers model:
    dnn.Train()
    # for training a tensorflow lite model:
    # if not dnn.isModelTrained:
    #    dnn.TrainDeepNeuralNetwork()
    # dnn.TestTrainedDeepNeuralNetwork()
    # print('Input image path: ')
    # print(dnn.PredictFlowerClass(input()))
    # entry point for the attempt to use Kivy AS UI
    # AiApp().run()
# endregion