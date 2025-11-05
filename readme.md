### Introduction

This is the code repository for the paper “Integrated AI Framework for Room-Temperature Atom Manipulation in Scanning Probe Microscopy.”
It contains the code used for atomic manipulation with four different AI models.


### Examples

- [predict_example.py](examples%2Fpredict_example.py)

It includes examples of image classification and object detection on scanned topography data using three different AI models.
- [atom_manipulation_decision_example.py](examples%2Fatom_manipulation_decision_example.py)

This is an example where AI models analyze the scanned images to identify individual Ag atoms and adjacent clean HUCs, and then determine the next position for atomic manipulation.

- [predict_manipulation_iz.py](examples%2Fpredict_manipulation_iz.py)

This example uses the Model4 architecture, which predicts the success of a manipulation event based on the I and Z curves obtained during the previous atomic manipulation.
When the model outputs True, the system automatically terminates the atomic manipulation phase and proceeds to perform a topographic scan to verify the manipulation result.