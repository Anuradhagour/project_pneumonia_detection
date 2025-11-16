# project_pneumonia_detection
Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli.
CXRs are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR, complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.
Automating Pneumonia screening in chest radiographs, providing affected area details through bounding box.
#The objective of this project is to build an algorithm to locate the position of inflammation in a medical image. The algorithm needs to locate lung opacities on chest radiographs automatically
The objective of the project is,
Learn to how to do build an Object Detection Model
Use transfer learning to fine-tune a model.
Learn to set the optimizers, loss functions, epochs, learning rate, batch size, checkpointing, early stopping etc.
Read different research papers of given domain to obtain the knowledge of advanced models for the given problem.
# **The input folder contains 4 important information**
stage_2_train_labels.csv - CSV file containing the patient id, bounding boxes and target label
stage_2_detailed_class_info.csv - CSV file containing the detail informaiton of patientid and the corresponding label
stage_2_train_images - directory contains train images in DICOM format
stage_2_test_images - directory contains test images in DICOM format
