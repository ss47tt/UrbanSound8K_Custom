# UrbanSound8K_Custom

How do I process, train, validate, and test the UrbanSound8K dataset?

1. I downloaded the dataset online.

2. There are total of 10 classes of sounds, as follows:
   
   a. air conditioner
   
   b. car horn
   
   c. children playing
   
   d. dog bark
   
   e. drilling
   
   f. engine idling
   
   g. gun shot
   
   h. jackhammer
   
   i. siren
   
   j. street music

4. I used Python to find all the wav audio files that are 4 seconds:

   a. air conditioner (997 files)
   
   b. car horn (203 files)
   
   c. children playing (969 files)
   
   d. dog bark (675 files)
   
   e. drilling (804 files)
   
   f. engine idling (961 files)
   
   g. gun shot (16 files)
   
   h. jackhammer (803 files)
   
   i. siren (897 files)
   
   j. street music (1000 files)

5. I excluded car horn and gun shot classes and left with 8 classes.
  
6. I randomly picked 500 files from each class using Python.

7. In the 500 files of each class, 50 are used for testing, 450 are used for training and validating.

8. I converted the files of each class from wav to numpy files for training, validating, and testing.

9. Each of the wav file is resampled to 16000 Hz, and set the maximum frequency to 8000 Hz on the log mel spectrogram.

10. For training and validating, 450 files of each class is randomly split into 405 files for training and 45 files for validating.

11. For testing, 50 files of each class are evaluated, and confusion matrix and classification report are saved.

12. The accuracy is about 95 percent for testing.
