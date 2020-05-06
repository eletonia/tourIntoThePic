# Blend Into the Picture

## Navigating through the code and submission
#### Code
All code lies within the python notebook Final_proj. The first section contains all of the fucntions used in this project. Additionally there is a utils.py file which contains code from a previous project from CS445 which allows users to select points within an image.

#### Report
BlendIntoThePicture_elengik2_vvanka2.pdf is the report detailing our motivation, approach, and steps we had taken to produce the attached results. Final_proj_jupyter_code.pdf is the pdf version of the jupyter notebook containing all of the code we've written 

#### Results
To view screenshots of the produced 3D rooms, there are folders for each of the examples we did as part of our submission (i.e. painting_output). The obj files and corresponding texture maps are located under the directory named the same as the image (i.e. painting).

##### Blender Flies
If you have Blender v2.83 and want to view our samples, we've added blend files corresponding to the sample images we did as part of the submission.

## Creating your own 3D Room
To create a 3D view of an image please follow the following instructions.

To ensure you can run the program without issues, make sure the following is installed in a conda environment
  - Python 3
  - Jupyter Notebook
  - matplotlib v3.1.1
  - numpy v1.17.0
  - opencv-contrib-python v4.1.0.25
  - scipy v1.3.1
  - scikit-image

## Getting Started

  - Start the jupyter notebook Final_proj
  - Scroll down to the section of Preparation, and set the img_file_name to the name of the image you want to use
  - Select three points in the image. first being top left of back wall, second being vanishing point, third being bottom right of the back wall
  - Set the focal length to some value
  - Run the code snippet containing coords3D=blendIntoPicture(mask_coords,img_file_name,focal)
  - Import the produced OBJ files into Blender. Make sure to select "Y Forward" and "Z up" as values in transform while importing
  - Set the camera to positin to (0,1,0) and viewing down -z axis. 
 
## Producing A Simple Foreground Object
If you would like to produce a single foreground image do the following
  - In the prompt to choose points in the image again, choose the top left and bottom right of the image you want set as part of foreground image.
  - In the next prompt, choose the bottom left and bottom right points of where on the ground plane you want to place the object. Ideally this should be on a plane parallel to the camera and backwall.
  - Execute the code snippet containing "foregroundObj(mask_coords,mask_fg_coords,mask_fg_ground_coords,img_file_name,coords3D,focal)" which will output a foreground obj file to import into Blender.
