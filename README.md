# Blend Into the Picture

## Navigating through the code and submission


To create a 3D view of an image please follow the following instructions.

To ensure you can run the program without issues, make sure the following is installed in a conda environment
  - Python 3
  - Jupyter Notebook
  - matplotlib==3.1.1
  - numpy==1.17.0
  - opencv-contrib-python==4.1.0.25
  - scipy==1.3.1
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
