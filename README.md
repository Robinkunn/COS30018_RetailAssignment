Retail Object Detection Model
This repository contains a retail object detection model for identifying misplaced items and out-of-stock products. You can use the provided GUI for easy interaction or run predictions using Spyder.

NOTES
1. Make sure retailassignment is in the same directory as the TF2-py39 environment.
2. Update the paths in predictions.py and gui.py as necessary to match your directory structure.
3. The GUI application allows you to upload images and videos for object detection using the specified models.
4. Make sure the whole retailassignment folder is put under the path C:\Users\Public\COS30082 

Dependencies for GUI
-tkinter
-shutil
-os
-tensorflow
-numpy
-PIL (Python Imaging Library)
-cv2 (OpenCV)

Dependencies for Predictions on Spyder
-os
-time
-tensorflow
-object_detection.utils.label_map_util
-object_detection.utils.visualization_utils as viz_utils




===Instructions===

===RUNNING PREDICTIONS ON SPYDER===
1.Ensure you are in the TF2-py39 environment.

2.Launch the GPU-compatible virtual environment by opening a terminal and activating the environment:

3.Type Spyder in the terminal to open the Spyder IDE.

4.Copy and paste predictions.py from the retailassignment directory into Spyder.

5. Update line 5 to the path of the model chosen accordingly #example for Multiclass = (repo_dir_path = r'PATH TO retailassignment\Multiclass')




===RUNNING GUI:===
1. Ensure you are in the TF2-py39 environment.

2. Launch the GPU-compatible virtual environment by opening a terminal and activating the environment:

3. Navigate to the retailassignment directory:

4. Run the gui.py script:
python gui.py




===RUNNING EVALUATION:===
Go to the directory C:\Users\Public\COS30082\TF2-py39\Lib\site-packages\pycocotools

--FOR MULTICLASS:--
1. replace cocoeval.py with cocoeval.py under folder C:\Users\Public\COS30082\retailassignment\cocoevalmulti

2. Run the following:
 
python C:\Users\Public\COS30082\TensorFlow\models\research\object_detection\model_main_tf2.py --model_dir=C:\Users\Public\COS30082\retailassignment\Multiclass\models\tf2\my_ssd_mobilenet_v2 --pipeline_config_path=C:\Users\Public\COS30082\retailassignment\Multiclass\models\tf2\my_ssd_mobilenet_v2\pipeline.config --checkpoint_dir=C:\Users\Public\COS30082\retailassignment\Multiclass\training


--FOR ONECLASS:--
1. replace cocoeval.py with cocoeval.py under folder C:\Users\Public\COS30082\retailassignment\cocoevaloneclass

Run the following for misplaced:
python C:\Users\Public\COS30082\TensorFlow\models\research\object_detection\model_main_tf2.py --model_dir=C:\Users\Public\COS30082\retailassignment\Misplaced\models\tf2\my_ssd_mobilenet_v2 --pipeline_config_path=C:\Users\Public\COS30082\retailassignment\Misplaced\models\tf2\my_ssd_mobilenet_v2\pipeline.config --checkpoint_dir=C:\Users\Public\COS30082\retailassignment\Misplaced\training

Run the following for outofstock:
python C:\Users\Public\COS30082\TensorFlow\models\research\object_detection\model_main_tf2.py --model_dir=C:\Users\Public\COS30082\retailassignment\Outofstock\models\tf2\my_ssd_mobilenet_v2 --pipeline_config_path=C:\Users\Public\COS30082\retailassignment\Outofstock\models\tf2\my_ssd_mobilenet_v2\pipeline.config --checkpoint_dir=C:\Users\Public\COS30082\retailassignment\Outofstock\training




===Training===

---Installations---
1. cd C:\Users\Public\COS30082 
 
2. Launch the GPU-compatible virtual environment:
C:\Users\Public\COS30082\TF2-py39\Scripts\activate 

3. Create a new folder named TensorFlow under the directory: C:\Users\Public\COS30082 
(Skip this step if TensorFlow folder is already existed)

4. cd C:\Users\Public\COS30082\TensorFlow

5. Run the following command:
git clone https://github.com/tensorflow/models.git 
(Before running this command, ensure that the TensorFlow folder is empty)

6. Protobuf Installation/Compilation commands:
cd C:\Users\Public\COS30082\Tensorflow\models\research 

set PATH=%PATH%;C:\Users\Public\COS30082\bin 

C:\Users\Public\COS30082\protoc-3.15.8-win64\bin\protoc object_detection/protos/*.proto --python_out=. 

7. Uninstall the previous TF object detection version 
pip uninstall tensorflow-object-detection-api  

8. Install the Object Detection API:
copy object_detection\packages\tf2\setup.py . 
python -m pip install . 


---Training---
1. cd C:\Users\Public\COS30082 

2. Launch the GPU-compatible virtual environment:
C:\Users\Public\COS30082\TF2-py39\Scripts\activate 

3. cd C:\Users\Public\COS30082\retailassignment\(the class that you want to train, e.g. Multiclass/Outofstock/Misplaced)   

4. Run the setconfigurations.py:
python setconfigurations.py 

5. Run the training command:
python C:\Users\Public\COS30082\Tensorflow\models\research\object_detection\model_main_tf2.py --pipeline_config_path="models\tf2\my_ssd_mobilenet_v2\pipeline.config" --model_dir="training" –alsologtostderr 


