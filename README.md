# court-vision
Automatically generate highlights from user uploaded basketball videos.

`detection.py` contains functions for both highlight detecting and generating. For now, it only works on local files.   
 
`generateImages.ipynb` generates images for model training from YouTube links. It can also save the YouTube video, as well as clipping the video into a shorter one for model testing (run `detection.py`)   

You can ignore `detect_track.ipynb`. I'm using it to develop ball tracking (compared to detecting, tracking is a more powerful option because it gives each detected objcet an unique id, but is harder to implement).  