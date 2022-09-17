## Case study
Generate room plan from input data
#### (Thought)-Process
- Looking at the data, "understanding" its structure, and what is desired,
  based on the given images of the input data and output image.
- Of course there's the option to do it manually through image editing,
  but we want an algorithmic automatic approach to do it.
- Leaving that aside, I would have first asked for previous knowledge
  about the problem, if there is some idea on how to approach it, if
  there are known techniques. These questions would be asked let's say
  to a "senior" (just someone with experience I guess). And even though
  you gave me the option to ask you questions, I didn't, just because
  it's the case study.
- My first initial thoughts were first line extraction through edge detection,
  and some kind of linear regression. Also in general the use of "computer vision
  techniques".
- Anyway, then there's always googling stuff. Even though lacking "proper niche
  terminology" to search, making it more complicated in some sense, I ended up
  finding some results like these:
  * [Line segment extraction based on seeded region growing](https://journals.sagepub.com/doi/pdf/10.1177/1729881418755245)
  * [Building Detection and Structure Line Extraction from Airborne LiDAR Data](https://www.researchgate.net/profile/Pai-Hui-Hsu/publication/267428229_Building_Detection_and_Structure_Line_Extraction_from_Airborne_LiDAR_Data/links/5711ed2f08aeebe07c024b64/Building-Detection-and-Structure-Line-Extraction-from-Airborne-LiDAR-Data.pdf)
  * [Stackoverflow post - Detecting corners from 2D point cloud data](https://stackoverflow.com/questions/59049990/how-can-i-detect-the-corner-from-2d-point-cloud-or-lidar-scanned-data) 
  * [Springer chapter abstract (no access)](https://link.springer.com/chapter/10.1007/978-3-540-36998-1_27)
  * And just waaaay more information. From scikit-learn, opencv documentation, multiple thesis, and stuff. Some is mentioned in the jupyter notebook.
- Research actually took me a long time, and also deciding how to start tackling the problem.

#### Execution process
Given basically by the jupyter notebook I'm also uploading with this file, both the notebook itself,
and the HTML version of it. Explanations of what I thought and did, and short analysis are also
written in the notebook itself. So please read it if you are interested in it. ;)

### How to execute
1. First install the requirements in your environment, with for example:
  ```bash
  pip install -r requirements.txt
  ```
2. Then, it's just simply executing the script, a few images will pop in probably your browser,
  and some will be written into the working directory. Also the output json file.
3. Inside the script there are some execution arguments that can be manually deactivated to False,
  they are on True by default, if desired. (Should have implemented through sys and command line,
  but didn't for now).

### End
Read conclusion on the jupyter notebook, but either way, with more guidance, and asking more questions
it would have been definitely less difficult. It was fun either way ;)

