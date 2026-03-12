# You can use this to start an interactive session quickly by running "python -i qstart.py".
# You can also copy this file to start writing a new script quickly, or you can start a new script with "from qstart import *".

from qlens_helper import *

q = QLens()
(lens,ptsrc,ptimgdata) = q.ptimg_objects()  
(lens,src,pixsrc,imgdata) = q.pix_objects()   
(params,dparams) = q.param_objects()         

