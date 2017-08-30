local py = require 'fb.python' -- Required for dlib

local M = {}
local FaceDetector = torch.class('FaceDetector', M)

function FaceDetector:__init()
	print('Initialising python libs...')
	local np = py.import('numpy')
	local dlib = py.import('dlib')
	print('Initialising detector...')
	local detector = dlib.get_frontal_face_detector()
	
	self.np = np
	self.detector = detector
end

function FaceDetector:detect(img)
	img = (img:clone()*255):byte():transpose(1,2):transpose(2,3) -- bring it in a pythonic format
	local py_img = py.ref(img)
	py_img.flags.writeable = 1
	local dets = self.detector(py_img,py.int(1))
	local detections = py.reval('[np.asarray([d.left(), d.top(), d.right(), d.bottom()]) for i, d in enumerate(dets)]',{dets=dets})	
	
	return py.eval(detections)
end

return M.FaceDetector
