local cv = require 'cv'
require 'cv.objdetect'

local M = {}
local FaceDetector = torch.class('FaceDetector', M)

function FaceDetector:__init(XML_frontalHaarCascade, XML_profileHaarCascade)
	local frontalHaarCascade = 'haarcascades/haarcascade_frontalface_default.xml'
        local profileHaarCascade = 'haarcascades/haarcascade_profileface.xml'
        
        -- Init Viola-Jones face detector
        print('=> Initialising the face detector...')
        print('=> Looking for '..frontalHaarCascade..' ...')
        local command = io.popen('locate '..frontalHaarCascade, 'r')
        local locateOutput = command:read()
        local _, endIndex = locateOutput:find(frontalHaarCascade)
        local frontalDetectorParamsFile = locateOutput:sub(1, endIndex) or XML_frontalHaarCascade
	command:close()
	assert(paths.filep(frontalDetectorParamsFile),
		frontalHaarCascade..' not found! Try specifing one manually.')

	print('=> Looking for '..profileHaarCascade..' ...')
	command = io.popen('locate '..profileHaarCascade, 'r')
        locateOutput = command:read()
        _, endIndex = locateOutput:find(profileHaarCascade)
        local profileDetectorParamsFile = locateOutput:sub(1, endIndex) or XML_profileHaarCascade
	command:close()
        assert(paths.filep(profileDetectorParamsFile),
                profileHaarCascade..' not found! Try specifing one manually.')

	self.frontalFaceCascade = cv.CascadeClassifier{frontalDetectorParamsFile}
	self.profileFaceCascade = cv.CascadeClassifier{profileDetectorParamsFile}
end

function FaceDetector:detect(img)
	local grayImg = (image.rgb2y(img)*255):byte()
        image.display(grayImg)	

	-- Stage I (detect frontal)
	local faces_frontal = self.frontalFaceCascade:detectMultiScale{grayImg}
	
	-- Stage II (detect profile)
	local faces_profile = self.profileFaceCascade:detectMultiScale{grayImg}

	return {faces_frontal, faces_profile}
end

return M.FaceDetector
