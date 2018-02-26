require 'torch'
require 'nn'
require 'nngraph'
require 'paths'
require 'image'
require 'xlua'

local utils = require 'utils'
local opts = require 'opts'(arg)

-- Load optional libraries
require('cunn')
require('cudnn')

-- Load optional data-loading libraries
matio = xrequire('matio') -- matlab
npy4th = xrequire('npy4th') -- python numpy

local FaceDetector = require 'facedetection_dlib'

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local fileList, requireDetectionCnt = utils.getFileList(opts)
local predictions = {}
local faceDetector = nil

if requireDetectionCnt > 0 then faceDetector = FaceDetector() end


--creat a cpu copy to clone model from it to gpu if limited gpu option is true
local cpuMemoryModel = torch.load(opts.model)
local model

if opts.limitedGpuMemory == 'true' and opts.device == 'gpu' then
  model = cpuMemoryModel:clone()
  
else 
  model = cpuMemoryModel
end

--if deveice spcified to be gpu 
if opts.device == 'gpu' then model = model:cuda() end


model:evaluate()


--if not limited Gpu memory and 3D-full prediction required then load the 3D depth prediction model to Gpu memory once
--other wise if limitedGpuMemory is true will load the depth model when needed and swap with th 2D prediction model to save some memory

local modelZ
local cpuMemoryModelZ

if opts.type == '3D-full' then 
    
  cpuMemoryModelZ = torch.load(opts.modelZ)
  
  if opts.device == 'gpu' and opts.limitedGpuMemory == 'false' then
      
    modelZ = cpuMemoryModelZ
    modelZ = modelZ:cuda()
    modelZ:evaluate()
		
  elseif opts.device == 'cpu'  then
  modelZ = cpuMemoryModelZ
  modelZ:evaluate()
  end
end

for i = 1, #fileList do

  collectgarbage()
  print("processing ",fileList[i].image)
  
  local img = image.load(fileList[i].image)
  
  -- Convert grayscale to pseudo-rgb
  if img:size(1)==1 then
    print("\n-- Convert grayscale to pseudo-rgb")
    img = torch.repeatTensor(img,3,1,1)
  end
   
  -- Detect faces, if needed
  local detectedFaces, detectedFace
  
  if fileList[i].scale == nil then
        
    detectedFaces = faceDetector:detect(img)      
    if(#detectedFaces<1) then 
      print('********skip ',fileList[i].image)
      goto continue 
      end -- When continue is missing
    
    -- Compute only for the first face for now
    fileList[i].center, fileList[i].scale =	utils.get_normalisation(detectedFaces[1])
    detectedFace = detectedFaces[1]
        
  end
  
  img = utils.crop(img, fileList[i].center, fileList[i].scale, 256):view(1,3,256,256)
    
  --cuda--
  if opts.device ~= 'cpu' then img = img:cuda() end
  
  local output = model:forward(img)[4]:clone()
  
  output:add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img))[4])))
  
  local preds_hm, preds_img = utils.getPreds(output, fileList[i].center, fileList[i].scale)

  preds_hm = preds_hm:view(68,2):float()*4
  
  --there is no need to save the output in the gpu now 
  output = nil 
  collectgarbage()
  
  --if limited gpu memory is true now it's the time to swap between 2D and 3D prediction models 
  if opts.limitedGpuMemory=='true' and opts.type == '3D-full' and opts.device == 'gpu' then
    model = nil
    collectgarbage()
    modelZ = (cpuMemoryModelZ:clone()):cuda()
    modelZ:evaluate()    
    collectgarbage()
    
  end
  
  --proceed to 3D depth prediction (if the gpu limited memory is false then the 3D prediction model would be already loaded at the gpu memory at initialize steps and the 2D model stay at its place)
  -- depth prediction
  if opts.type == '3D-full' then
    out = torch.zeros(68, 256, 256)
    for i=1,68 do
      if preds_hm[i][1] > 0 then
          utils.drawGaussian(out[i], preds_hm[i], 2)
      end
    end
    out = out:view(1,68,256,256)
    local inputZ = torch.cat(img:float(), out, 2)

    if opts.device ~= 'cpu' then inputZ = inputZ:cuda() end
    
    local depth_pred = modelZ:forward(inputZ):float():view(68,1) 
    
    preds_hm = torch.cat(preds_hm, depth_pred, 2)
    preds_img = torch.cat(preds_img:view(68,2), depth_pred*(1/(256/(200*fileList[i].scale))),2)
    
  end
  
  --put back the 2D prediction model and pop the 3D model
  if opts.limitedGpuMemory=='true' and opts.type == '3D-full' and opts.device == 'gpu' then
    modelZ = nil
    collectgarbage()    
    model = (cpuMemoryModel:clone()):cuda()
    model:evaluate()
  end

  if opts.mode == 'demo' then
      
    if detectedFace ~= nil then
        -- Converting it to the predicted space (for plotting)
        detectedFace[{{3,4}}] = utils.transform(torch.Tensor({detectedFace[3],detectedFace[4]}), fileList[i].center, fileList[i].scale, 256)
        detectedFace[{{1,2}}] = utils.transform(torch.Tensor({detectedFace[1],detectedFace[2]}), fileList[i].center, fileList[i].scale, 256)

        detectedFace[3] = detectedFace[3]-detectedFace[1]
        detectedFace[4] = detectedFace[4]-detectedFace[2]
    end
    utils.plot(img, preds_hm, detectedFace)
  
  end

  if opts.save then
      local dest = opts.output..'/'..paths.basename(fileList[i].image, '.'..paths.extname(fileList[i].image))
      if opts.outputFormat == 't7' then
        torch.save(dest..'.t7', preds_img)
      elseif opts.outputFormat == 'txt' then
        -- csv without header
        local out = torch.DiskFile(dest .. '.txt', 'w')
        for i=1,68 do
              if preds_img:size(2)==3 then
                  out:writeString(tostring(preds_img[{i,1}]) .. ',' .. tostring(preds_img[{i,2}]) .. ',' .. tostring(preds_img[{i,3}]) .. '\n')
              else
                out:writeString(tostring(preds_img[{i,1}]) .. ',' .. tostring(preds_img[{i,2}]) .. '\n')
              end
        end
        out:close()
      end
      xlua.progress(i, #fileList)
  end

  if opts.mode == 'eval' then
      predictions[i] = preds_img:clone() + 1.75
      xlua.progress(i,#fileList)
  end
	
  collectgarbage();

  ::continue::
      
end

if opts.mode == 'eval' then
    predictions = torch.cat(predictions,1)
    local dists = utils.calcDistance(predictions,fileList)
    utils.calculateMetrics(dists)
end
