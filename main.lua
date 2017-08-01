require 'torch'
require 'nn'
require 'nngraph'
require 'paths'

require 'image'
require 'xlua'
local utils = require 'utils'
local opts = require 'opts'(arg)

-- Load optional libraries
xrequire('cunn')
xrequire('cudnn')

require 'cudnn'

-- Load optional data-loading libraries
xrequire('matio') -- matlab
npy4th = xrequire('npy4th') -- python numpy

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local fileList = utils.getFileList(opts)
local predictions = {}

local model = torch.load(opts.model)
local modelZ
if opts.type == '3D-full' then
    modelZ = torch.load(opts.modelZ)
    if opts.device ~= 'cpu' then modelZ = modelZ:cuda() end
    modelZ:evaluate()
end

if opts.device == 'gpu' then model = model:cuda() end
model:evaluate()

for i = 1, #fileList do
    local img = image.load(fileList[i].image)
    if img:size(1)==1 then
        img = torch.repeatTensor(img,3,1,1)
    end
    originalSize = img:size()
    
    img = utils.crop(img, fileList[i].center, fileList[i].scale, 256):view(1,3,256,256)
    if opts.device ~= 'cpu' then img = img:cuda() end

    local output = model:forward(img)[4]:clone()
    output:add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img))[4])))
    local preds_hm, preds_img = utils.getPreds(output, fileList[i].center, fileList[i].scale)

    preds_hm = preds_hm:view(68,2):float()*4
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
--        preds_img = torch.cat(preds_img, depth_pred, 2)
    end

    if opts.mode == 'demo' then
        utils.plot(img, preds_hm)
    end

    if opts.save then
        torch.save(opts.output..'/'..paths.basename(fileList[i].image, '.'..paths.extname(fileList[i].image))..'.t7', preds_img)
    end

    if opts.mode == 'eval' then
        predictions[i] = preds_img:clone()+1.75
        xlua.progress(i,#fileList)
    end
end

if opts.mode == 'eval' then
    predictions = torch.cat(predictions,1)
    local dists = utils.calcDistance(predictions,fileList)
    utils.calculateMetrics(dists)
end
