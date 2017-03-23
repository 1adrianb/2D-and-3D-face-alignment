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

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local fileList = utils.getFileList(opts)
local predictions = {}

local model = torch.load(opts.model)

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

    if opts.mode == 'demo' then
        utils.plot(img, preds_hm:view(68,2)*4)
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
