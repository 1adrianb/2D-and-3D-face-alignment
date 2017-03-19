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

local model
if opts.type == '2D' then 
    model = torch.load('models/2D-FAN.t7')
else
    model = torch.load('models/3D-FAN.t7')
end

if opts.device == 'gpu' then model = model:cuda() end
model:evaluate()

for i = 1, #fileList do
    local img = image.load(fileList[i].image)
    originalSize = img:size()
    
    img = utils.crop(img, fileList[i].center, fileList[i].scale, 256):view(1,3,256,256)
    if opts.device ~= 'cpu' then img = img:cuda() end

    local output = model:forward(img)[4]
    output:add(utils.flip(utils.shuffleLR(opts, model:forward(utils.flip(img))[4])))

    local preds_hm, preds_img = utils.getPreds(output, fileList[i].center, fileList[i].scale)

    if opts.mode == 'demo' then
        utils.plot(img, preds_hm:view(68,2))
        io.read()
    end

    if opts.mode == 'eval' then
        predictions = preds_img:clone()
        xlua.progress(i,#fileList)
    end
end

if opts.mode == 'eval' then
    predictions = torch.cat(predictions,1)
    local dists = utils.calcDistance(predictions,fileList)
    utils.calculateMetrics()
end