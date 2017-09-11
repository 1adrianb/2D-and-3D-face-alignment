local py = require 'fb.python' -- Required for plotting

-- Import python libraries
py.exec([=[
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
]=])

local utils = {}

local buffer = {}
function utils.drawGaussian(img, pt, sigma)
    local height, width = img:size(1), img:size(2)

    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    if (ul[1] > width or ul[2] > height or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1

    if not buffer[size] then
        buffer[size] = image.gaussian(size):float()
    end
    local g = buffer[size]
    
    -- Usable gaussian range
    local g_x = {math.max(1, 2-ul[1]), math.min(size, size + (width - br[1]))}
    local g_y = {math.max(1, 2-ul[2]), math.min(size, size + (height - br[2]))}

    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], width)}
    local img_y = {math.max(1, ul[2]), math.min(br[2], height)}
    
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    return img
end

-- Transform the coordinates from the original image space to the cropped one
function utils.transform(pt, center, scale, res, invert)
    -- Define the transformation matrix
    local pt_new = torch.ones(3)
    pt_new[1], pt_new[2] = pt[1], pt[2]

    local h = 200*scale
    local t = torch.eye(3)
    t[1][1], t[2][2] = res/h, res/h
    t[1][3], t[2][3] = res*(-center[1]/h+0.5), res*(-center[2]/h+0.5)

    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_new):sub(1,2):int()
    return new_point
end

-- Crop based on the image center & scale
function utils.crop(img, center, scale, res)
    local l1 = utils.transform({1,1}, center, scale, res, true)
    local l2 = utils.transform({res,res}, center, scale, res, true)

    local pad = math.floor(torch.norm((l1 - l2):float())/2 - (l2[1]-l1[1])/2)

    local newDim = torch.IntTensor({img:size(1), l2[2] - l1[2], l2[1] - l1[1]})
    local newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
    local height, width = img:size(2), img:size(3)

    local newX = torch.Tensor({math.max(1, -l1[1]+1), math.min(l2[1], width) - l1[1]})
    local newY = torch.Tensor({math.max(1, -l1[2]+1), math.min(l2[2], height) - l1[2]})
    local oldX = torch.Tensor({math.max(1, l1[1]+1), math.min(l2[1], width)})
    local oldY = torch.Tensor({math.max(1, l1[2]+1), math.min(l2[2], height)})

    newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))

    newImg = image.scale(newImg,res,res)
    return newImg
end

function utils.getPreds(heatmaps, center, scale)
    if heatmaps:nDimension() == 3 then heatmaps = heatmaps:view(1, unpack(heatmaps:size():totable())) end

    -- Get locations of maximum activations
    local max, idx = torch.max(heatmaps:view(heatmaps:size(1), heatmaps:size(2), heatmaps:size(3) * heatmaps:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % heatmaps:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(heatmaps:size(3)):floor():add(1)

    for i = 1,preds:size(1) do        
        for j = 1,preds:size(2) do
            local hm = heatmaps[{i,j,{}}]
            local pX, pY = preds[{i,j,1}], preds[{i,j,2}]
            if pX > 1 and pX < 64 and pY > 1 and pY < 64 then
                local diff = torch.FloatTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                preds[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    preds:add(-0.5)

    -- Get the coordinates in the original space
    local preds_orig = torch.zeros(preds:size())
    for i = 1, heatmaps:size(1) do
        for j = 1, heatmaps:size(2) do
            preds_orig[i][j] = utils.transform(preds[i][j],center,scale,heatmaps:size(3),true)
        end
    end
    return preds, preds_orig+1
end

function utils.shuffleLR(x)
    local matched_parts = {
			{1,17},{2,16},{3,15},
            {4,14}, {5,13}, {6,12}, {7,11}, {8,10},
            {18,27},{19,26},{20,25},{21,24},{22,23},
            {37,46},{38,45},{39,44},{40,43},
            {42,47},{41,48},
            {32,36},{33,35},
			{51,53},{50,54},{49,55},{62,64},{61,65},{68,66},{60,56},
            {59,57}
		}

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(2, idx1, 1):clone()
        x:narrow(2, idx1, 1):copy(x:narrow(2, idx2, 1))
        x:narrow(2, idx2, 1):copy(tmp)
    end

    return x
end

function utils.flip(x)
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function utils.calcDistance(predictions,groundTruth)
    local n = predictions:size()[1]
    gnds = torch.Tensor(n,68,2)
    for i=1,n do
        gnds[{{i},{},{}}] = groundTruth[i].points
    end

    local dists = torch.Tensor(predictions:size(2),predictions:size(1))
    -- Calculate L2
    for i = 1,predictions:size(1) do
        for j = 1,predictions:size(2) do
            dists[j][i] = torch.dist(gnds[i][j],predictions[i][j])/groundTruth[i].bbox_size
	end
    end

    return dists
end

--http://stackoverflow.com/questions/640642/how-do-you-copy-a-lua-table-by-value
function table.copy(t)
   if t == nil then
      return {}
   end
   local u = { }
   for k, v in pairs(t) do u[k] = v end
   return setmetatable(u, getmetatable(t))
end

function utils.get_normalisation(bbox)
    local minX, minY, maxX, maxY = unpack(bbox:totable())
    
    local center = torch.FloatTensor{maxX-(maxX-minX)/2, maxY-(maxY-minY)/2}
    center[2] =center[2]-((maxY-minY)*0.12)
    
    return center, (math.abs(maxX-minX)+math.abs(maxY-minY))/195, math.sqrt((maxX-minX)*(maxY-minY))
end

function utils.bounding_box(iterable)
    local mins = torch.min(iterable, 1):view(2)
    local maxs = torch.max(iterable, 1):view(2)

    local center = torch.FloatTensor{maxs[1]-(maxs[1]-mins[1])/2, maxs[2]-(maxs[2]-mins[2])/2}
    center[2] =center[2]-((maxs[2]-mins[2])*0.12)

    return center, (maxs[1]-mins[1]+maxs[2]-mins[2])/195, math.sqrt((maxs[1]-mins[1])*(maxs[2]-mins[2])) --center, scale, normby
end

-- Requires fb.python
function utils.plot(surface, points, detectedFace)
py.exec([=[
if preds.shape[1]==2:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input.swapaxes(0,1).swapaxes(1,2))
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)

    if ('detected_face' in vars() or 'detected_face' in globals()) and (detected_face is not None):
        ax.add_patch(
            patches.Rectangle(
                (detected_face[0], detected_face[1]),
                detected_face[2],
                detected_face[3],
                fill=False,
                edgecolor="red"
            )
        )

    plt.show()
elif preds.shape[1]==3:
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(input.swapaxes(0,1).swapaxes(1,2))
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')

    if ('detected_face' in vars() or 'detected_face' in globals()) and (detected_face is not None):
        ax.add_patch(
            patches.Rectangle(
                (detected_face[0], detected_face[1]),
                detected_face[2],
                detected_face[3],
                fill=False,
                edgecolor="red"
            )
        )

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
    
    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()

]=],{input=surface:float():view(3,256,256), preds = points, detected_face = detectedFace})	
end

function utils.readpts(file_path)
	lines = {}
	for line in io.lines(file_path) do
		lines[#lines+1] = line
	end
	
	local num_points = tonumber(lines[2]:split(' ')[2])
	local pts = torch.Tensor(num_points,2)
	for i = 4,3+num_points do
		pts[{{i-3},{}}] = torch.Tensor{lines[i]:split(' ')[1],lines[i]:split(' ')[2]}
	end
	
	return pts
end

function utils.loadUnkownFile(filePath)
    local fileData = nil
    if paths.filep(filePath..'.t7') then
        fileData = torch.load(filePath..'.t7')
    elseif paths.filep(filePath..'.mat') then
        fileData = matio.load(filePath..'.mat')
    elseif paths.filep(filePath..'.npy') then
        fileData = npy4th.loadnpy(filePath..'.npy')
    elseif paths.filep(filePath..'.pts') then
        fileData = utils.readpts(filePath..'.pts')
    end
    return fileData
end

function utils.getFileList(opts)
    print('Scanning directory for data...')
    local data_path = opts.input
    local filesList = {}
    local requireDetectionCnt = 0
    for f in paths.files(data_path, function (file) return file:find('.jpg') or file:find('.png') end) do
        -- Check if we have .t7, .mat, .npy or .pts file
        local pts = utils.loadUnkownFile(paths.concat(data_path,f:sub(1,#f-4)))
	    local data_pts = {}

        if pts ~= nil then
            local center, scale, normby = utils.bounding_box(pts)
            data_pts.image = data_path..f
            data_pts.scale = scale
            data_pts.center = center
            data_pts.points = pts
            data_pts.bbox_size = normby

            filesList[#filesList+1] = data_pts

	elseif paths.filep(data_path..f:sub(1,#f-4)..'_bb.t7') then -- TODO: Improve this
            local bdBox = utils.loadUnkownFile(data_path..f:sub(1,#f-4)..'_bb') -- minX, minY, maxX, maxY
            local center, scale, normby = utils.get_normalisation(bdBox) 
            data_pts.image = data_path..f
            data_pts.scale = scale
            data_pts.center = center
            data_pts.points = torch.zeros(68,2) -- holder for pts
            data_pts.bbox_size = normby

            filesList[#filesList+1] = data_pts
        elseif opts.detectFaces then
            data_pts.image = data_path..f
            data_pts.points = torch.zeros(68,2)

            filesList[#filesList+1] = data_pts
            requireDetectionCnt = requireDetectionCnt + 1
        end
    end
    print('Found '..#filesList..' images')
    print(requireDetectionCnt..' images require a face detector')
    return filesList, requireDetectionCnt
end

function utils.calculateMetrics(dists)
local errors = torch.mean(dists,1):view(dists:size(2))
py.exec([=[
axes1 = np.linspace(0,1,1000)
axes2 = np.zeros(1000)
print(errors.shape[0])
for i in range(1000):
    axes2[i] = (errors<axes1[i]).sum()/float(errors.shape[0])

plt.xlim(0,7)
plt.ylim(0,100)
plt.yticks(np.arange(0,110,10))
plt.xticks(np.arange(0,8,1))

plt.grid()
plt.title('NME (%)', fontsize=20)
plt.xlabel('NME (%)', fontsize=16)
plt.ylabel('Test Images (%)', fontsize=16)
plt.plot(axes1*100,axes2*100,'b-',label='FAN (Ours)',lw=3)
plt.legend(loc=4, fontsize=16)

plt.show()
print('AUC: ',np.sum(axes2[:70])/70)
]=],{errors = errors})
end

return utils
