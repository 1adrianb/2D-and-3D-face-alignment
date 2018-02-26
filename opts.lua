local function parse( arg )
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Face Alignment Network demo script, ICCV 2017')
    cmd:text('Please visit https://www.adrianbulat.com for additional details')
    cmd:text()
    cmd:text('Options:')

    -- Options
    cmd:option('-mode', 'demo', 'Options: demo | generate | eval')
    cmd:option('-type', '2D', 'Options: 2D | 3D | 3D-full')

    -- Models location
    cmd:option('-model', 'models/2D-FAN.t7', 'Path to the t7 model')
    cmd:option('-modelZ', 'models/depth.t7', 'Path to the t7 model for depth prediction')

    -- Data options
    cmd:option('-save', 'true', 'Save predictions in the folder specified as output')
    cmd:option('-detectFaces', 'false', 'Use VJ opencv implementation to detect faces') 
    cmd:option('-input',  'dataset/LS3D-W/', 'Path to the dataset.')
    cmd:option('-output',  'dataset/LS3D-W/out/', 'Path where to save the predictions.')
    cmd:option('-device', 'cuda', 'Options: cpu, gpu')
    cmd:option('-limitedGpuMemory','false', 'if true : swap between loaded 2D/3D models in Gpu to save some space, if 3D-full prediction required')
    cmd:option('-outputFormat', 't7', 'Output format: t7 | txt')

    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.save = opt.save ~= 'false'
    opt.detectFaces = opt.detectFaces ~= 'false'

    if not paths.dirp(opt.input) then
        cmd:error('error: data directory was not found\n')
    end
    if not paths.dirp(opt.output) and not paths.mkdir(opt.output) then
        cmd:error('error: unable to create output directory: '..opt.output..'\n')
    end

    return opt
end

return parse
