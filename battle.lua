require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'lsun-test',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 1,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'improved-mse',
   noise = 'normal',       -- uniform / normal
   criterion = 'mse',
   m1 ='lsun-mse',
   m2 ='lsun-abs',
   testing = 1,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = 2 -- torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

-- M1
local netG1 = nn.Sequential()
local netD1 = nn.Sequential()
-- M2
local netG2 = nn.Sequential()
local netD2 = nn.Sequential()

----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG1, cudnn)
      cudnn.convert(netG2, cudnn)
      cudnn.convert(netD1, cudnn)
      cudnn.convert(netD2, cudnn)
   end
   netD1:cuda();           netG1:cuda();           --criterion:cuda()
   netD2:cuda();           netG2:cuda();           --criterion:cuda()
end


if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- Calculates test ratio
local r_test = function()
   -- test with real data
   local d1_err = 0
   local d2_err = 0
   for i=0,data:size(),opt.batchSize do
   	data_tm:reset(); data_tm:resume()
   	local real = data:getBatch()
	if real == nil then break end
   	data_tm:stop()
   	input:copy(real)
   	cnt1 = netD1:forward(input):lt(.5):sum()
   	input:copy(real)
   	cnt2 = netD2:forward(input):lt(.5):sum()
--	print(('%d \t %d'):format(cnt1,cnt2)) 
   	d1_err = d1_err + cnt1
   	d2_err = d2_err + cnt2
	if i % 6400 == 0 then 
		print(('r_test: %d / %d | %d'):format(d1_err,d2_err,i)) 
		--disp.image(real, {win=opt.display_id, title=opt.name})
	end
   end
   return d1_err / d2_err
end

-- Calculates sample ratio
local r_sample = function()
   -- test with real data
   local d1_err = 0
   local d2_err = 0
   for i=0,data:size(),opt.batchSize do
	if opt.noise == 'uniform' then -- regenerate random noise
	    noise:uniform(-1, 1)
	elseif opt.noise == 'normal' then
	    noise:normal(0, 1)
	end
   	local fake_g1 = netG1:forward(noise)
   	local fake_g2 = netG2:forward(noise)

 	-- Evaluate D1(G2(z)) and D2(G1(z)) 
   	input:copy(fake_g2)
   	d1_err = d1_err + netD1:forward(input):ge(.5):sum()
   	input:copy(fake_g1)
   	d2_err = d2_err + netD2:forward(input):ge(.5):sum()
	if i % 6400 == 0 then 
		print(('r_sample: %d / %d | %d'):format(d1_err,d2_err, i))
		disp.image(fake_g1, {win=opt.display_id, title=opt.m1})
		disp.image(fake_g2, {win=opt.display_id*3, title=opt.m2})
	 end
   end
   return d1_err / d2_err
end

-- test
    print('Battle Result:' .. opt.m1 .. ' vs. ' .. opt.m2)
    print('epoch \t| r_test  \t| r_sample')
for epoch = 1,25,1 do
    print('epoch ' .. epoch)
    netD1 = torch.load('checkpoints/' .. opt.m1 .. '_' .. epoch .. '_net_D.t7') ; netD1:evaluate()
    netG1 = torch.load('checkpoints/' .. opt.m1 .. '_' .. epoch .. '_net_G.t7') ; netG1:evaluate()
    netD2 = torch.load('checkpoints/' .. opt.m2 .. '_' .. epoch .. '_net_D.t7') ; netD2:evaluate()
    netG2 = torch.load('checkpoints/' .. opt.m2 .. '_' .. epoch .. '_net_G.t7') ; netG2:evaluate()

    print(('%s vs %s in %d : r_test = %.3f \t r_sample = %.3f'):format(opt.m1, opt.m2, epoch, r_test(),r_sample()))
end
