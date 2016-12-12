require 'lmdb'
require 'image'
tds=require 'tds'
ffi = require 'ffi'

--list = {'bedroom'}
list = {'church_outdoor'}

-- list = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
--         'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}

root = os.getenv('DATA_ROOT') or os.getenv('HOME') .. '/local/lsun'

for i=1,#list do
   local name = list[i] .. '_train_lmdb'
   print('opening lmdb database: ', name)
   db = lmdb.env{Path=paths.concat(root, name), RDONLY=true}
   db:open()
   reader = db:txn(true)
   cursor = reader:cursor()
   hsh = tds.hash()
   db_size = db:stat().entries

   count = 1
   local cont = true
   while cont do
      local key,data = cursor:get()
      hsh[count] = key
      print('Reading: ', count, '   Key:', key)
      count = count + 1
      if not cursor:next() or count == db_size/2 then
          cont = false
      end
   end

   local train_sz = math.floor(#hsh/2)
   local test_sz = #hsh - train_sz
   hsh_train = torch.CharTensor(train_sz, #hsh[1])
   hsh_test = torch.CharTensor(test_sz, #hsh[1])
   for i=1,train_sz do ffi.copy(hsh_train[i]:data(), hsh[i], #hsh[1]) end
   for i=1,test_sz  do ffi.copy(hsh_test[i]:data(), hsh[i+train_sz], #hsh[1]) end

   local indexfile_train = paths.concat(root, name .. '_hashes_chartensor.t7')
   local indexfile_test = paths.concat(root, name .. '_hashes_chartensor_test.t7')
   torch.save(indexfile_train, hsh_train)
   torch.save(indexfile_test, hsh_test)
   print('wrote index file at: ', indexfile_train .. ' with ' .. train_sz .. ' keys')
   print('wrote index file at: ', indexfile_test .. ' with ' .. test_sz .. ' keys')
end

print("you're all set")
