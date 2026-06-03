#!/bin/csh -f 

if ( $#argv < 1 ) then
    echo "用法: $0 <文件path> "
    echo " $0 path"
    exit 1
endif


set SRC = $argv[1]
   set file_list = (A143 A70 D33 D106)

echo "处理文件: $file_list"


foreach file ($file_list)
	cd ${SRC}/$file
	set files = (*.llde)
	
	if ($#files > 0) then
	    set first = $files[1]
	    echo "第一个llde文件是: $first"
	else
	    echo "没有找到 llde 文件"
	endif
	ln -s /home/ffan/inv/inversion/myanmar/input/test/${file}/dem.grd 
	ln -s /home/ffan/inv/inversion/myanmar/input/test/${file}/*.PRM 
	ln -s /home/ffan/inv/inversion/myanmar/input/test/${file}/*.LED 
	/home/ffan/inv/inversion/myanmar/input/test/lltenude.sh $first dem.grd >&log&
	cd ../..
end
wait

echo "Done \!\!\!"